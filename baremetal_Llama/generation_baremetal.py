import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from baremetal_Llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer 

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, disk_offload

from torch.cuda import nvtx

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

#donghyeon: where Llama that we use is defined. 
class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        #if not torch.distributed.is_initialized():
        #    torch.distributed.init_process_group("nccl")
        #if not model_parallel_is_initialized():
        #    if model_parallel_size is None:
        #        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        #    initialize_model_parallel(model_parallel_size)

        #local_rank = int(os.environ.get("LOCAL_RANK", 0))
        #torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        
        #naive 
        start_time = time.time()
        nvtx.range_push("torch.load checkpoint")
        checkpoint = torch.load(Path(ckpt_dir)/ "consolidated.00.pth", map_location="cpu")
        nvtx.range_pop()
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor) # this must cause model to be initialized to GPU
        nvtx.range_push('initialize model')
        model = Transformer(model_args)
        nvtx.range_pop()
        nvtx.range_push('load state dict')
        #os.kill(os.getpid(), signal.SIGUSR1)
        model.load_state_dict(checkpoint, strict=False)
        #os.kill(os.getpid(), signal.SIGUSR1)
        nvtx.range_pop()
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        
        '''
        #offload
        start_time = time.time()
        #checkpoint = "./llama-2-7b/consolidated.00.pth"
        checkpoint = "./offload/param_no_ropefreq/consolidated.00.pth"
        
        #for key in checkpoint.keys():         
        #    print(f"{key}: {checkpoint[key].size()}")
        
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor) # this must cause model to be initialized to GPU
        
        with init_empty_weights():
            model = Transformer(model_args)
        device_map = torch.load("./offload/device_map_5050_RMS.pth")
        
        model = load_checkpoint_and_dispatch(
            model, checkpoint=checkpoint, device_map=device_map, offload_folder="./offload/space", 
            offload_state_dict=True,
            no_split_module_classes=['TransformerBlock','RMSNorm', 'Embedding', 'Linear'] 
            #make sure 1)Layer-wise load, 2)No residual broken
            )
            #model, checkpoint=checkpoint, device_map='auto', offload_state_dict=True, dtype='float16' ) 
            #possible float32 to float16 overhead at multiple times (and to cuda.HalfTensor?)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        '''
        return Llama(model, tokenizer)
        #'''
        
        
        
        
        
        
        
        #checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        #checkpoints_hf = [str(path) for path in Path(ckpt_dir).glob("*.pth")]
        #print('ckpt_dir: ', ckpt_dir)
        #print('checkpoints: ', checkpoints)
        #print('checkpoints_hf: ', checkpoints_hf)
        #assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        #assert model_parallel_size == len(
        #    checkpoints
        #), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        #ckpt_path = checkpoints[get_model_parallel_rank()]
        
        
        

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        #donghyeon model defined here 
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    #donghyeon: Thus the actual 'processing' is done here 
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens) #the first dimension, the batch size. 
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        #determine the forward pass parameter for each batch. (total_len)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        #looped forward pass for each token, (batched)
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        #endor_total_activations = [] #donghyeon_endor
        if min_prompt_len == total_len: 
            print("_------------------------------I don't think this is visited")
            #logits, endor_prompt_activations = self.model.forward(tokens, prev_pos) #donghyeon_endor
            logits = self.model.forward(tokens, prev_pos) #donghyeon_endor
            #endor_total_activations.append(endor_prompt_activations) #donghyeon_endor
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        #nvtx.range_push('Prompt + Decoding stage')
        for cur_pos in range(min_prompt_len, total_len): #here is where cur_pos is incremented.
            #print("------------------------")
            print("this pass using token indices:", prev_pos, "~", cur_pos)
            #print(tokens)
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos) #donghyeon_endor
            #logits, endor_single_token_activations = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos) #donghyeon_endor
            #endor_total_activations.append(endor_single_token_activations) #donghyeon_endor
            #forward params: Input Tensor and start_pos 
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                print('-----EOS token has been reached!')
                break
        #nvtx.range_pop()
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None) #, endor_total_activations) #donghyeon_endor

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        print("-----max_gen_length: ", max_gen_len)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        print("length of the input sequence is:", len(prompt_tokens[0]))
        #generation_tokens, generation_logprobs, endor_activations_final = self.generate( #donghyeon_endor
        generation_tokens, generation_logprobs = self.generate( #donghyeon_endor
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        #file_path = './profile_log/activation_2.pth'#donghyeon_endor
        #torch.save(endor_activations_final, file_path)
        #print('file saved')
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        print("length of the output sequence is: ", len(generation_tokens[0]))
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token