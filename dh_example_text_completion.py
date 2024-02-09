# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List
#donghyeon
from torch.profiler import profile, record_function, ProfilerActivity

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is"]
        #is there a way to print out the sequence length for this: 
        #donghyeon: purposefully tailored to process just one sequence. 
    
    '''
    #donghyeon CPU profiler    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print("\n==================================\n")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))
    '''
    '''
    #donghyeon CPU-GPU profiler    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print("\n==================================\n")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))
    print("\n==================================\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("\n==================================\n")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=15))
    '''

    #0208
    '''
    #donghyeon CPU-GPU profiler + stack export and print stat   
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    #print("\n==================================\n")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))
    print("\n==================================\n")
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    #print("\n==================================\n")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=15))
    print("\n==================================\n")
    # Print aggregated stats
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
    print("\n==================================\n")

    # prof.export_chrome_trace("text_completion_trace.json")
    '''
    
    results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    #printing output
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    

if __name__ == "__main__":
    fire.Fire(main)
