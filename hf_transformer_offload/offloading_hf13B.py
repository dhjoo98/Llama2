# single python file inference flow
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

from accelerate import init_empty_weights,  disk_offload, infer_auto_device_map
#0319 two points of monkeypatcg nvtxannotating: load_checkpoint_and_dispatch and generate, at below. 

from mp.big_modeling import load_checkpoint_and_dispatch
# L_C_A_D() > utils.modeling.load_ckpt_in_model (device branchs) [L. 1520]
#    > utils.modeling.load_state_dict() [L.1376]
#          specifies device_map: (even disks are to mapped to cpu (first?).)
#          safetensor.safe_open() for each device's weightList.> basically load the actual weight per device. and return it.
#    from this forward, is I assume, the actual disk offloading part.
#    won't take the 8bit quantize branch 
#        for disk and cpu, both set_module_tensor_to_device > offload_model [L.1685]
#        for else, just set_module_tensor_to_device
#    > utils.modeling.set_module_tensor_to_device()
        #so what does this do? (for both cpu and dick, param_device: 'meta')
#       official description:A helper function to set a given tensor (parameter of buffer) of a module on a specific device (***********note that doing
#    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).
#           ''
#    > utils.offload.offload_weight() > np.memmap (both are offloaded first to CPU)
# > dispatch() itself is verry short. 

#figure out how runtime generate is done, right before degbeling into cuda APIs
#try out blktrace sent to hana Ram, Ba

from torch.cuda import nvtx

import json 

json_file_path = './datasets/squadv2/concatenated_300_sentences.json'

with open(json_file_path, 'r') as input_texts:
    squad_data = json.load(input_texts)

#print(len(squad_data))
print(len(squad_data[0]))
print(squad_data[0])
#note huggingface's example
'''
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="disk",
)

sequences = pipeline(
    squad_data[0],
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")



#do I really want to look at hf Llama source rn? 
'''
#checkpoint = "~/Workspace/llama2_nsight/Llama2/llama-2-7b"  # /consolidated.00.pth"
model_name = "meta-llama/Llama-2-13b-hf"
model_save_path = "/home/dhjoo/Workspace/llama2_nsight/Llama2/hf_transformer_offload/weights_13B"#/hf_weight_7b.pth"
    #straight download from huggingface repo
#'''
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
nvtx.range_push('init empty')
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
nvtx.range_pop()
nvtx.range_push('infer DM')
device_map = infer_auto_device_map(model, 
                                   max_memory={0:'13GB','cpu':'45GB'},
                                   no_split_module_classes=["LlamaRMSNorm","LlamaDecoderLayer"],
                                    dtype="float16"
                                   )
nvtx.range_pop()
print(device_map)
device_map["model.layers.21"] = "disk"
device_map["model.layers.22"] = "disk"
device_map["model.layers.23"] = "disk"
device_map["model.layers.24"] = "disk"
device_map["model.layers.25"] = "disk"
device_map["model.layers.26"] = "disk"
device_map["model.layers.27"] = "disk"
device_map["model.layers.28"] = "disk"
device_map["model.layers.29"] = "disk"
device_map["model.layers.30"] = "disk"
device_map["model.layers.31"] = "disk"
device_map["model.layers.32"] = "disk"
device_map["model.layers.33"] = "disk"
device_map["model.layers.34"] = "disk"
device_map["model.layers.35"] = "disk"
device_map["model.layers.36"] = "disk"
device_map["model.layers.37"] = "disk"
device_map["model.layers.38"] = "disk"
device_map["model.layers.39"] = "disk"


#thank you, https://huggingface.co/blog/accelerate-large-models
#print(device_map)
#note that, model on meta device is only needed to derive device, map.
#del model 
#afterwards, we can use naive execution 
#But as this is fuckary,
#model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_save_path, device_map = device_map,
#                                             offload_folder="./hf_transformer_offload/offload", 
#                                            offload_state_dict=True,
#                                            torch_dtype=torch.float16
#                                            )

'''
device_map = torch.load("./hf_transformer_offload/device_map/device_map_5050.pth")
for i in range(0,32):
    key = "model.layers."+str(i)+".self_attn.rotary_emb.inv_freq"
    device_map[key] = 0
print(device_map)
'''
nvtx.range_push('load and dispatch')
model = load_checkpoint_and_dispatch(
            model, checkpoint=model_save_path, 
            device_map=device_map,  #gives us killed now, even with "auto"
            offload_folder="./hf_transformer_offload/offload", 
            offload_state_dict=True,
            dtype="float16")
            #no_split_module_classes=["LlamaRMSNorm","LlamaDecoderLayer"])   
nvtx.range_pop()
model.tie_weights() #from issue 2520 

input_sequence = "Hey LLM, if I give you this"
#input_tokens = tokenizer(["Here's a smaller sequence"])
input_tokens = tokenizer([input_sequence])
#input_tokens = tokenizer(squad_data[0])
print("Input sequence: ", input_sequence,"\n", "Length: ", len(input_tokens[0]))
#print(input_tokens)
input_tensor = torch.tensor(input_tokens.input_ids).to(0)
input_mask = torch.tensor(input_tokens.attention_mask).to(0)

#this is enough for HF Llama!
nvtx.range_push('Generation')




#so how does this simple function, call weights scattered across the system? 





generate_ids = model.generate(input_ids = input_tensor, attention_mask = input_mask, max_length = 20)
nvtx.range_pop()
#why does this bottleneck so much?
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Output sequence: ", output, "\n", "Output Length: ", len(generate_ids[0]))

#print(model.state_dict)
#checkpoint = model.state_dict()
#for key in checkpoint.keys():         
#            print(f"{key}: {checkpoint[key].size()}")
'''
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#with init_empty_weights():
#    model = LlamaModelForCausalLM() #, cache_dir=custom_cache_dir)
#model = load_checkpoint_and_dispatch(
#    model, device_map="auto"
#)
print("model loaded")
#print(model.state_dict())
torch.save(model.state_dict(), model_save_path)
print("weight saved")
#with init_empty_weights():
'''
'''
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("tokenizer complete")
config = AutoConfig.from_pretrained(model_name)
print("auto config")
model = AutoModelForCausalLM.from_config(config)
print("loading state_dict")
state_dict = torch.load(model_save_path)
print("loading state_dict to model")
model.load_state_dict(state_dict)
print('fuck yeahhhhh')
'''