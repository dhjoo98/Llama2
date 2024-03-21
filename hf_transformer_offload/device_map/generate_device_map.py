from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, disk_offload

from torch.cuda import nvtx

#import json 

model_name = "meta-llama/Llama-2-7b-hf"
model_save_path = "/home/dhjoo/Workspace/llama2_nsight/Llama2/hf_transformer_offload/weights_2"#/hf_weight_7b.pth"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print("done")
model = load_checkpoint_and_dispatch(
            model, checkpoint=model_save_path, device_map="auto", #offload_folder="./offload/space", 
            offload_state_dict=True,
        )   
print("donedone")
checkpoint = model.state_dict()

device_map = {}
for key in checkpoint.keys():
    if key == "lm_head.weight":
        device_map[key] = 0
    else:
        parsed_key = key.split('.')
        index = parsed_key[2]
        if index == "freqs": #not in hf llama
            continue
            #device_map[key] = 0
        elif index == "weight":
            #device_map[key] = "cpu"
            device_map[key] = 0
        elif int(index) < 16:
            device_map[key] = "cpu"
        else:
            device_map[key] = "disk"

torch.save(device_map,"/home/dhjoo/Workspace/llama2_nsight/Llama2/hf_transformer_offload/device_map/device_map_5050.pth")