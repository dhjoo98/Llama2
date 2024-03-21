# single python file inference flow
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, disk_offload

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
model_name = "meta-llama/Llama-2-7b-hf"
model_save_path = "/home/dhjoo/Workspace/llama2_nsight/Llama2/hf_transformer_offload/weights_2"#/hf_weight_7b.pth"
    #straight download from huggingface repo
#'''
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print("done")
device_map = torch.load("./hf_transformer_offload/device_map/device_map_5050.pth")
for i in range(0,32):
    key = "model.layers."+str(i)+".self_attn.rotary_emb.inv_freq"
    if i < 16:
        device_map[key] = 'cpu'
    else:    
        device_map[key] = 'disk'
print(device_map)
#model = load_checkpoint_and_dispatch(
#            model, checkpoint=model_save_path, device_map=device_map, offload_folder="./hf_transformer_offload/offload", 
#            offload_state_dict=True,
#            no_split_module_classes=["LlamaRMSNorm","LlamaDecoderLayer"]
#        )   


#processing input sentence 
input_tokens = tokenizer(squad_data[0])
print(input_tokens)
input_tensor = torch.tensor(input_tokens.input_ids).to(0)
input_mask = torch.tensor(input_tokens.attention_mask).to(0)

#this is enough for HF Llama!
generate_ids = model.generate(input_ids = input_tensor, attention_mask = input_mask, max_length = 90)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Otuput:", output)

print(model.state_dict)
print("donedone")
#checkpoint = model.state_dict()
#for key in checkpoint.keys():         
#            print(f"{key}: {checkpoint[key].size()}")
print("donedonedone")
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