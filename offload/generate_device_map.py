import torch 

state_dict = torch.load("/home/dhjoo/Workspace/llama2_nsight/Llama2/llama-2-7b/consolidated.00.pth")
device_map = {}
for key in state_dict.keys():
    parsed_key = key.split('.')
    index = parsed_key[1]
    if index == "freqs":
        continue
        #device_map[key] = 0
    elif index == "weight":
        #device_map[key] = "cpu"
        device_map[key] = 0
    elif int(index) < 16:
        device_map[key] = "cpu"
    else:
        device_map[key] = "disk"

torch.save(device_map,"/home/dhjoo/Workspace/llama2_nsight/Llama2/offload/device_map_5050_RMS.pth")