import torch 
import torch.nn as nn


class test_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_size = 100000
        self.out_size = 10000
    def forward(self, input_mat):
        attention = nn.Linear(self.in_size, self.out_size, device=0) #device mapping here is necessary.
        output = attention(input_mat)
        del attention
        return output 


'''
import torch 
from bearmetal_Llama.test import test_layer
model = test_layer()
model.to(0) # does increase VRAM usage
inp = torch.ones(100000) #doesn't increase DRAM, possibly an optimization 
inp.to(0) # does increase VRAM usage

model.forward(inp) # does increase VRAM usage, which means that linear layer is initialized here.
'''

'''
so.. 
del model 
del inp
torch.cuda.empty_cache() 

doesn't work free VRAM space. Only when VScode itself is turned off

'''

'''
actually cleaning VRAM is a hardtask indeed. 


although, 
    doing a 
    inp = torch.rand(10000, device=0)
    model = test_layer()
    model.forward(inp)
        uses GPU (even if no model.to(0))

tomorrow morning, try with a llama from hf. 
'''


'''
source codes to refernece. 
https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py#L277 
https://github.com/huggingface/accelerate/issues/1629 
https://github.com/abetlen/llama-cpp-python/issues/223
https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling 
https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference 
https://huggingface.co/docs/accelerate/v0.27.2/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch

if it goes south
deepspeed: 
https://www.deepspeed.ai/2022/09/09/zero-inference.html#when-to-use-zero-inference 
https://github.com/microsoft/DeepSpeed 

'''