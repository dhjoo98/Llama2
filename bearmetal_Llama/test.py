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

