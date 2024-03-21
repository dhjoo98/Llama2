#!/bin/bash 
#don't forget chmod +x to make this executables
source activate llama22

python ./hf_transformer_offload/offloading_hf13B.py    

source deactivate 