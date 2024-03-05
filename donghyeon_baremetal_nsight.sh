#!/bin/bash 

source activate llama22

python baremetal_example_text_completion.py    --ckpt_dir llama-2-7b     --tokenizer_path tokenizer.model --temperature 0.0
#python dh_example_text_completion.py    --ckpt_dir llama-2-7b     --tokenizer_path tokenizer.model --temperature 0.0


source deactivate 