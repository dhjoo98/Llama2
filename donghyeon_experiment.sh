#!/bin/bash 

source ../../llama2_env1/bin/activate

output_file="./profile_log/text_completion_$1.out"

sbatch -o "$output_file" donghyeon_run.sh 

watch -n 1 squeue -u dhjoo98

deactivate