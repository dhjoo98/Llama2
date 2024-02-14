#!/bin/bash

#SBATCH -J run_llama2
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32000

# trash bin
# SBATCH -o ./profile_log/text_completion_.out
# SBATCH -o ./profile_log/text_completion_"$log_name".out
# SBATCH -o ./profile_log/run_llama2_dh_text_completion_CUDA_stack_0207.out

output_file="./profile_log/nsys/text_completion_$1.out"

# for nsight systems 
#~/nsight-systems-2024.1.1/bin/nsys profile --trace=cuda,nvtx,cudnn,cublas --capture-range=nvtx --gpu-metrics-device=help -output="$output_file" \
torchrun --nproc_per_node 1 dh_example_text_completion.py \
    --ckpt_dir llama-2-7b \
    --tokenizer_path tokenizer.model \
    #--max_seq_len 4096 \
		#--max_gen_len 512 \
    #--max_batch_size 6 \
		--temperature 0.0;


#todo: implement extraction with PyTorch hook.
