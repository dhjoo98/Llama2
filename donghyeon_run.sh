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

torchrun --nproc_per_node 1 dh_example_text_completion.py \
    --ckpt_dir llama-2-7b \
    --tokenizer_path tokenizer.model \
    --max_seq_len 4096 \
		--max_gen_len 512 \
    --max_batch_size 6 \
		--temperature 0.0;
