#!/bin/bash

# ! Specify the dataset and the number of samples
dataset="PACS"
num_sample=5 # 20 in default

# Get the current timestamp
current_time=$(date +"%Y-%m-%d-%H_%M_%S")

# Define the base output directory
base_output_dir="./exp_output"
timestamped_output_dir="${base_output_dir}/${current_time}"

# Run the CLIP evaluation script with the new output directory
python ./evaluation/eval_clip.py --dataset $dataset --output_dir "$timestamped_output_dir" --num_sample $num_sample --save_samples

# Evaluate LLaVA model using CLIP's samples
CUDA_VISIBLE_DEVICES=0,1 python ./evaluation/eval_llava.py --dataset $dataset --continue_dir="$timestamped_output_dir"