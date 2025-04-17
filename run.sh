#!/bin/bash

# Define the output file
output="e5_squad"
TRUE=false
# Run the Python script in the background with nohup
nohup python src/main.py \
  --mode both \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4 \
  --output "$output" \
  --dataset squad \
  --patience 2 \
  --accumulation_steps 2 \
  --top_k 0 \
  --eval_steps 908\
  --max_top_k 0  > "${output}.log" 2>&1 &
