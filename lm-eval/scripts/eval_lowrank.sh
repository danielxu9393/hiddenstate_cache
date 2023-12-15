#!/bin/bash
task=$1
shots=5
i_list=(1024 1536 2048 3000 3500 3750)
# for ((i=12; i>=7; i--)); do
for number in "${i_list[@]}"; do
    cache_type=full
    # rank=$((2**$i))
    rank=$number
    python -u evaluate_model.py \
    --output-file runs/${task}_qkv/${shots}-${cache_type}-${rank}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --cache-type ${cache_type} \
    --recent-size 5 \
    --enable-hscache \
    --enable-lowrank \
    --rank ${rank}
done

python -u evaluate_model.py \
  --output-file runs/${task}_qkv/${shots}-kv.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --recent-size 5 \
  --use-kv


