#!/bin/bash
num_tokens=$1
cache_size=$2
rank=$3
dataset_name=pg19
task=None

cache_type=full
python -u eval_long_ppl_hidden_state_cache.py \
  --ppl-file ppl/${dataset_name}/${cache_type}_${rank}.txt \
  --log-file log/${dataset_name}/${cache_type}_${rank}.txt \
  --task ${task} \
  --dataset_name ${dataset_name} \
  --cache-type ${cache_type} \
  --cache-size ${cache_size} \
  --enable-hscache \
  --enable-lowrank \
  --rank ${rank} \
  --num-eval-tokens ${num_tokens}