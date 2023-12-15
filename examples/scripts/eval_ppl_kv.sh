#!/bin/bash
num_tokens=$1
cache_size=$2
dataset_name=pg19
task=None

python -u eval_long_ppl_hidden_state_cache.py \
  --ppl-file ppl/${dataset_name}/kv.txt \
  --log-file log/${dataset_name}/kv.txt \
  --task ${task} \
  --dataset_name ${dataset_name} \
  --use-kv \
  --cache-size ${cache_size} \
  --num-eval-tokens ${num_tokens}