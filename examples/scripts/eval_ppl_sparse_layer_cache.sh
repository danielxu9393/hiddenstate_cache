#!/bin/bash
num_tokens=$1
cache_size=$2
recent_size=$3
dataset_name=pg19
task=None

cache_type=half
python -u eval_long_ppl_hidden_state_cache.py \
  --ppl-file ppl/${dataset_name}/${cache_type}.txt \
  --log-file log/${dataset_name}/${cache_type}.txt \
  --task ${task} \
  --dataset_name ${dataset_name} \
  --cache-type ${cache_type} \
  --cache-size ${cache_size} \
  --recent-size ${recent_size} \
  --enable-hscache \
  --num-eval-tokens ${num_tokens}

python -u eval_long_ppl_hidden_state_cache.py \
  --ppl-file ppl/${dataset_name}/kv.txt \
  --log-file log/${dataset_name}/kv.txt \
  --task ${task} \
  --dataset_name ${dataset_name} \
  --use-kv \
  --cache-size ${cache_size} \
  --num-eval-tokens ${num_tokens}

cache_type=full
python -u eval_long_ppl_hidden_state_cache.py \
  --ppl-file ppl/${dataset_name}/${cache_type}.txt \
  --log-file log/${dataset_name}/${cache_type}.txt \
  --task ${task} \
  --dataset_name ${dataset_name} \
  --cache-type ${cache_type} \
  --cache-size ${cache_size} \
  --recent-size ${recent_size} \
  --enable-hscache \
  --num-eval-tokens ${num_tokens}

cache_type=3
python -u eval_long_ppl_hidden_state_cache.py \
  --ppl-file ppl/${dataset_name}/${cache_type}.txt \
  --log-file log/${dataset_name}/${cache_type}.txt \
  --task ${task} \
  --dataset_name ${dataset_name} \
  --cache-type ${cache_type} \
  --cache-size ${cache_size} \
  --recent-size ${recent_size} \
  --enable-hscache \
  --num-eval-tokens ${num_tokens}
