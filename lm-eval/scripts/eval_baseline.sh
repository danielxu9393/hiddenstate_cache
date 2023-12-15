#!/bin/bash
task=$1
shots=5
python -u evaluate_model.py \
  --output-file runs/${task}/${shots}-kv.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --recent-size 5 \
  --use-kv


cache_type=half
python -u evaluate_model.py \
  --output-file runs/${task}/${shots}-${cache_type}-baseline.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache \
  --baseline

cache_type=3
python -u evaluate_model.py \
  --output-file runs/${task}/${shots}-${cache_type}-baseline.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache \
  --baseline

cache_type=4
python -u evaluate_model.py \
  --output-file runs/${task}/${shots}-${cache_type}-baseline.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache \
  --baseline

