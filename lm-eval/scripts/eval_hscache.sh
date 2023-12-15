#!/bin/bash
task=$1
cache_type=full
python -u evaluate_model.py \
  --output-file runs/${task}-${shots}-${cache_type}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache

cache_type=half
python -u evaluate_model.py \
  --output-file runs/${task}-${shots}-${cache_type}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache

cache_type=none
python -u evaluate_model.py \
  --output-file runs/${task}-${shots}-${cache_type}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --recent-size 5

cache_type=recent
python -u evaluate_model.py \
  --output-file runs/${task}-${shots}-${cache_type}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache

cache_type=3
python -u evaluate_model.py \
  --output-file runs/${task}-${shots}-${cache_type}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache

cache_type=4
python -u evaluate_model.py \
  --output-file runs/${task}-${shots}-${cache_type}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --cache-type ${cache_type} \
  --recent-size 5 \
  --enable-hscache