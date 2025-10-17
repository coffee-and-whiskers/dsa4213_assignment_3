#!/bin/bash

mkdir -p logs/eval results

CUDA_VISIBLE_DEVICES=3 python model/evaluate_full.py \
    --model_path model/checkpoints/huatuo_full/final \
    --test_data data/huatuo_formatted/huatuo_test.jsonl \
    --output results/eval_huatuo_full.json \
    --batch_size 16 \
    --max_length 512 \
    --lang zh \
    > logs/eval/eval_huatuo_full.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python model/evaluate_full.py \
    --model_path model/checkpoints/aimc_full/final \
    --test_data data/aimc_formatted/aimc_test.jsonl \
    --output results/eval_aimc_full.json \
    --batch_size 16 \
    --max_length 512 \
    --lang en \
    > logs/eval/eval_aimc_full.log 2>&1 &
