#!/bin/bash

mkdir -p logs/eval results

CUDA_VISIBLE_DEVICES=2 python model/evaluate_lora.py \
    --model_path model/checkpoints/huatuo_lora_baseline/final \
    --test_data data/huatuo_formatted/huatuo_test.jsonl \
    --output results/eval_huatuo_baseline_lora.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang zh \
    > logs/eval/eval_huatuo_baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python model/evaluate_lora.py \
    --model_path model/checkpoints/aimc_lora_baseline/final \
    --test_data data/aimc_formatted/aimc_test.jsonl \
    --output results/eval_aimc_baseline_lora.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang en \
    > logs/eval/eval_aimc_baseline.log 2>&1 &
