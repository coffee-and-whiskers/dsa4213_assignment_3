#!/bin/bash

mkdir -p logs/eval results

CUDA_VISIBLE_DEVICES=3 python model/evaluate_lora.py \
    --model_path model/checkpoints/huatuo_lora_gs/huatuo_gs_1_lr5e-04_r16_alpha64_do0.05/final \
    --test_data data/huatuo_formatted/huatuo_test.jsonl \
    --output results/eval_huatuo_gs_1.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang zh \
    > logs/eval/eval_huatuo_gs_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python model/evaluate_lora.py \
    --model_path model/checkpoints/huatuo_lora_gs/huatuo_gs_2_lr5e-04_r24_alpha96_do0.05/final \
    --test_data data/huatuo_formatted/huatuo_test.jsonl \
    --output results/eval_huatuo_gs_2.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang zh \
    > logs/eval/eval_huatuo_gs_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python model/evaluate_lora.py \
    --model_path model/checkpoints/huatuo_lora_gs/huatuo_gs_3_lr3e-04_r16_alpha64_do0.05/final \
    --test_data data/huatuo_formatted/huatuo_test.jsonl \
    --output results/eval_huatuo_gs_3.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang zh \
    > logs/eval/eval_huatuo_gs_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python model/evaluate_lora.py \
    --model_path model/checkpoints/huatuo_lora_gs/huatuo_gs_4_lr5e-04_r8_alpha32_do0.05/final \
    --test_data data/huatuo_formatted/huatuo_test.jsonl \
    --output results/eval_huatuo_gs_4.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang zh \
    > logs/eval/eval_huatuo_gs_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python model/evaluate_lora.py \
    --model_path model/checkpoints/aimc_lora_gs/aimc_gs_1_lr2.0e-04_r16_alpha64_do0.1/final \
    --test_data data/aimc_formatted/aimc_test.jsonl \
    --output results/eval_aimc_gs_1.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang en \
    > logs/eval/eval_aimc_gs_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python model/evaluate_lora.py \
    --model_path model/checkpoints/aimc_lora_gs/aimc_gs_2_lr1.5e-04_r12_alpha48_do0.1/final \
    --test_data data/aimc_formatted/aimc_test.jsonl \
    --output results/eval_aimc_gs_2.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang en \
    > logs/eval/eval_aimc_gs_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python model/evaluate_lora.py \
    --model_path model/checkpoints/aimc_lora_gs/aimc_gs_3_lr2.0e-04_r12_alpha48_do0.05/final \
    --test_data data/aimc_formatted/aimc_test.jsonl \
    --output results/eval_aimc_gs_3.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang en \
    > logs/eval/eval_aimc_gs_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python model/evaluate_lora.py \
    --model_path model/checkpoints/aimc_lora_gs/aimc_gs_4_lr2.0e-04_r12_alpha48_do0.15/final \
    --test_data data/aimc_formatted/aimc_test.jsonl \
    --output results/eval_aimc_gs_4.json \
    --base_model google/mt5-base \
    --use_lora \
    --batch_size 16 \
    --max_length 512 \
    --lang en \
    > logs/eval/eval_aimc_gs_4.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python model/evaluate_lora.py \
#     --model_path model/checkpoints/huatuo_lora_baseline/final \
#     --test_data data/huatuo_formatted/huatuo_test.jsonl \
#     --output results/eval_huatuo_baseline_lora.json \
#     --base_model google/mt5-base \
#     --use_lora \
#     --batch_size 16 \
#     --max_length 512 \
#     --lang zh \
#     > logs/eval/eval_huatuo_baseline.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 python model/evaluate_lora.py \
#     --model_path model/checkpoints/aimc_lora_baseline/final \
#     --test_data data/aimc_formatted/aimc_test.jsonl \
#     --output results/eval_aimc_baseline_lora.json \
#     --base_model google/mt5-base \
#     --use_lora \
#     --batch_size 16 \
#     --max_length 512 \
#     --lang en \
#     > logs/eval/eval_aimc_baseline.log 2>&1 &
