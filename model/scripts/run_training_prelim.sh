#!/bin/bash

cd /workspace/nikerlas/llm

mkdir -p logs/train

echo "starting full fine-tuning on aimc dataset (gpu 3)"
CUDA_VISIBLE_DEVICES=3 nohup python3 model/train_full.py data/aimc_formatted/aimc_train.jsonl model/checkpoints/aimc_full > logs/train/aimc_full.log 2>&1 &
AIMC_FULL_PID=$!

echo "starting lora fine-tuning on aimc dataset (gpu 3)"
CUDA_VISIBLE_DEVICES=3 nohup python3 model/train_lora.py data/aimc_formatted/aimc_train.jsonl model/checkpoints/aimc_lora_baseline > logs/train/aimc_lora_baseline.log 2>&1 &
AIMC_LORA_PID=$!

echo "starting full fine-tuning on huatuo dataset (gpu 4)"
CUDA_VISIBLE_DEVICES=4 nohup python3 model/train_full.py data/huatuo_formatted/huatuo_train.jsonl model/checkpoints/huatuo_full > logs/train/huatuo_full.log 2>&1 &
HUATUO_FULL_PID=$!

echo "starting lora fine-tuning on huatuo dataset (gpu 4)"
CUDA_VISIBLE_DEVICES=4 nohup python3 model/train_lora.py data/huatuo_formatted/huatuo_train.jsonl model/checkpoints/huatuo_lora_baseline > logs/train/huatuo_lora_baseline.log 2>&1 &
HUATUO_LORA_PID=$!

echo "all training jobs started"
echo "aimc full (gpu 3): $AIMC_FULL_PID"
echo "aimc lora (gpu 3): $AIMC_LORA_PID"
echo "huatuo full (gpu 4): $HUATUO_FULL_PID"
echo "huatuo lora (gpu 4): $HUATUO_LORA_PID"

echo "monitor logs with: tail -f logs/train/*.log"
