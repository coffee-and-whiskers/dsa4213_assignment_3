#!/bin/bash

# Create output directories
mkdir -p results/qualitative logs/qualitative

# Run AIMC qualitative generation on GPU 2 (most free memory)
echo "Starting AIMC qualitative generation..."
CUDA_VISIBLE_DEVICES=2 python model/generate_qualitative_aimc.py \
    --questions_path data/qualitative_questions_aimc.json \
    --output_dir results/qualitative \
    --max_length 512 \
    --temperatures 0.7 1.0 1.3 \
    > logs/qualitative/aimc_generation.log 2>&1

echo "AIMC generation completed. Check logs/qualitative/aimc_generation.log for details."

# Run Huatuo qualitative generation on GPU 2
echo "Starting Huatuo qualitative generation..."
CUDA_VISIBLE_DEVICES=2 python model/generate_qualitative_huatuo.py \
    --questions_path data/qualitative_questions_huatuo.json \
    --output_dir results/qualitative \
    --max_length 512 \
    --temperatures 0.7 1.0 1.3 \
    > logs/qualitative/huatuo_generation.log 2>&1

echo "Huatuo generation completed. Check logs/qualitative/huatuo_generation.log for details."
echo ""
echo "All qualitative generation completed!"
echo "Results saved to results/qualitative/"
echo "Logs saved to logs/qualitative/"
