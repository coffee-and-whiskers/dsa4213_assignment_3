"""
Grid Search for Huatuo LoRA Training
Based on analysis of 10-epoch baseline:
- Final train_loss: 3.224, eval_loss: 2.707
- Train loss plateaued at ~3.2 from epoch 5 onward (steps ~7,500+)
- Eval loss continued decreasing throughout training (unusual pattern indicating underfitting)
- Learning rate: 3e-4 with cosine decay, warmup_ratio: 0.1
- Current config: r=8, alpha=32, dropout=0.1, weight_decay=0.01, batch_size=8

Key Observations:
1. Train loss plateau at 3.2 indicates insufficient model capacity to fit training data
2. Eval loss lower than train loss (2.71 < 3.22) confirms severe underfitting
3. Cosine LR decay reduced optimization power during plateau phase
4. No signs of overfitting - eval loss never increased

Strategy:
1. Increase LoRA rank (r=16, r=24) to provide greater model capacity
2. Higher learning rate (5e-4) to maintain optimization power through middle epochs
3. Reduce regularization (dropout=0.05, weight_decay=0.001) since underfitting, not overfitting
4. Target goal: train_loss < 2.5 
"""

import os
import sys
import json
import itertools
from datetime import datetime
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_lora import train_lora


def run_grid_search():
    # Huatuo-specific grid search parameters
    # Loss plateaued at ~3.2, suggesting we need more capacity and better optimization

    param_grid = {
        'learning_rate': [
            5e-4,   # Higher - might escape plateau faster
            3e-4,   # Baseline
            2e-4,   # Lower - more stable convergence
        ],
        'lora_r': [
            16,     # Double capacity - combat underfitting
            8,      # Baseline
            24,     # Higher capacity exploration
        ],
        'lora_alpha': [
            32,     # Baseline (alpha=r*4)
            64,     # Higher scaling with r=16
            16,     # Lower - less aggressive adaptation
        ],
        'lora_dropout': [
            0.05,   # Less dropout - loss is high, may be over-regularizing
            0.1,    # Baseline
            0.15,   # More dropout - try better generalization
        ],
        'batch_size': [
            8,      # Baseline
            16,     # Larger batch - more stable gradients
        ],
        'warmup_ratio': [
            0.05,   # Shorter warmup
            0.1,    # Baseline
            0.15,   # Longer warmup for stability
        ],
        'weight_decay': [
            0.01,   # Baseline
            0.001,  # Less regularization - loss is already high
            0.05,   # More regularization
        ],
    }

    # Top 4 configurations targeting the underfitting problem
    # Strategy: More capacity (r=16,24) + Higher LR (5e-4) + Less regularization (dropout=0.05, wd=0.001)
    priority_configs = [
        # 1. BEST BET: Combined optimization (high cap + high LR + less reg)
        {'learning_rate': 5e-4, 'lora_r': 16, 'lora_alpha': 64, 'lora_dropout': 0.05, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.001},

        # 2. Maximum capacity test
        {'learning_rate': 5e-4, 'lora_r': 24, 'lora_alpha': 96, 'lora_dropout': 0.05, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.001},

        # 3. Conservative improvement (moderate LR + good capacity)
        {'learning_rate': 3e-4, 'lora_r': 16, 'lora_alpha': 64, 'lora_dropout': 0.05, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.001},

        # 4. Isolate LR effect (baseline capacity, test if LR alone helps)
        {'learning_rate': 5e-4, 'lora_r': 8, 'lora_alpha': 32, 'lora_dropout': 0.05, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.001},
    ]

    train_jsonl_path = 'data/huatuo_formatted/huatuo_train.jsonl'
    base_output_dir = 'model/checkpoints/huatuo_lora_gs'
    os.makedirs(base_output_dir, exist_ok=True)

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Starting Huatuo LoRA Grid Search with {len(priority_configs)} priority configurations")
    print(f"Baseline: train_loss=3.628, plateaued at ~3.2")
    print(f"Strategy: Increase capacity, optimize learning rate, reduce over-regularization\n")

    for idx, config in enumerate(priority_configs, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{len(priority_configs)}")
        print(f"{'='*80}")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Create run-specific output directory
        run_name = f"huatuo_gs_{idx}_lr{config['learning_rate']:.0e}_r{config['lora_r']}_alpha{config['lora_alpha']}_do{config['lora_dropout']}"
        output_dir = os.path.join(base_output_dir, run_name)

        try:
            # Train with 6 epochs for grid search (vs 8 baseline)
            train_lora(
                train_jsonl_path=train_jsonl_path,
                output_dir=output_dir,
                batch_size=config['batch_size'],
                epochs=8,
                learning_rate=config['learning_rate'],
                lora_r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                warmup_ratio=config['warmup_ratio'],
                weight_decay=config['weight_decay'],
                use_wandb=True,
                wandb_project='huatuo-lora-gs',
                wandb_run_name=run_name
            )

            # Read final results from wandb or training logs
            result = {
                'config': config,
                'run_name': run_name,
                'status': 'completed',
                'output_dir': output_dir
            }

            print(f"\n✓ Configuration {idx} completed successfully")

        except Exception as e:
            print(f"\n✗ Configuration {idx} failed: {str(e)}")
            result = {
                'config': config,
                'run_name': run_name,
                'status': 'failed',
                'error': str(e)
            }

        results.append(result)

        # Save intermediate results
        results_file = os.path.join(base_output_dir, f'grid_search_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Grid Search Complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")

    # Summary
    completed = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nSummary:")
    print(f"  Completed: {len(completed)}/{len(priority_configs)}")
    print(f"  Failed: {len(failed)}/{len(priority_configs)}")

    return results


if __name__ == "__main__":
    results = run_grid_search()
