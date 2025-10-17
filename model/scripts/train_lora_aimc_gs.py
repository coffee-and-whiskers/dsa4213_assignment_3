"""
Grid Search for AIMC LoRA Training
Based on analysis of 10-epoch baseline:
- Final train_loss: 2.016, eval_loss: 1.756
- Train loss shows high variance/oscillations from epoch 4 onward (range 1.91-2.20)
- Eval loss decreases smoothly throughout training (2.7 → 1.76)
- Eval loss lower than train loss indicates mild underfitting
- Learning rate: 3e-4 with cosine decay, warmup_ratio: 0.1
- Current config: r=8, alpha=32, dropout=0.1, weight_decay=0.01, batch_size=8

Key Observations:
1. Train loss oscillates heavily around 2.0-2.2, unlike Huatuo's smooth plateau
2. Much better absolute performance than Huatuo (35% lower losses)
3. Eval loss still improving at epoch 10 but with diminishing returns
4. Variance of ~0.29 in final 50 steps suggests optimization instability

Diagnosis:
- NOT a capacity bottleneck (losses already low, model can fit the task)
- Problem is optimization instability and convergence precision
- AIMC's mixed content (detailed responses + deflections) creates inconsistent training signals
- LR of 3e-4 may be too high, causing model to "bounce around" optimal point

Strategy:
1. Lower learning rate (2e-4, 1.5e-4) for stable, precise convergence
2. Moderate capacity increases (r=12, r=16) - not aggressive like Huatuo
3. Bidirectional regularization exploration (dropout 0.05 and 0.15) to empirically determine optimal
4. Target: train_loss < 1.9, reduced variance (< 0.20 range), stable convergence
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
    # AIMC-specific grid search parameters
    # Focus on precision and stability, not aggressive capacity increases

    param_grid = {
        'learning_rate': [
            2e-4,   # Lower for precision
            1.5e-4, # Very stable convergence
            3e-4,   # Baseline - too oscillatory
        ],
        'lora_r': [
            8,      # Baseline
            12,     # Moderate increase - controlled improvement
            16,     # Higher capacity - but not aggressive like r=24
        ],
        'lora_alpha': [
            32,     # Baseline (r=8)
            48,     # For r=12
            64,     # For r=16
        ],
        'lora_dropout': [
            0.05,   # Less regularization - test if helps stability
            0.1,    # Baseline
            0.15,   # More regularization - test if reduces variance
        ],
        'batch_size': [
            8,      # Baseline
        ],
        'warmup_ratio': [
            0.1,    # Baseline
        ],
        'weight_decay': [
            0.01,   # Baseline
        ],
    }

    # Top 4 configurations for precision/stability optimization
    # Strategy: Lower LR for precision + Moderate capacity + Explore regularization bidirectionally
    priority_configs = [
        # 1. BEST BET: Higher capacity + moderate LR for precision
        {'learning_rate': 2e-4, 'lora_r': 16, 'lora_alpha': 64, 'lora_dropout': 0.1, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.01},

        # 2. Maximum precision: moderate capacity + lowest LR
        {'learning_rate': 1.5e-4, 'lora_r': 12, 'lora_alpha': 48, 'lora_dropout': 0.1, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.01},

        # 3. Test less regularization (might reduce variance by better fitting heterogeneous data)
        {'learning_rate': 2e-4, 'lora_r': 12, 'lora_alpha': 48, 'lora_dropout': 0.05, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.01},

        # 4. Test more regularization (might reduce variance by constraining overfitting to noise)
        {'learning_rate': 2e-4, 'lora_r': 12, 'lora_alpha': 48, 'lora_dropout': 0.15, 'batch_size': 8, 'warmup_ratio': 0.1, 'weight_decay': 0.01},
    ]

    train_jsonl_path = 'data/aimc_formatted/aimc_train.jsonl'
    base_output_dir = 'model/checkpoints/aimc_lora_gs'
    os.makedirs(base_output_dir, exist_ok=True)

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Starting AIMC LoRA Grid Search with {len(priority_configs)} priority configurations")
    print(f"Baseline: train_loss=2.016, eval_loss=1.756, high variance (1.91-2.20 range)")
    print(f"Strategy: Precision tuning with lower LR, moderate capacity, bidirectional regularization\n")

    for idx, config in enumerate(priority_configs, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{len(priority_configs)}")
        print(f"{'='*80}")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Create run-specific output directory
        run_name = f"aimc_gs_{idx}_lr{config['learning_rate']:.1e}_r{config['lora_r']}_alpha{config['lora_alpha']}_do{config['lora_dropout']}"
        output_dir = os.path.join(base_output_dir, run_name)

        try:
            # Train with 6 epochs (baseline showed main convergence by epoch 3-4)
            train_lora(
                train_jsonl_path=train_jsonl_path,
                output_dir=output_dir,
                batch_size=config['batch_size'],
                epochs=6,
                learning_rate=config['learning_rate'],
                lora_r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                warmup_ratio=config['warmup_ratio'],
                weight_decay=config['weight_decay'],
                use_wandb=True,
                wandb_project='aimc-lora-gs',
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
    print(f"\nNext steps:")
    print(f"  1. Check WandB dashboard for training curves and variance")
    print(f"  2. Compare final train_loss (target: < 1.9 with < 0.20 variance)")
    print(f"  3. Look for smooth, stable convergence without oscillations")
    print(f"  4. Select best config for potential extended training")

    return results


if __name__ == "__main__":
    results = run_grid_search()