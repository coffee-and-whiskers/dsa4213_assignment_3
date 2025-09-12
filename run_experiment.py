#!/usr/bin/env python
"""
Run the complete language modeling experiment
"""

import subprocess
import sys
import time


def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Failed to run: {e}")
        return False


def main():
    print("LANGUAGE MODELING EXPERIMENT")
    print("="*60)
    print("This will:")
    print("1. Preprocess the data")
    print("2. Train LSTM, RNN, and Transformer models")
    print("3. Run ablation studies")
    print("4. Generate text samples")
    print("5. Create visualizations")
    print("="*60)
    
    steps = [
        ("python src/utils/preprocess.py", "Step 1: Preprocessing data"),
        ("python src/training/train.py", "Step 2: Training models (this will take time)"),
        ("python src/training/ablation.py", "Step 3: Running ablation studies"),
        ("python src/evaluation/generate.py", "Step 4: Generating text samples"),
        ("python src/evaluation/visualize.py", "Step 5: Creating visualizations")
    ]
    
    start_time = time.time()
    
    for cmd, description in steps:
        if not run_command(cmd, description):
            print(f"\nFailed at: {description}")
            print("Please check the error and run the command manually.")
            sys.exit(1)
        time.sleep(1)  # Small pause between steps
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print("\nResults saved:")
    print("- Model checkpoints: results/models/best_*.pt")
    print("- Training results: results/all_results.json")
    print("- Ablation results: results/ablation_results.json")
    print("- Generated samples: results/generated_samples.txt")
    print("- Visualizations: results/figures/*.png")
    print("- Summary: results/experiment_summary.txt")


if __name__ == "__main__":
    main()