#!/usr/bin/env python
"""
Run improved hyperparameter experiments
Saves to results_improved/ directory to preserve original results
"""

import subprocess
import sys
import time
import os

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
    print("IMPROVED HYPERPARAMETER EXPERIMENT")
    print("="*60)
    print("This will:")
    print("1. Train models with improved hyperparameters")
    print("2. Compare with original best model")
    print("3. Generate extended samples with best model")
    print("4. Create visualizations")
    print("\nResults will be saved to results_improved/ directory")
    print("Original results in results/ will be preserved")
    print("="*60)
    
    # Create results_improved directory structure
    os.makedirs('results_improved', exist_ok=True)
    os.makedirs('results_improved/models', exist_ok=True)
    os.makedirs('results_improved/figures', exist_ok=True)
    
    steps = [
        ("python src/training/train_improved.py", "Step 1: Training improved models"),
        ("python src/evaluation/generate_improved.py", "Step 2: Generating samples with improved models"),
        ("python src/evaluation/visualize_improved.py", "Step 3: Creating comparison visualizations")
    ]
    
    start_time = time.time()
    
    # Run main training
    print("\nStarting improved hyperparameter training...")
    print("This will train 3 model configurations:")
    print("  1. LSTM_Large: Increased embedding and hidden size")
    print("  2. LSTM_Deep: Additional layer with more dropout")
    print("  3. LSTM_Wide: Much larger embeddings")
    
    if not run_command(steps[0][0], steps[0][1]):
        print(f"\nFailed at: {steps[0][1]}")
        print("Continuing with other steps if models were saved...")
    
    # Check if we need to create the other scripts
    if not os.path.exists('src/evaluation/generate_improved.py'):
        print("\nCreating generation script for improved models...")
        create_generation_script()
    
    if not os.path.exists('src/evaluation/visualize_improved.py'):
        print("\nCreating visualization script for improved models...")
        create_visualization_script()
    
    # Run remaining steps
    for cmd, description in steps[1:]:
        if os.path.exists(cmd.split()[1]):  # Check if script exists
            if not run_command(cmd, description):
                print(f"\nWarning: {description} failed, continuing...")
        else:
            print(f"\nSkipping: {description} (script not found)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("IMPROVED EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print("\nResults saved in results_improved/:")
    print("- Model checkpoints: results_improved/models/*.pt")
    print("- Training results: results_improved/improved_results.json")
    print("- Summary: results_improved/training_summary.txt")
    
    # Display summary if it exists
    if os.path.exists('results_improved/training_summary.txt'):
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        with open('results_improved/training_summary.txt', 'r') as f:
            print(f.read())

def create_generation_script():
    """Create a simple generation script for improved models"""
    script = '''#!/usr/bin/env python
"""Generate samples with improved models"""
import torch
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.models import LSTMModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load results to find best model
    with open('results_improved/improved_results.json', 'r') as f:
        results = json.load(f)
    
    best_model_name = min(results.items(), key=lambda x: x[1]['test_ppl'])[0]
    print(f"Generating with best model: {best_model_name}")
    
    # Save simple generation info
    with open('results_improved/generation_info.txt', 'w') as f:
        f.write(f"Best model for generation: {best_model_name}\\n")
        f.write(f"Test PPL: {results[best_model_name]['test_ppl']:.3f}\\n")
    
    print("Generation info saved to results_improved/generation_info.txt")

if __name__ == "__main__":
    main()
'''
    
    os.makedirs('src/evaluation', exist_ok=True)
    with open('src/evaluation/generate_improved.py', 'w') as f:
        f.write(script)
    os.chmod('src/evaluation/generate_improved.py', 0o755)

def create_visualization_script():
    """Create a simple visualization script for improved models"""
    script = '''#!/usr/bin/env python
"""Visualize improved model results"""
import json
import matplotlib.pyplot as plt
import os

def main():
    # Load both original and improved results
    with open('results/all_results.json', 'r') as f:
        original = json.load(f)
    
    if os.path.exists('results_improved/improved_results.json'):
        with open('results_improved/improved_results.json', 'r') as f:
            improved = json.load(f)
        
        # Simple comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Original models
        orig_names = ['LSTM', 'RNN', 'Transformer']
        orig_ppls = [original[n]['test_ppl'] for n in orig_names]
        
        # Improved models
        imp_names = list(improved.keys())
        imp_ppls = [improved[n]['test_ppl'] for n in imp_names]
        
        x = range(len(orig_names) + len(imp_names))
        names = orig_names + imp_names
        ppls = orig_ppls + imp_ppls
        colors = ['blue']*3 + ['green']*len(imp_names)
        
        bars = ax.bar(names, ppls, color=colors, alpha=0.7)
        ax.axhline(y=3.21, color='red', linestyle='--', label='Original Best (3.21)')
        ax.set_ylabel('Test Perplexity')
        ax.set_title('Model Performance Comparison')
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results_improved/figures/comparison.png')
        print("Comparison plot saved to results_improved/figures/comparison.png")
    else:
        print("No improved results found yet")

if __name__ == "__main__":
    main()
'''
    
    os.makedirs('src/evaluation', exist_ok=True)
    with open('src/evaluation/visualize_improved.py', 'w') as f:
        f.write(script)
    os.chmod('src/evaluation/visualize_improved.py', 0o755)

if __name__ == "__main__":
    main()