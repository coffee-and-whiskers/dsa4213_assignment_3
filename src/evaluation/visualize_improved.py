#!/usr/bin/env python
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
