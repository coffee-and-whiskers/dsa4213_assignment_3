#!/usr/bin/env python
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
        f.write(f"Best model for generation: {best_model_name}\n")
        f.write(f"Test PPL: {results[best_model_name]['test_ppl']:.3f}\n")
    
    print("Generation info saved to results_improved/generation_info.txt")

if __name__ == "__main__":
    main()
