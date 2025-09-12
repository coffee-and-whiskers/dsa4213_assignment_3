#!/usr/bin/env python
"""
Training with improved hyperparameters based on our analysis
Improvements:
1. Larger hidden size (256 -> 384) for more capacity
2. Lower learning rate (0.001 -> 0.0005) for more stable training
3. Increased embedding size (128 -> 192) for richer representations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.models import LSTMModel, RNNModel, SimpleTransformerLM
from src.training.train import train_epoch, evaluate

def train_improved_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open('data/processed/char_vocab.json', 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    from torch.utils.data import DataLoader, TensorDataset
    
    X_train = np.load('data/processed/char_X_train.npy')
    y_train = np.load('data/processed/char_y_train.npy')
    X_val = np.load('data/processed/char_X_val.npy')
    y_val = np.load('data/processed/char_y_val.npy')
    X_test = np.load('data/processed/char_X_test.npy')
    y_test = np.load('data/processed/char_y_test.npy')
    
    print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create output directory for improved models
    os.makedirs('results_improved', exist_ok=True)
    os.makedirs('results_improved/models', exist_ok=True)
    
    print("\n" + "="*60)
    print("IMPROVED HYPERPARAMETER CONFIGURATIONS")
    print("="*60)
    
    # Define improved configurations based on our analysis
    configs = [
        {
            "model_name": "LSTM_Large",
            "model_class": LSTMModel,
            "embed_size": 192,    # Increased from 128
            "hidden_size": 384,   # Increased from 256
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.0005,  # Decreased from 0.001
            "weight_decay": 0.0001,   # Added L2 regularization
            "epochs": 25
        },
        {
            "model_name": "LSTM_Deep",
            "model_class": LSTMModel,
            "embed_size": 128,
            "hidden_size": 256,
            "num_layers": 3,      # Increased from 2
            "dropout": 0.25,      # Slightly increased
            "learning_rate": 0.0007,
            "weight_decay": 0.0001,
            "epochs": 25
        },
        {
            "model_name": "LSTM_Wide",
            "model_class": LSTMModel,
            "embed_size": 256,    # Much larger embeddings
            "hidden_size": 320,   # Moderately larger hidden
            "num_layers": 2,
            "dropout": 0.15,      # Slightly less dropout
            "learning_rate": 0.0003,  # Lower LR for stability
            "weight_decay": 0.00005,
            "epochs": 25
        }
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n{'='*40}")
        print(f"Training {config['model_name']}")
        print(f"{'='*40}")
        print("Configuration:")
        for key, val in config.items():
            if key not in ["model_class", "epochs"]:
                print(f"  {key}: {val}")
        
        # Create model
        model = config['model_class'](
            vocab_size=vocab_size,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_ppl': [],
            'val_loss': [], 'val_ppl': []
        }
        
        best_val_ppl = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stop_patience = 5
        
        print(f"\nStarting training for {config['epochs']} epochs...")
        start_time = time.time()
        
        for epoch in range(config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_ppl = train_epoch(
                model, train_loader, optimizer, criterion, device, clip_value=1.0
            )
            
            # Validate
            val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(float(train_loss))
            history['train_ppl'].append(float(train_ppl))
            history['val_loss'].append(float(val_loss))
            history['val_ppl'].append(float(val_ppl))
            
            # Save best model
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ppl': float(val_ppl),
                    'config': config
                }, f'results_improved/models/{config["model_name"].lower()}_best.pt')
            else:
                patience_counter += 1
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{config['epochs']}: "
                  f"Train Loss={train_loss:.4f}, Train PPL={train_ppl:.2f}, "
                  f"Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}, "
                  f"Time={time.time()-epoch_start:.1f}s")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model for testing
        checkpoint = torch.load(
            f'results_improved/models/{config["model_name"].lower()}_best.pt',
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
        
        total_time = time.time() - start_time
        
        # Store results
        all_results[config['model_name']] = {
            'model_name': config['model_name'],
            'config': {k: v for k, v in config.items() if k not in ['model_class', 'epochs']},
            'test_loss': float(test_loss),
            'test_ppl': float(test_ppl),
            'best_val_ppl': float(best_val_ppl),
            'best_epoch': best_epoch + 1,
            'total_params': total_params,
            'total_training_time': float(total_time),
            'history': history
        }
        
        print(f"\nResults for {config['model_name']}:")
        print(f"  Test PPL: {test_ppl:.3f}")
        print(f"  Best Val PPL: {best_val_ppl:.3f} (epoch {best_epoch+1})")
        print(f"  Training time: {total_time/60:.1f} minutes")
    
    # Save all results
    with open('results_improved/improved_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compare with original best
    print("\n" + "="*60)
    print("COMPARISON WITH ORIGINAL BEST MODEL")
    print("="*60)
    print("\nOriginal LSTM (from previous run):")
    print("  Test PPL: 3.21")
    print("  Parameters: 984,740")
    print("  Config: embed=128, hidden=256, layers=2, dropout=0.2, lr=0.001")
    
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Test PPL: {result['test_ppl']:.3f}")
        improvement = ((3.21 - result['test_ppl']) / 3.21) * 100
        if improvement > 0:
            print(f"  Improvement: {improvement:.1f}% better than original")
        else:
            print(f"  Performance: {-improvement:.1f}% worse than original")
        print(f"  Parameters: {result['total_params']:,}")
    
    # Find best model
    best_model = min(all_results.items(), key=lambda x: x[1]['test_ppl'])
    
    # Create summary
    with open('results_improved/training_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("IMPROVED HYPERPARAMETER EXPERIMENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Model: {best_model[0]}\n")
        f.write(f"Test Perplexity: {best_model[1]['test_ppl']:.3f}\n")
        f.write(f"Parameters: {best_model[1]['total_params']:,}\n\n")
        f.write("Configuration:\n")
        for key, val in best_model[1]['config'].items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        if best_model[1]['test_ppl'] < 3.21:
            f.write(f"SUCCESS: Improved over original by {((3.21 - best_model[1]['test_ppl'])/3.21)*100:.1f}%!\n")
        else:
            f.write("Original configuration still performs best.\n")
        f.write("\nAll Models Tested:\n")
        for name, result in sorted(all_results.items(), key=lambda x: x[1]['test_ppl']):
            f.write(f"  {name}: PPL={result['test_ppl']:.3f}, Params={result['total_params']:,}\n")
    
    print(f"\nBest model: {best_model[0]} with test PPL of {best_model[1]['test_ppl']:.3f}")
    print("Results saved to results_improved/")
    
    return all_results

if __name__ == "__main__":
    train_improved_models()