import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import json
import time
import pickle
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.preprocess import CharTokenizer, load_and_split_data, create_data_loaders
from src.models.models import LSTMModel
from src.training.train import train_epoch, evaluate


def run_ablation_study(train_loader, val_loader, test_loader, tokenizer, device):
    """Run ablation studies on LSTM model"""
    
    print("="*60)
    print("ABLATION STUDIES")
    print("="*60)
    
    ablation_results = {}
    
    # Study 1: Dropout comparison (0.0 vs 0.2)
    print("\n1. Dropout Ablation Study")
    print("-"*40)
    
    for dropout in [0.0, 0.2]:
        print(f"\nTraining with dropout={dropout}")
        
        model = LSTMModel(
            vocab_size=tokenizer.vocab_size,
            embed_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=dropout
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        history = {'train_ppl': [], 'val_ppl': []}
        
        for epoch in range(15):  # Fewer epochs for ablation
            start_time = time.time()
            
            # Train
            train_loss, train_ppl = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
            
            history['train_ppl'].append(train_ppl)
            history['val_ppl'].append(val_ppl)
            
            print(f"  Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}")
        
        # Test
        test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
        
        ablation_results[f'dropout_{dropout}'] = {
            'test_ppl': test_ppl,
            'final_val_ppl': history['val_ppl'][-1],
            'history': history,
            'overfitting_gap': history['val_ppl'][-1] - history['train_ppl'][-1]
        }
        
        print(f"  Final Test PPL: {test_ppl:.2f}")
        print(f"  Overfitting gap: {ablation_results[f'dropout_{dropout}']['overfitting_gap']:.2f}")
    
    # Study 2: Context length comparison (128 vs 256)
    print("\n2. Context Length Ablation Study")
    print("-"*40)
    
    for seq_length in [128, 256]:
        print(f"\nTraining with sequence_length={seq_length}")
        
        # For sequence length study, we need to recreate data loaders
        # Skip seq_length=256 since we already have those results
        if seq_length == 256:
            train_loader_seq = train_loader
            val_loader_seq = val_loader
            test_loader_seq = test_loader
        else:
            # For seq_length=128, we need to recreate with shorter sequences
            # Load the numpy data and create new loaders
            X_train = np.load('data/processed/char_X_train.npy')
            y_train = np.load('data/processed/char_y_train.npy')
            X_val = np.load('data/processed/char_X_val.npy')
            y_val = np.load('data/processed/char_y_val.npy')
            X_test = np.load('data/processed/char_X_test.npy')
            y_test = np.load('data/processed/char_y_test.npy')
            
            # Truncate to seq_length
            X_train = X_train[:, :seq_length]
            y_train = y_train[:, :seq_length]
            X_val = X_val[:, :seq_length]
            y_val = y_val[:, :seq_length]
            X_test = X_test[:, :seq_length]
            y_test = y_test[:, :seq_length]
            
            from torch.utils.data import TensorDataset, DataLoader
            train_dataset_seq = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
            val_dataset_seq = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
            test_dataset_seq = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
            
            train_loader_seq = DataLoader(train_dataset_seq, batch_size=32, shuffle=True)
            val_loader_seq = DataLoader(val_dataset_seq, batch_size=32, shuffle=False)
            test_loader_seq = DataLoader(test_dataset_seq, batch_size=32, shuffle=False)
        
        model = LSTMModel(
            vocab_size=tokenizer.vocab_size,
            embed_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.2
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        history = {'train_ppl': [], 'val_ppl': []}
        
        for epoch in range(15):
            # Train
            train_loss, train_ppl = train_epoch(model, train_loader_seq, optimizer, criterion, device)
            
            # Validate
            val_loss, val_ppl = evaluate(model, val_loader_seq, criterion, device)
            
            history['train_ppl'].append(train_ppl)
            history['val_ppl'].append(val_ppl)
            
            print(f"  Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}")
        
        # Test
        test_loss, test_ppl = evaluate(model, test_loader_seq, criterion, device)
        
        ablation_results[f'seq_length_{seq_length}'] = {
            'test_ppl': test_ppl,
            'final_val_ppl': history['val_ppl'][-1],
            'history': history,
            'convergence_speed': history['val_ppl'][4]  # PPL at epoch 5
        }
        
        print(f"  Final Test PPL: {test_ppl:.2f}")
        print(f"  PPL at epoch 5: {history['val_ppl'][4]:.2f}")
    
    # Save ablation results
    with open('results/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    print("\nDropout Study:")
    print(f"  No dropout (0.0): Test PPL = {ablation_results['dropout_0.0']['test_ppl']:.2f}")
    print(f"  With dropout (0.2): Test PPL = {ablation_results['dropout_0.2']['test_ppl']:.2f}")
    print(f"  Overfitting reduction: {ablation_results['dropout_0.0']['overfitting_gap'] - ablation_results['dropout_0.2']['overfitting_gap']:.2f}")
    
    print("\nContext Length Study:")
    print(f"  Seq length 128: Test PPL = {ablation_results['seq_length_128']['test_ppl']:.2f}")
    print(f"  Seq length 256: Test PPL = {ablation_results['seq_length_256']['test_ppl']:.2f}")
    print(f"  Improvement with longer context: {ablation_results['seq_length_128']['test_ppl'] - ablation_results['seq_length_256']['test_ppl']:.2f}")
    
    return ablation_results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    
    # Load numpy data directly
    if os.path.exists('data/processed/char_X_train.npy'):
        print("Using pre-prepared numpy data...")
        import numpy as np
        X_train = np.load('data/processed/char_X_train.npy')
        y_train = np.load('data/processed/char_y_train.npy')
        X_val = np.load('data/processed/char_X_val.npy')
        y_val = np.load('data/processed/char_y_val.npy')
        X_test = np.load('data/processed/char_X_test.npy')
        y_test = np.load('data/processed/char_y_test.npy')
        
        with open('data/processed/char_vocab.json', 'r') as f:
            vocab_data = json.load(f)
        
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create minimal tokenizer
        class MinimalTokenizer:
            def __init__(self, vocab_data):
                self.vocab_size = vocab_data['vocab_size']
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
            def encode(self, text):
                return [self.char_to_idx.get(char, 1) for char in text]
            def decode(self, indices):
                return ''.join([self.idx_to_char.get(idx, '?') for idx in indices])
        
        tokenizer = MinimalTokenizer(vocab_data)
    else:
        train_texts, val_texts, test_texts = load_and_split_data('data/raw/nosleep_1939posts.json')
        
        # Load tokenizer
        tokenizer = CharTokenizer()
        tokenizer.load('data/processed/tokenizer.pkl')
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_texts, val_texts, test_texts, tokenizer,
            seq_length=256, batch_size=32
        )
    
    # Run ablation studies
    ablation_results = run_ablation_study(train_loader, val_loader, test_loader, tokenizer, device)
    
    print("\nAblation results saved to results/ablation_results.json")


if __name__ == "__main__":
    main()