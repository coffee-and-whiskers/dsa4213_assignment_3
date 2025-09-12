import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import pickle
import json
from datetime import datetime
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.preprocess import CharTokenizer, load_and_split_data, create_data_loaders
from src.models.models import LSTMModel, RNNModel, SimpleTransformerLM


def train_epoch(model, train_loader, optimizer, criterion, device, clip_value=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _ = model(x)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        y = y.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        # Track loss
        total_loss += loss.item() * y.size(0)
        total_tokens += y.size(0)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            output, _ = model(x)
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))
            y = y.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, y)
            
            # Track loss
            total_loss += loss.item() * y.size(0)
            total_tokens += y.size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def generate_text(model, tokenizer, device, seed_text="The", max_length=200, temperature=1.0):
    """Generate text using the model"""
    model.eval()
    
    # Encode seed text
    tokens = [tokenizer.char_to_idx.get('<BOS>', 2)] + tokenizer.encode(seed_text)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([tokens[-256:]], dtype=torch.long).to(device)  # Use last 256 tokens
            
            # Get predictions
            output, _ = model(x)
            logits = output[0, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if we hit EOS
            if next_token == tokenizer.char_to_idx.get('<EOS>', 3):
                break
            
            tokens.append(next_token)
    
    # Decode to text
    generated = tokenizer.decode(tokens[1:])  # Skip BOS token
    generated = generated.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '')
    
    return generated


def train_model(model_class, model_name, train_loader, val_loader, test_loader, 
                tokenizer, device, config):
    """Train a single model"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Get vocab size from config or tokenizer
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = config.get('vocab_size', 164)  # Default char vocab size
    
    # Initialize model
    model = model_class(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    # Training history
    history = {
        'train_loss': [],
        'train_ppl': [],
        'val_loss': [],
        'val_ppl': [],
        'epoch_times': []
    }
    
    best_val_ppl = float('inf')
    
    # Training loop
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, criterion, 
                                           device, config['clip_value'])
        
        # Validate
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['epoch_times'].append(epoch_time)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'config': config
            }, f'results/models/best_{model_name.lower()}_model.pt')
            print(f"  Saved best model (Val PPL: {val_ppl:.2f})")
        
        # Early stopping
        if epoch > 10 and val_ppl > min(history['val_ppl'][-5:]) * 1.1:
            print("Early stopping triggered")
            break
    
    # Test evaluation
    test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")
    
    # Generate samples
    print(f"\nGenerating samples with different temperatures:")
    for temp in [0.7, 1.0, 1.3]:
        text = generate_text(model, tokenizer, device, "The door", temperature=temp)
        print(f"\nTemperature {temp}:")
        print(text[:200])
    
    # Save results
    results = {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_ppl': test_ppl,
        'best_val_ppl': best_val_ppl,
        'total_params': total_params,
        'total_training_time': sum(history['epoch_times']),
        'history': history,
        'config': config
    }
    
    with open(f'results/{model_name.lower()}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data directly from numpy files
    print("Loading data...")
    
    # Check if numpy files exist
    if os.path.exists('data/processed/char_X_train.npy'):
        print("Loading character-level data from numpy files...")
        X_train = np.load('data/processed/char_X_train.npy')
        y_train = np.load('data/processed/char_y_train.npy')
        X_val = np.load('data/processed/char_X_val.npy')
        y_val = np.load('data/processed/char_y_val.npy')
        X_test = np.load('data/processed/char_X_test.npy')
        y_test = np.load('data/processed/char_y_test.npy')
        
        # Load vocabulary and create minimal tokenizer
        with open('data/processed/char_vocab.json', 'r') as f:
            vocab_data = json.load(f)
            vocab_size = vocab_data['vocab_size']
        
        # Create a minimal tokenizer object for compatibility
        class MinimalTokenizer:
            def __init__(self, vocab_data):
                self.vocab_size = vocab_data['vocab_size']
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
                
            def encode(self, text):
                """Convert text to indices"""
                return [self.char_to_idx.get(char, 1) for char in text]  # 1 is <UNK>
                
            def decode(self, indices):
                chars = [self.idx_to_char.get(idx, '?') for idx in indices]
                return ''.join(chars)
        
        tokenizer = MinimalTokenizer(vocab_data)
        
        # Create datasets
        train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
        
        # Base configuration
        base_config = {
            'vocab_size': vocab_size,
            'embed_size': 128,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'num_epochs': 30,
            'batch_size': 32,
            'seq_length': 256,
            'clip_value': 1.0
        }
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=base_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=base_config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=base_config['batch_size'], shuffle=False)
        
    else:
        # Fallback to old method
        train_texts, val_texts, test_texts = load_and_split_data('data/raw/nosleep_1939posts.json')
        
        # Load or create tokenizer
        tokenizer = CharTokenizer()
        if os.path.exists('data/processed/tokenizer.pkl'):
            tokenizer.load('data/processed/tokenizer.pkl')
        else:
            tokenizer.fit(train_texts + val_texts + test_texts)
            tokenizer.save('data/processed/tokenizer.pkl')
        
        vocab_size = tokenizer.vocab_size
        
        # Base configuration
        base_config = {
            'vocab_size': vocab_size,
            'embed_size': 128,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'num_epochs': 30,
            'batch_size': 32,
            'seq_length': 256,
            'clip_value': 1.0
        }
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_texts, val_texts, test_texts, tokenizer,
            seq_length=base_config['seq_length'],
            batch_size=base_config['batch_size']
        )
    
    # Train models
    models_to_train = [
        (LSTMModel, "LSTM"),
        (RNNModel, "RNN"),
        (SimpleTransformerLM, "Transformer")
    ]
    
    all_results = {}
    
    for model_class, model_name in models_to_train:
        # Adjust config for transformer
        config = base_config.copy()
        if model_name == "Transformer":
            config['num_heads'] = 4  # For transformer
            
        model, results = train_model(
            model_class, model_name,
            train_loader, val_loader, test_loader,
            tokenizer, device, config
        )
        
        all_results[model_name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Test Perplexity: {results['test_ppl']:.2f}")
        print(f"  Best Val Perplexity: {results['best_val_ppl']:.2f}")
        print(f"  Training Time: {results['total_training_time']:.2f}s")
        print(f"  Parameters: {results['total_params']:,}")
    
    # Save combined results
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nAll results saved to all_results.json")


if __name__ == "__main__":
    main()