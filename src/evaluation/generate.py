import torch
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.models import LSTMModel, RNNModel, SimpleTransformerLM
from src.utils.preprocess import CharTokenizer


def generate_text(model, tokenizer, device, seed_text="The", max_length=200, temperature=1.0):
    """Generate text using the model"""
    model.eval()
    
    # Encode seed text
    tokens = [tokenizer.char_to_idx.get('<BOS>', 2)] + tokenizer.encode(seed_text)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([tokens[-256:]], dtype=torch.long).to(device)
            
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer from vocab
    import json
    with open('char_vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
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
    
    # Seed texts for horror generation
    seed_texts = [
        "The door",
        "I woke up",
        "She smiled",
        "The mirror",
        "In the darkness"
    ]
    
    temperatures = [0.7, 1.0, 1.3]
    
    # Models to generate from
    models_info = [
        ('LSTM', LSTMModel, 'best_lstm_model.pt'),
        ('RNN', RNNModel, 'best_rnn_model.pt'),
        ('Transformer', SimpleTransformerLM, 'best_transformer_model.pt')
    ]
    
    print("="*60)
    print("TEXT GENERATION SAMPLES")
    print("="*60)
    
    for model_name, model_class, checkpoint_path in models_info:
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Initialize model
            config = checkpoint['config']
            if model_name == 'Transformer':
                model = model_class(
                    vocab_size=tokenizer.vocab_size,
                    embed_size=config['embed_size'],
                    num_heads=4,
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                ).to(device)
            else:
                model = model_class(
                    vocab_size=tokenizer.vocab_size,
                    embed_size=config['embed_size'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"\n{model_name} Model Generation")
            print("-"*40)
            
            for temp in temperatures:
                print(f"\nTemperature = {temp}:")
                for seed in seed_texts[:2]:  # Use first 2 seeds for brevity
                    text = generate_text(model, tokenizer, device, seed, max_length=150, temperature=temp)
                    print(f"\nSeed: '{seed}'")
                    print(f"Generated: {text[:200]}...")
                    
        except FileNotFoundError:
            print(f"\n{model_name} model checkpoint not found. Train the model first.")
            continue
    
    # Save some samples to file
    print("\n\nGenerating samples for documentation...")
    
    samples = {}
    for model_name, model_class, checkpoint_path in models_info:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint['config']
            
            if model_name == 'Transformer':
                model = model_class(
                    vocab_size=tokenizer.vocab_size,
                    embed_size=config['embed_size'],
                    num_heads=4,
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                ).to(device)
            else:
                model = model_class(
                    vocab_size=tokenizer.vocab_size,
                    embed_size=config['embed_size'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            samples[model_name] = {}
            for temp in temperatures:
                samples[model_name][f'temp_{temp}'] = []
                for seed in seed_texts:
                    text = generate_text(model, tokenizer, device, seed, max_length=200, temperature=temp)
                    samples[model_name][f'temp_{temp}'].append({
                        'seed': seed,
                        'generated': text[:300]
                    })
                    
        except FileNotFoundError:
            continue
    
    # Save samples
    with open('generated_samples.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("GENERATED TEXT SAMPLES\n")
        f.write("="*60 + "\n\n")
        
        for model_name in samples:
            f.write(f"\n{model_name} MODEL\n")
            f.write("-"*40 + "\n")
            
            for temp_key in samples[model_name]:
                temp = temp_key.replace('temp_', '')
                f.write(f"\nTemperature = {temp}:\n")
                
                for sample in samples[model_name][temp_key][:3]:  # First 3 samples
                    f.write(f"\nSeed: '{sample['seed']}'\n")
                    f.write(f"{sample['generated']}\n")
    
    print("Samples saved to generated_samples.txt")


if __name__ == "__main__":
    main()