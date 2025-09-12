#!/usr/bin/env python
"""
Generate extended text samples using the best model with optimal hyperparameters
Based on our results:
- Best model: LSTM (PPL 3.21)
- Best dropout: 0.2
- Best sequence length: 256
- Best temperature for balance: 1.0
"""

import torch
import torch.nn as nn
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.models import LSTMModel

def generate_extended_text(model, tokenizer, device, seed_text="", max_length=500, temperature=1.0):
    """Generate extended text using the model"""
    model.eval()
    
    # Encode seed text
    if seed_text:
        tokens = [tokenizer.char_to_idx.get('<BOS>', 2)] + tokenizer.encode(seed_text)
    else:
        tokens = [tokenizer.char_to_idx.get('<BOS>', 2)]
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (use last 256 tokens for context)
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
            generated_tokens.append(next_token)
    
    # Decode to text
    generated = tokenizer.decode(generated_tokens)
    generated = generated.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '')
    
    return generated


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    with open('data/processed/char_vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    class CharTokenizer:
        def __init__(self, vocab_data):
            self.vocab_size = vocab_data['vocab_size']
            self.char_to_idx = vocab_data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        def encode(self, text):
            return [self.char_to_idx.get(char, 1) for char in text]
        
        def decode(self, indices):
            return ''.join([self.idx_to_char.get(idx, '?') for idx in indices])
    
    tokenizer = CharTokenizer(vocab_data)
    
    # Load best LSTM model (based on our results)
    print("\nLoading best LSTM model...")
    model = LSTMModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2  # Best dropout from ablation
    ).to(device)
    
    checkpoint = torch.load('results/models/best_lstm_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extended generation prompts for horror stories
    prompts = [
        ("The basement door had been locked for twenty years", "basement_horror"),
        ("I found a diary in the walls of my new house", "diary_discovery"),
        ("The children in the neighborhood won't stop staring", "creepy_children"),
        ("My reflection started moving on its own", "mirror_horror"),
        ("The GPS kept taking me to the same abandoned house", "gps_mystery"),
        ("Every night at 3:33 AM, I hear breathing", "night_terror"),
        ("The old woman at the bus stop wasn't there yesterday", "phantom_woman"),
        ("My dog refuses to enter the guest bedroom", "animal_instinct")
    ]
    
    # Different generation settings based on our findings
    generation_configs = [
        {"temp": 0.8, "length": 500, "desc": "Conservative Extended"},
        {"temp": 1.0, "length": 750, "desc": "Balanced Long-form"},
        {"temp": 1.1, "length": 1000, "desc": "Creative Narrative"}
    ]
    
    results = []
    
    print("\nGenerating extended samples...")
    print("="*60)
    
    for config in generation_configs:
        print(f"\n{config['desc']} (Temperature={config['temp']}, Length={config['length']})")
        print("-"*40)
        
        for prompt, prompt_id in prompts[:4]:  # Use first 4 prompts for each config
            print(f"  Generating from: '{prompt[:30]}...'")
            
            generated = generate_extended_text(
                model, tokenizer, device,
                seed_text=prompt,
                max_length=config['length'],
                temperature=config['temp']
            )
            
            results.append({
                "config": config['desc'],
                "temperature": config['temp'],
                "max_length": config['length'],
                "prompt": prompt,
                "prompt_id": prompt_id,
                "generated": generated,
                "actual_length": len(generated)
            })
    
    # Save extended generations
    with open('results/extended_generations.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXTENDED TEXT GENERATION SAMPLES\n")
        f.write("Using Best Model: LSTM (Test PPL: 3.21)\n")
        f.write("Optimal Hyperparameters: Dropout=0.2, Seq_Length=256, Hidden=256\n")
        f.write("="*80 + "\n\n")
        
        for config_name in ["Conservative Extended", "Balanced Long-form", "Creative Narrative"]:
            config_results = [r for r in results if r['config'] == config_name]
            
            f.write(f"\n{config_name.upper()}\n")
            f.write("="*60 + "\n")
            f.write(f"Temperature: {config_results[0]['temperature']}\n")
            f.write(f"Target Length: {config_results[0]['max_length']} characters\n")
            f.write("-"*60 + "\n\n")
            
            for result in config_results:
                f.write(f"Prompt: \"{result['prompt']}\"\n")
                f.write(f"Length: {result['actual_length']} characters\n")
                f.write("-"*40 + "\n")
                f.write(result['generated'] + "\n")
                f.write("\n" + "="*40 + "\n\n")
    
    # Also save as JSON for analysis
    with open('results/extended_generations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} extended samples")
    print("Results saved to:")
    print("  - results/extended_generations.txt")
    print("  - results/extended_generations.json")
    
    # Generate statistics
    print("\nGeneration Statistics:")
    print("-"*40)
    for config_name in ["Conservative Extended", "Balanced Long-form", "Creative Narrative"]:
        config_results = [r for r in results if r['config'] == config_name]
        avg_length = sum(r['actual_length'] for r in config_results) / len(config_results)
        print(f"{config_name}:")
        print(f"  Average length: {avg_length:.0f} characters")
        print(f"  Samples: {len(config_results)}")

if __name__ == "__main__":
    main()