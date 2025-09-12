#!/usr/bin/env python
"""
Generate text samples using the improved LSTM_Large model
"""

import torch
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.models import LSTMModel

def generate_text(model, tokenizer, device, seed_text="", max_length=500, temperature=1.0):
    """Generate text using the model"""
    model.eval()
    
    # Encode seed text
    if seed_text:
        tokens = [tokenizer['char_to_idx'].get('<BOS>', 2)] + [tokenizer['char_to_idx'].get(c, 1) for c in seed_text]
    else:
        tokens = [tokenizer['char_to_idx'].get('<BOS>', 2)]
    
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
            if next_token == tokenizer['char_to_idx'].get('<EOS>', 3):
                break
            
            tokens.append(next_token)
    
    # Decode to text
    generated = ''.join([tokenizer['idx_to_char'].get(str(idx), '?') for idx in tokens[1:]])
    generated = generated.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '')
    
    return generated

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open('data/processed/char_vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    # Check if improved model exists
    model_path = 'results_improved/models/lstm_large_best.pt'
    if not os.path.exists(model_path):
        print(f"Improved model not found at {model_path}")
        print("Please run the improved training first: python src/training/train_improved.py")
        return
    
    # Load the improved LSTM_Large model
    print("\nLoading LSTM_Large (improved) model...")
    model = LSTMModel(
        vocab_size=vocab_data['vocab_size'],
        embed_size=192,  # Improved config
        hidden_size=384,  # Improved config
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded - Test PPL: 3.186")
    
    # Horror prompts
    prompts = [
        "The basement door had been locked for twenty years",
        "My reflection started moving on its own",
        "The children in the neighborhood won't stop staring",
        "Every night at 3:33 AM, I hear breathing",
        "The old photographs showed people I don't remember"
    ]
    
    # Generate samples at different temperatures
    results = []
    print("\nGenerating improved samples...")
    print("="*60)
    
    for temp in [0.8, 1.0, 1.1]:
        print(f"\nTemperature {temp}")
        print("-"*40)
        
        for prompt in prompts[:3]:  # Use first 3 prompts
            print(f"Generating from: '{prompt[:40]}...'")
            
            generated = generate_text(
                model, vocab_data, device,
                seed_text=prompt,
                max_length=500,
                temperature=temp
            )
            
            results.append({
                'temperature': temp,
                'prompt': prompt,
                'generated': generated[len(prompt):],  # Remove prompt from output
                'full_text': prompt + generated[len(prompt):],
                'length': len(generated)
            })
    
    # Save results
    with open('results_improved/improved_generation_samples.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMPROVED MODEL GENERATION SAMPLES\n")
        f.write("Model: LSTM_Large (Test PPL: 3.186)\n")
        f.write("Config: embed=192, hidden=384, layers=2, dropout=0.2, lr=0.0005\n")
        f.write("="*80 + "\n\n")
        
        for temp in [0.8, 1.0, 1.1]:
            temp_results = [r for r in results if r['temperature'] == temp]
            
            f.write(f"\nTEMPERATURE {temp}\n")
            f.write("="*60 + "\n\n")
            
            for i, result in enumerate(temp_results, 1):
                f.write(f"Example {i}:\n")
                f.write(f"Prompt: \"{result['prompt']}\"\n")
                f.write(f"Generated ({result['length']} chars):\n")
                f.write("-"*40 + "\n")
                f.write(result['full_text'] + "\n")
                f.write("\n" + "="*40 + "\n\n")
    
    # Also save as JSON
    with open('results_improved/improved_generation_samples.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} samples")
    print("Results saved to:")
    print("  - results_improved/improved_generation_samples.txt")
    print("  - results_improved/improved_generation_samples.json")
    
    # Display a few examples
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (Temperature 1.0):")
    print("="*60)
    sample = [r for r in results if r['temperature'] == 1.0][0]
    print(f"Prompt: \"{sample['prompt']}\"")
    print(f"Generated: ...{sample['generated'][:200]}...")

if __name__ == "__main__":
    main()