import matplotlib.pyplot as plt
import json
import numpy as np


def plot_training_curves():
    """Plot training and validation curves for all models"""
    
    # Load results
    with open('all_results.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Model Training Curves', fontsize=16)
    
    models = ['LSTM', 'RNN', 'Transformer']
    colors = ['blue', 'green', 'red']
    
    for idx, (model_name, color) in enumerate(zip(models, colors)):
        if model_name in results:
            history = results[model_name]['history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Loss curves
            ax = axes[0, idx]
            ax.plot(epochs, history['train_loss'], label='Train', color=color, linestyle='-')
            ax.plot(epochs, history['val_loss'], label='Val', color=color, linestyle='--')
            ax.set_title(f'{model_name} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cross-Entropy Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Perplexity curves
            ax = axes[1, idx]
            ax.plot(epochs, history['train_ppl'], label='Train', color=color, linestyle='-')
            ax.plot(epochs, history['val_ppl'], label='Val', color=color, linestyle='--')
            ax.set_title(f'{model_name} Perplexity')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Perplexity')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
    print("Saved training_curves.png")
    
    # Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Model Comparison', fontsize=16)
    
    model_names = list(results.keys())
    test_ppls = [results[m]['test_ppl'] for m in model_names]
    train_times = [results[m]['total_training_time'] for m in model_names]
    params = [results[m]['total_params'] for m in model_names]
    
    # Test perplexity comparison
    ax = axes[0]
    bars = ax.bar(model_names, test_ppls, color=colors)
    ax.set_ylabel('Test Perplexity')
    ax.set_title('Test Perplexity (Lower is Better)')
    for bar, val in zip(bars, test_ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom')
    
    # Training time comparison
    ax = axes[1]
    bars = ax.bar(model_names, train_times, color=colors)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Total Training Time')
    for bar, val in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}s', ha='center', va='bottom')
    
    # Parameters comparison
    ax = axes[2]
    bars = ax.bar(model_names, params, color=colors)
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Size')
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{val/1000:.0f}K', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    print("Saved model_comparison.png")


def plot_ablation_results():
    """Plot ablation study results"""
    
    try:
        with open('ablation_results.json', 'r') as f:
            ablation = json.load(f)
    except FileNotFoundError:
        print("Ablation results not found. Run ablation.py first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ablation Study Results', fontsize=16)
    
    # Dropout comparison - training curves
    ax = axes[0, 0]
    epochs = range(1, len(ablation['dropout_0.0']['history']['val_ppl']) + 1)
    ax.plot(epochs, ablation['dropout_0.0']['history']['train_ppl'], 
            label='Train (No Dropout)', color='red', linestyle='-')
    ax.plot(epochs, ablation['dropout_0.0']['history']['val_ppl'], 
            label='Val (No Dropout)', color='red', linestyle='--')
    ax.plot(epochs, ablation['dropout_0.2']['history']['train_ppl'], 
            label='Train (Dropout=0.2)', color='blue', linestyle='-')
    ax.plot(epochs, ablation['dropout_0.2']['history']['val_ppl'], 
            label='Val (Dropout=0.2)', color='blue', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Dropout Effect on Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dropout comparison - bar chart
    ax = axes[0, 1]
    dropout_vals = ['0.0', '0.2']
    test_ppls = [ablation[f'dropout_{d}']['test_ppl'] for d in dropout_vals]
    overfitting = [ablation[f'dropout_{d}']['overfitting_gap'] for d in dropout_vals]
    
    x = np.arange(len(dropout_vals))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_ppls, width, label='Test PPL', color='skyblue')
    bars2 = ax.bar(x + width/2, overfitting, width, label='Overfitting Gap', color='coral')
    
    ax.set_xlabel('Dropout Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(dropout_vals)
    ax.set_ylabel('Perplexity')
    ax.set_title('Dropout Impact')
    ax.legend()
    
    # Context length comparison - training curves
    ax = axes[1, 0]
    epochs = range(1, len(ablation['seq_length_128']['history']['val_ppl']) + 1)
    ax.plot(epochs, ablation['seq_length_128']['history']['val_ppl'], 
            label='Seq Length 128', color='green', linestyle='-')
    ax.plot(epochs, ablation['seq_length_256']['history']['val_ppl'], 
            label='Seq Length 256', color='purple', linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title('Context Length Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Context length comparison - bar chart
    ax = axes[1, 1]
    seq_lengths = ['128', '256']
    test_ppls = [ablation[f'seq_length_{s}']['test_ppl'] for s in seq_lengths]
    conv_speed = [ablation[f'seq_length_{s}']['convergence_speed'] for s in seq_lengths]
    
    x = np.arange(len(seq_lengths))
    bars1 = ax.bar(x - width/2, test_ppls, width, label='Test PPL', color='lightgreen')
    bars2 = ax.bar(x + width/2, conv_speed, width, label='PPL at Epoch 5', color='lightcoral')
    
    ax.set_xlabel('Sequence Length')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.set_ylabel('Perplexity')
    ax.set_title('Context Length Impact')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=100, bbox_inches='tight')
    print("Saved ablation_results.png")


def create_summary_table():
    """Create a summary table of all results"""
    
    with open('all_results.json', 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    
    # Model comparison table
    print("\nModel Performance Comparison:")
    print("-"*60)
    print(f"{'Model':<15} {'Test PPL':<12} {'Val PPL':<12} {'Train Time':<15} {'Parameters':<15}")
    print("-"*60)
    
    for model_name in ['LSTM', 'RNN', 'Transformer']:
        if model_name in results:
            r = results[model_name]
            print(f"{model_name:<15} {r['test_ppl']:<12.2f} {r['best_val_ppl']:<12.2f} "
                  f"{r['total_training_time']:<15.1f} {r['total_params']:<15,}")
    
    # Configuration details
    print("\n\nTraining Configuration:")
    print("-"*60)
    if results:
        config = list(results.values())[0]['config']
        for key, value in config.items():
            print(f"{key:<20}: {value}")
    
    # Best model
    best_model = min(results.items(), key=lambda x: x[1]['test_ppl'])
    print(f"\n\nBest Model: {best_model[0]} (Test PPL: {best_model[1]['test_ppl']:.2f})")
    
    # Save to text file
    with open('experiment_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("LANGUAGE MODELING EXPERIMENT RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Performance Comparison:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Model':<15} {'Test PPL':<12} {'Val PPL':<12} {'Train Time(s)':<15} {'Parameters':<15}\n")
        f.write("-"*60 + "\n")
        
        for model_name in ['LSTM', 'RNN', 'Transformer']:
            if model_name in results:
                r = results[model_name]
                f.write(f"{model_name:<15} {r['test_ppl']:<12.2f} {r['best_val_ppl']:<12.2f} "
                       f"{r['total_training_time']:<15.1f} {r['total_params']:<15,}\n")
        
        f.write(f"\n\nBest Model: {best_model[0]} (Test PPL: {best_model[1]['test_ppl']:.2f})\n")
    
    print("\nSummary saved to experiment_summary.txt")


def main():
    print("Generating visualizations...")
    plot_training_curves()
    plot_ablation_results()
    create_summary_table()
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()