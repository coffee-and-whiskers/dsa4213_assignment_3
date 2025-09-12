#!/usr/bin/env python3
"""
Simplified corpus preparation for training - processes in chunks to avoid memory issues
"""

import json
import pickle
import numpy as np
from collections import Counter
import re

print("Loading corpus...")
with open('cleaned_corpus_final.txt', 'r', encoding='utf-8') as f:
    content = f.read()

stories = [s.strip() for s in content.split('='*80) if s.strip()]
print(f"Loaded {len(stories)} stories")

# Join all stories with separator for character-level
full_text = "\n<STORY>\n".join(stories[:500])  # Use first 500 stories to avoid memory issues
print(f"Using {len(full_text):,} characters for training")

# Character-level preparation
print("\n=== Character-level preparation ===")
chars = sorted(list(set(full_text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
print(f"Vocabulary size: {len(chars)} unique characters")

# Encode text
encoded = [char_to_idx[ch] for ch in full_text]

# Create sequences
seq_length = 256  # As specified in task.md
sequences = []
targets = []

print(f"Creating sequences of length {seq_length}...")
for i in range(0, len(encoded) - seq_length, 100):  # Step by 100 to reduce data size
    sequences.append(encoded[i:i + seq_length])
    targets.append(encoded[i + 1:i + seq_length + 1])

X = np.array(sequences)
y = np.array(targets)

# 80/10/10 split
n_samples = len(sequences)
train_idx = int(n_samples * 0.8)
val_idx = int(n_samples * 0.9)

X_train = X[:train_idx]
y_train = y[:train_idx]
X_val = X[train_idx:val_idx]
y_val = y[train_idx:val_idx]
X_test = X[val_idx:]
y_test = y[val_idx:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Save character-level data
np.save('char_X_train.npy', X_train)
np.save('char_y_train.npy', y_train)
np.save('char_X_val.npy', X_val)
np.save('char_y_val.npy', y_val)
np.save('char_X_test.npy', X_test)
np.save('char_y_test.npy', y_test)

with open('char_vocab.json', 'w') as f:
    json.dump({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': len(chars),
        'seq_length': seq_length
    }, f)

print("✅ Character-level data saved")

# Word-level preparation
print("\n=== Word-level preparation ===")
all_words = []
for story in stories[:500]:  # Use first 500 stories
    words = re.findall(r'\b\w+\b|[.!?;,]', story.lower())
    all_words.extend(words)

# Build vocabulary
word_freq = Counter(all_words)
most_common = word_freq.most_common(9996)  # Leave room for special tokens

word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
for word, _ in most_common:
    if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)

idx_to_word = {idx: word for word, idx in word_to_idx.items()}
print(f"Vocabulary size: {len(word_to_idx)} words")

# Encode stories and create sequences
seq_length = 128  # As specified in task.md
sequences = []
targets = []

for story in stories[:500]:
    words = re.findall(r'\b\w+\b|[.!?;,]', story.lower())
    encoded = [word_to_idx.get(w, 1) for w in words]  # 1 is <UNK>
    
    if len(encoded) > seq_length:
        for i in range(0, len(encoded) - seq_length, 50):  # Step by 50
            sequences.append(encoded[i:i + seq_length])
            targets.append(encoded[i + 1:i + seq_length + 1])

X = np.array(sequences)
y = np.array(targets)

# 80/10/10 split
n_samples = len(sequences)
train_idx = int(n_samples * 0.8)
val_idx = int(n_samples * 0.9)

X_train = X[:train_idx]
y_train = y[:train_idx]
X_val = X[train_idx:val_idx]
y_val = y[train_idx:val_idx]
X_test = X[val_idx:]
y_test = y[val_idx:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Save word-level data
np.save('word_X_train.npy', X_train)
np.save('word_y_train.npy', y_train)
np.save('word_X_val.npy', X_val)
np.save('word_y_val.npy', y_val)
np.save('word_X_test.npy', X_test)
np.save('word_y_test.npy', y_test)

with open('word_vocab.pkl', 'wb') as f:
    pickle.dump({
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': len(word_to_idx),
        'seq_length': seq_length
    }, f)

print("✅ Word-level data saved")

# Create config files
char_config = {
    "model_type": "lstm",
    "tokenization": "character",
    "vocab_size": len(chars),
    "embedding_size": 128,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "gradient_clip": 1.0,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "batch_size": 64,
    "seq_length": 256,
    "epochs": 50
}

word_config = {
    "model_type": "lstm",
    "tokenization": "word",
    "vocab_size": len(word_to_idx),
    "embedding_size": 256,
    "hidden_size": 256,
    "num_layers": 1,
    "dropout": 0.1,
    "gradient_clip": 1.0,
    "learning_rate": 0.0003,
    "optimizer": "AdamW",
    "batch_size": 64,
    "seq_length": 128,
    "epochs": 50
}

with open('char_config.json', 'w') as f:
    json.dump(char_config, f, indent=2)

with open('word_config.json', 'w') as f:
    json.dump(word_config, f, indent=2)

print("\n✅ Configuration files created")
print("\n📊 Summary:")
print(f"   Character-level: {len(chars)} vocab, {X_train.shape[0]} train sequences")
print(f"   Word-level: {len(word_to_idx)} vocab, {X_train.shape[0]} train sequences")
print("\n🎯 Ready for training with task.md specifications!")