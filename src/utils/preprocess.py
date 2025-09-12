import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle


class CharTokenizer:
    """Character-level tokenizer for small corpus"""
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Special tokens
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        # Add characters
        for char in sorted(all_chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = len(self.char_to_idx)
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")
        
    def encode(self, text):
        """Convert text to indices"""
        indices = [self.char_to_idx.get(char, 1) for char in text]  # 1 is <UNK>
        return indices
    
    def decode(self, indices):
        """Convert indices back to text"""
        chars = [self.idx_to_char.get(idx, '<UNK>') for idx in indices]
        return ''.join(chars)
    
    def save(self, path):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))


class HorrorDataset(Dataset):
    """Dataset for horror stories"""
    def __init__(self, texts, tokenizer, seq_length=256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Encode all texts
        self.encoded_texts = []
        for text in texts:
            encoded = tokenizer.encode(text)
            if len(encoded) > 10:  # Skip very short texts
                self.encoded_texts.append(encoded)
        
        # Create sequences
        self.sequences = []
        for encoded in self.encoded_texts:
            # Add BOS and EOS tokens
            encoded = [2] + encoded + [3]  # <BOS> text <EOS>
            
            # Create overlapping sequences
            for i in range(0, len(encoded) - seq_length, seq_length // 2):
                seq = encoded[i:i + seq_length + 1]
                if len(seq) == seq_length + 1:
                    self.sequences.append(seq)
        
        print(f"Created {len(self.sequences)} sequences from {len(texts)} texts")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def load_and_split_data(json_path, train_ratio=0.8, val_ratio=0.1):
    """Load data and create train/val/test splits"""
    
    # Load posts
    with open(json_path, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    # Extract texts - handle both old format (title+follow_up) and new format (text)
    texts = []
    for post in posts:
        if 'text' in post:
            # New format from r/nosleep
            text = post['text']
        elif 'title' in post and 'follow_up' in post:
            # Old format from r/TwoSentenceHorror
            text = post['title'] + ' ' + post['follow_up']
        else:
            continue
            
        text = text.strip()
        if text and len(text.split()) >= 50:  # Minimum 50 words
            texts.append(text)
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(texts)
    
    # Split
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    print(f"Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    # Calculate total characters
    total_chars = sum(len(text) for text in texts)
    print(f"Total characters in corpus: {total_chars:,}")
    
    return train_texts, val_texts, test_texts


def create_data_loaders(train_texts, val_texts, test_texts, tokenizer, seq_length=256, batch_size=32):
    """Create DataLoaders for training"""
    
    train_dataset = HorrorDataset(train_texts, tokenizer, seq_length)
    val_dataset = HorrorDataset(val_texts, tokenizer, seq_length)
    test_dataset = HorrorDataset(test_texts, tokenizer, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import os
    import sys
    
    # Check if data is already prepared
    required_files = [
        'char_X_train.npy', 'char_y_train.npy',
        'char_X_val.npy', 'char_y_val.npy', 
        'char_X_test.npy', 'char_y_test.npy',
        'word_X_train.npy', 'word_y_train.npy',
        'word_X_val.npy', 'word_y_val.npy',
        'word_X_test.npy', 'word_y_test.npy',
        'char_vocab.json', 'word_vocab.pkl'
    ]
    
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        print("Data already prepared!")
        print("Found all required files:")
        for f in required_files:
            size = os.path.getsize(f) / (1024 * 1024)  # Size in MB
            print(f"  {f}: {size:.2f} MB")
        print("\nData is ready for training.")
    else:
        # Try to load from nosleep dataset if available
        if os.path.exists('nosleep_1939posts.json'):
            print("Loading nosleep dataset...")
            train_texts, val_texts, test_texts = load_and_split_data('nosleep_1939posts.json')
        elif os.path.exists('two_sentence_horror_2602posts.json'):
            print("Loading two_sentence_horror dataset...")
            train_texts, val_texts, test_texts = load_and_split_data('two_sentence_horror_2602posts.json')
        else:
            print("ERROR: No dataset found!")
            print("Missing files:")
            for f in required_files:
                if not os.path.exists(f):
                    print(f"  - {f}")
            sys.exit(1)
        
        # Create tokenizer
        tokenizer = CharTokenizer()
        tokenizer.fit(train_texts + val_texts + test_texts)
        tokenizer.save('tokenizer.pkl')
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_texts, val_texts, test_texts, tokenizer, seq_length=256, batch_size=32
        )
        
        # Test
        print(f"\nDataLoader test:")
        x, y = next(iter(train_loader))
        print(f"Input shape: {x.shape}, Target shape: {y.shape}")
        print(f"Sample text: {tokenizer.decode(x[0][:50].tolist())}")