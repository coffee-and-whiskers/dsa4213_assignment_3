import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMModel(nn.Module):
    """LSTM Language Model"""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        batch_size = x.size(0)
        
        # Embedding
        embeds = self.embedding(x)  # (batch, seq_len, embed_size)
        embeds = self.dropout(embeds)
        
        # LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        lstm_out, hidden = self.lstm(embeds, hidden)  # (batch, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)  # (batch, seq_len, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


class RNNModel(nn.Module):
    """Vanilla RNN Language Model"""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         nonlinearity='tanh')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        batch_size = x.size(0)
        
        # Embedding
        embeds = self.embedding(x)  # (batch, seq_len, embed_size)
        embeds = self.dropout(embeds)
        
        # RNN
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        rnn_out, hidden = self.rnn(embeds, hidden)  # (batch, seq_len, hidden_size)
        rnn_out = self.dropout(rnn_out)
        
        # Output layer
        output = self.fc(rnn_out)  # (batch, seq_len, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerDecoder(nn.Module):
    """Simple Transformer Decoder for Language Modeling"""
    def __init__(self, vocab_size, embed_size=128, num_heads=8, hidden_size=256, 
                 num_layers=2, dropout=0.2, max_len=5000):
        super(TransformerDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a mask for autoregressive decoding"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, memory=None):
        # x shape: (batch, seq_len)
        seq_len = x.size(1)
        
        # Create attention mask for autoregressive generation
        device = x.device
        mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_size)  # (batch, seq_len, embed_size)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_size)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_size)
        x = self.dropout(x)
        
        # Since we're doing language modeling without encoder, we use self-attention only
        # The memory would be from an encoder in a full transformer
        if memory is None:
            # For pure decoder (language modeling), we don't use cross-attention
            # So we pass the input as both tgt and memory
            output = self.transformer_decoder(
                tgt=x,
                memory=x,  # In pure LM, we don't have separate memory
                tgt_mask=mask,
                memory_mask=mask
            )
        else:
            output = self.transformer_decoder(x, memory, tgt_mask=mask)
        
        # Output projection
        output = self.fc(output)  # (batch, seq_len, vocab_size)
        
        return output, None  # Return None for hidden state (compatibility with RNN models)


class SimpleTransformerLM(nn.Module):
    """Simplified Transformer for Language Modeling (decoder-only)"""
    def __init__(self, vocab_size, embed_size=128, num_heads=4, hidden_size=256, 
                 num_layers=2, dropout=0.2, max_len=512):
        super(SimpleTransformerLM, self).__init__()
        
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token and positional embeddings
        token_embeds = self.embedding(x)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(pos_ids)
        
        x = self.dropout(token_embeds + pos_embeds)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        output = self.fc(x)
        
        return output, None


class TransformerBlock(nn.Module):
    """A single transformer block"""
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with causal mask
        seq_len = x.size(1)
        device = x.device
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        
        return x