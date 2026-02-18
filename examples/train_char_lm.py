#!/usr/bin/env python3
"""
Character-level Language Model with Equilibrium Propagation

This example demonstrates EP training on a simple character-level LM task.
Unlike classification, LM training shows different dynamics:
- Sequential prediction rather than single-label classification
- Energy-based formulation may offer different convergence properties
- Local learning rules could affect how context is learned

Run: python examples/train_char_lm.py
"""

import torch
import torch.nn as nn
import time
from typing import Tuple

from mep import smep, muon_backprop


class CharLM(nn.Module):
    """Simple character-level language model."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embed(x)
        hidden, _ = self.rnn(embed)
        return self.head(hidden)


def load_shakespeare() -> Tuple[str, dict, dict]:
    """Load Shakespeare text and create vocab."""
    # Simple Shakespeare corpus (subset)
    text = """
    ROMEO: But, soft! what light through yonder window breaks?
    It is the east, and Juliet is the sun.
    Arise, fair sun, and kill the envious moon,
    Who is already sick and pale with grief,
    That thou her maid art far more fair than she.
    
    JULIET: O Romeo, Romeo! wherefore art thou Romeo?
    Deny thy father and refuse thy name;
    Or, if thou wilt not, be but sworn my love,
    And I'll no longer be a Capulet.
    
    ROMEO: I take thee at thy word.
    Call me but love, and I'll be new baptized;
    Henceforth I never will be Romeo.
    
    JULIET: What man art thou that thus bescreened in night
    So stumblest on my counsel?
    
    ROMEO: By a name
    I know not how to tell thee who I am.
    """ * 10  # Repeat for more data
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return text, char_to_idx, idx_to_char


def create_batches(text: str, char_to_idx: dict, seq_len: int, batch_size: int):
    """Create training batches."""
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    
    # Create sequences
    sequences = []
    targets = []
    for i in range(0, len(data) - seq_len, batch_size):
        seq = data[i:i+seq_len]
        tgt = data[i+1:i+seq_len+1]
        if len(seq) == seq_len:
            sequences.append(seq)
            targets.append(tgt)
    
    return torch.stack(sequences), torch.stack(targets)


def train_epoch_ep(model, optimizer, sequences, targets, device):
    """Train one epoch with EP."""
    model.train()
    total_loss = 0
    
    for seq, tgt in zip(sequences, targets):
        seq, tgt = seq.to(device), tgt.to(device)
        
        # EP step - predict next character
        optimizer.step(x=seq, target=tgt)
        optimizer.zero_grad()
    
    # Compute loss for reporting
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for seq, tgt in zip(sequences, targets):
            seq, tgt = seq.to(device), tgt.to(device)
            logits = model(seq)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(sequences)


def train_epoch_bp(model, optimizer, sequences, targets, device):
    """Train one epoch with backprop."""
    model.train()
    total_loss = 0
    
    for seq, tgt in zip(sequences, targets):
        seq, tgt = seq.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        logits = model(seq)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
    
    return total_loss / len(sequences)


def generate_text(model, char_to_idx, idx_to_char, seed_text: str, max_len: int = 100, device='cpu'):
    """Generate text from model."""
    model.eval()
    
    # Encode seed
    chars = list(seed_text)
    
    with torch.no_grad():
        for _ in range(max_len):
            seq = torch.tensor([[char_to_idx.get(ch, 0) for ch in chars[-20:]]], device=device)
            logits = model(seq)
            next_char_idx = logits[0, -1].argmax().item()
            chars.append(idx_to_char.get(next_char_idx, ' '))
    
    return ''.join(chars)


def main():
    print("=" * 60)
    print("Character-level Language Model: EP vs Backprop")
    print("=" * 60)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 3
    SEQ_LEN = 32
    BATCH_SIZE = 16
    
    # Load data
    print("\nLoading Shakespeare corpus...")
    text, char_to_idx, idx_to_char = load_shakespeare()
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size} characters")
    print(f"Corpus length: {len(text)} characters")
    
    # Create batches
    sequences, targets = create_batches(text, char_to_idx, SEQ_LEN, BATCH_SIZE)
    print(f"Training sequences: {len(sequences)}")
    
    # Train with Backprop
    print("\n" + "-" * 60)
    print("Training with Backpropagation")
    print("-" * 60)
    
    model_bp = CharLM(vocab_size).to(DEVICE)
    opt_bp = muon_backprop(model_bp.parameters(), lr=0.01)
    
    start = time.time()
    for epoch in range(EPOCHS):
        loss = train_epoch_bp(model_bp, opt_bp, sequences, targets, DEVICE)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss:.3f}, Time={elapsed:.1f}s")
    
    print("\nGenerated text (backprop):")
    print(generate_text(model_bp, char_to_idx, idx_to_char, "ROMEO: ", device=DEVICE))
    
    # Train with EP
    print("\n" + "-" * 60)
    print("Training with Equilibrium Propagation")
    print("-" * 60)
    
    model_ep = CharLM(vocab_size).to(DEVICE)
    opt_ep = smep(
        model_ep.parameters(),
        model=model_ep,
        lr=0.005,  # Lower LR for EP
        mode='ep',
        settle_steps=10,
        loss_type='cross_entropy',
    )
    
    start = time.time()
    for epoch in range(EPOCHS):
        loss = train_epoch_ep(model_ep, opt_ep, sequences, targets, DEVICE)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss:.3f}, Time={elapsed:.1f}s")
    
    print("\nGenerated text (EP):")
    print(generate_text(model_ep, char_to_idx, idx_to_char, "ROMEO: ", device=DEVICE))
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("- EP trains without backpropagation through time")
    print("- Energy-based formulation may affect learning dynamics")
    print("- Generated text quality depends on training duration")
    print("=" * 60)


if __name__ == "__main__":
    main()
