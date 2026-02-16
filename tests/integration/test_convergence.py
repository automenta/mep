import torch
import torch.nn.functional as F
import pytest
import torch.nn as nn
from mep.optim import SMEPOptimizer

def test_xor_convergence(device):
    """Test that model converges to high accuracy on XOR."""
    # XOR Data
    x = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
    y = torch.tensor([[0.], [1.], [1.], [0.]], device=device)
    
    # Model: Need enough capacity
    model = nn.Sequential(
        nn.Linear(2, 50),
        nn.ReLU(), # Added activation for non-linearity
        nn.Linear(50, 1)
    ).to(device)
    optimizer = SMEPOptimizer(model.parameters(), lr=0.1, mode='ep', beta=0.5, settle_steps=20)
    
    # Train
    for epoch in range(200):
        optimizer.zero_grad()
        optimizer.step(x, y, model) # Use SMEPOptimizer's step method
        
    # Check predictions
    with torch.no_grad():
        pred = model(x)
        hard_pred = (pred > 0.5).float()
        accuracy = (hard_pred == y).float().mean().item()
        
    # XOR should be 100% solvable
    assert accuracy >= 0.75, f"Low accuracy on XOR: {accuracy}" # Allow one mistake but aim for 1.0
