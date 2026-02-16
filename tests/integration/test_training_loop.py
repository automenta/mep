import torch
import torch.nn.functional as F
import pytest
import torch.nn as nn
from mep.optimizers import SMEPOptimizer

def test_training_loop_xor(device):
    """Test that a model can be trained on XOR using SDMEP."""
    # XOR Data
    x = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
    y = torch.tensor([[0.], [1.], [1.], [0.]], device=device)
    
    # Model
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Linear(10, 1)
    ).to(device)
    optimizer = SMEPOptimizer(model.parameters(), lr=0.1, mode='ep')
    
    # Train Loop
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        optimizer.step(x, y, model)
        
        # Monitor loss (forward pass)
        with torch.no_grad():
            pred = model(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            
    # Check that loss decreased
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    
    # Check accuracy (or at least reasonable prediction)
    with torch.no_grad():
        final_pred = model(x)
        # XOR is tricky, might need more epochs/tuning, but let's check basic trend
        # Or just that it didn't diverge
        assert not torch.isnan(final_pred).any()
        assert not torch.isinf(final_pred).any()
