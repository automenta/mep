"""
Tests for spectral norm constraints.
"""

import torch
import pytest
from mep import smep, sdmep


def test_spectral_constraint_scaling(device):
    """Test that spectral constraint correctly scales down large norms."""
    model = torch.nn.Linear(20, 20, bias=False).to(device)
    
    # Initialize with large weights
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(model.weight)
        S[0] = 10.0  # Force large singular value
        model.weight.copy_(U @ torch.diag(S) @ Vh)
    
    gamma = 0.95
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.1,
        gamma=gamma,
        mode='backprop'
    )
    
    # Run steps to enforce constraint
    for _ in range(10):
        x = torch.randn(4, 20, device=device)
        y = torch.randn(4, 20, device=device)
        output = model(x)
        loss = torch.nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
    
    # Check final spectral norm
    U_final, S_final, Vh_final = torch.linalg.svd(model.weight.detach())
    final_norm = S_final[0].item()
    
    # Should be close to gamma or less
    assert final_norm <= gamma + 0.05, f"Spectral norm {final_norm} > {gamma + 0.05}"


def test_spectral_constraint_no_change_if_small(device):
    """Test that spectral constraint does not affect small norms."""
    model = torch.nn.Linear(20, 20, bias=False).to(device)
    
    # Initialize with small weights
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(model.weight)
        S = S * 0.1  # Scale down
        model.weight.copy_(U @ torch.diag(S) @ Vh)
    
    orig_weight = model.weight.detach().clone()
    
    gamma = 1.0  # Larger than any SV
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.1,
        gamma=gamma,
        mode='backprop',
        weight_decay=0.0,
        momentum=0.0
    )
    
    # Zero gradients - no update should happen
    for _ in range(5):
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()
    
    # Should be unchanged (except maybe numerical noise)
    assert torch.allclose(model.weight, orig_weight, atol=1e-5)
