import torch
import pytest
from mep.optim import SMEPOptimizer, SDMEPOptimizer

def test_smep_step(device):
    """Test that SMEPOptimizer takes a step and updates parameters."""
    # Simple Weight
    w = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True, device=device)
    w.grad = torch.ones_like(w)
    
    optimizer = SMEPOptimizer([w], lr=0.1)
    
    # Fake gradient
    w.grad = torch.randn_like(w)
    wd_before = w.clone()
    
    optimizer.step()
    
    # Check that parameters changed
    assert not torch.allclose(w, wd_before)

def test_sdmep_step(device):
    """Test that SDMEPOptimizer takes a step and updates parameters."""
    w = torch.randn(10, 10, requires_grad=True, device=device)
    optimizer = SDMEPOptimizer([w], lr=0.1)
    
    w.grad = torch.randn_like(w)
    wd_before = w.clone()
    
    optimizer.step()
    
    assert not torch.allclose(w, wd_before)

def test_sdmep_spectral_constraint(device):
    """Test that SDMEPOptimizer enforces spectral constraint."""
    # Create a parameter with large spectral norm
    # Singular values: 10, 1, 1...
    w = torch.zeros(10, 10, requires_grad=True, device=device)
    with torch.no_grad():
        w[0, 0] = 10.0
    
    # Target gamma = 1.0 (default is 0.95 in optimizer, let's use default)
    # Spectral norm is 10.0. Gamma is 0.95.
    # Optimizer should scale it down to ~0.95.
    
    optimizer = SDMEPOptimizer([w], lr=0.1, gamma=0.95)
    
    # We need a grad to trigger step, but we want to test the constraint enforcement
    # which happens at the end of step.
    # If we set grad=0, momentum buffer is 0. Update is 0.
    # Then it enforces constraint on p.data.
    
    w.grad = torch.zeros_like(w)
    
    # We need to call step multiple times because spectral norm estimation is iterative
    # but initially it uses random noise so might not capture max SV immediately.
    # But for a diagonal matrix, it should be fast.
    
    for _ in range(5):
        optimizer.step()
        
    # Check spectral norm
    U, S, V = torch.linalg.svd(w.detach())
    spectral_norm = S[0].item()
    
    # Should be <= gamma (approx)
    # The constraint: if sigma > gamma: p.mul(gamma/sigma) -> new sigma = gamma
    assert spectral_norm <= 0.96 # Allow small tolerance
