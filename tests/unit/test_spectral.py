import torch
import pytest
from mep.optim import SDMEPOptimizer

def test_spectral_constraint_scaling(device):
    """Test that spectral constraint correctly scales down large norms."""
    # Create a parameter with large spectral norm
    w = torch.randn(20, 20, device=device)
    U, S, Vh = torch.linalg.svd(w)
    S[0] = 10.0 # Force large singular value
    w = U @ torch.diag(S) @ Vh
    w.requires_grad_(True)
    
    gamma = 0.95
    optimizer = SDMEPOptimizer([w], lr=0.1, gamma=gamma)
    w.grad = torch.zeros_like(w)
    
    # Run steps to enforce constraint
    # Initial estimate of u, v is random, so might take a few steps to converge to top SV
    # But for 10.0 vs others (random ~1-2), it should be quick.
    
    for _ in range(10):
        optimizer.step()
        
    # Check final spectral norm
    U_final, S_final, Vh_final = torch.linalg.svd(w.detach())
    final_norm = S_final[0].item()
    
    # Should be close to gamma or less
    assert final_norm <= gamma + 0.05 # Allow small tolerance

def test_spectral_constraint_no_change_if_small(device):
    """Test that spectral constraint does not affect small norms."""
    # Create a parameter with small spectral norm
    w = torch.randn(20, 20, device=device)
    U, S, Vh = torch.linalg.svd(w)
    S = S * 0.1 # Scale down
    w = U @ torch.diag(S) @ Vh
    w.requires_grad_(True)
    
    orig_w = w.detach().clone()
    
    gamma = 1.0 # Larger than any SV
    optimizer = SDMEPOptimizer([w], lr=0.1, gamma=gamma, wd=0.0, momentum=0.0)
    w.grad = torch.zeros_like(w) # No gradient update
    
    for _ in range(5):
        optimizer.step()
        
    # Should be unchanged (except maybe numerical noise)
    assert torch.allclose(w, orig_w, atol=1e-5)
