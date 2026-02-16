import torch
import torch.nn as nn
import pytest
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer, LocalEPMuon, NaturalEPMuon

@pytest.fixture
def simple_model(device):
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    ).to(device)
    return model

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

def test_smep_spectral_timing(device, simple_model):
    """Test that SMEPOptimizer runs with spectral_timing='during_settling'."""
    optimizer = SMEPOptimizer(
        simple_model.parameters(),
        model=simple_model,
        mode='ep',
        spectral_timing='during_settling',
        spectral_lambda=0.1,
        use_spectral_constraint=True,
        gamma=0.95
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    # Run step
    optimizer.step(x=x, target=y)

    # Check that it ran (no crash)

def test_local_ep_muon_step(device, simple_model):
    """Test LocalEPMuon updates parameters using local gradients."""
    optimizer = LocalEPMuon(
        simple_model.parameters(),
        model=simple_model,
        mode='ep',
        beta=0.1
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]

    optimizer.step(x=x, target=y)

    # Check if weights changed
    for p, p_old in zip(simple_model.parameters(), w_before):
        if p.requires_grad:
            assert not torch.allclose(p, p_old), f"Parameter {p.shape} did not update"
            assert p.grad is not None, "Gradient should be populated"

def test_natural_ep_muon_step(device, simple_model):
    """Test NaturalEPMuon updates parameters using Fisher approximation."""
    optimizer = NaturalEPMuon(
        simple_model.parameters(),
        model=simple_model,
        mode='ep',
        beta=0.1,
        fisher_approx='empirical'
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]

    optimizer.step(x=x, target=y)

    # Check update
    for p, p_old in zip(simple_model.parameters(), w_before):
        if p.requires_grad:
            assert not torch.allclose(p, p_old), f"Parameter {p.shape} did not update"
