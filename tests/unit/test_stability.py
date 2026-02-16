import torch
import torch.nn as nn
import pytest
from mep.optim import SMEPOptimizer, SDMEPOptimizer

def test_batch_size_energy_consistency(device):
    """Test that energy-per-sample is consistent across batch sizes.
    
    Note: EP gradients may vary slightly across batch sizes due to different
    settling dynamics, but the energy function should scale correctly.
    """
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    
    # Fix random seed
    torch.manual_seed(42)
    for p in model.parameters():
        p.data.normal_(0, 0.1)
    
    # Create base data
    base_x = torch.randn(16, 10).to(device)
    base_target = torch.randint(0, 5, (16,)).to(device)
    
    optimizer = SMEPOptimizer(model.parameters(), mode='ep', beta=0.5, settle_steps=10)
    
    # Test different batch sizes
    energies_per_sample = []
    for bs in [8, 16]:
        x = base_x[:bs]
        target = base_target[:bs]
        
        # Get settled states
        states = optimizer._settle(model, x, target=target, beta=0.5)
        
        # Compute energy
        structure = optimizer._inspect_model(model)
        target_vec = optimizer._prepare_target(target, states[-1].shape[-1])
        energy = optimizer._compute_energy(model, x, states, structure, target_vec, beta=0.5)
        
        # Energy should be per-sample (normalized by batch_size internally)
        energies_per_sample.append(energy.item())
    
    # Energies should be similar (both are per-sample averages)
    # Allow some variance since different batch sizes may settle differently
    rel_diff = abs(energies_per_sample[0] - energies_per_sample[1]) / (abs(energies_per_sample[0]) + 1e-8)
    assert rel_diff < 0.5, \
        f"Energy per sample varies too much: {energies_per_sample}, rel_diff={rel_diff:.3f}"

def test_settling_convergence(device):
    """Test that energy monotonically decreases during settling."""
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
    ).to(device)
    
    optimizer = SMEPOptimizer(model.parameters(), mode='ep', settle_steps=20)
    
    x = torch.randn(4, 5).to(device)
    target = torch.randint(0, 3, (4,)).to(device)
    
    # Manually track energy during settling by calling _settle with increasing steps
    energies = []
    for steps in [1, 5, 10, 15, 20]:
        optimizer.defaults['settle_steps'] = steps
        states = optimizer._settle(model, x, target=target, beta=0.5)
        
        # Compute energy of settled states
        structure = optimizer._inspect_model(model)
        target_vec = optimizer._prepare_target(target, states[-1].shape[-1])
        energy = optimizer._compute_energy(model, x, states, structure, target_vec, beta=0.5)
        energies.append(energy.item())
    
    # Verify energy decreases (allowing small numerical fluctuations)
    for i in range(1, len(energies)):
        # Energy should decrease or stay similar (within 1% tolerance for numerical noise)
        assert energies[i] <= energies[i-1] * 1.01, \
            f"Energy increased during settling: {energies}"

def test_gradient_magnitude_sanity(device):
    """Test that EP gradients have reasonable magnitudes."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    
    optimizer = SMEPOptimizer(model.parameters(), mode='ep', beta=0.5)
    
    x = torch.randn(16, 10).to(device)
    target = torch.randint(0, 5, (16,)).to(device)
    
    optimizer.zero_grad()
    optimizer.step(x, target, model)
    
    # Check that gradients exist and are finite
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        
        # Check reasonable magnitude (not too small, not too large)
        grad_norm = p.grad.norm().item()
        assert 1e-6 < grad_norm < 1e3, \
            f"Gradient norm out of reasonable range for {name}: {grad_norm}"

def test_smep_vs_sdmep_consistency(device):
    """Test that SMEP and SDMEP produce similar results on small matrices."""
    # Use small matrices so both use Muon (not Dion)
    model_smep = nn.Sequential(nn.Linear(5, 3)).to(device)
    model_sdmep = nn.Sequential(nn.Linear(5, 3)).to(device)
    
    # Copy weights
    model_sdmep.load_state_dict(model_smep.state_dict())
    
    opt_smep = SMEPOptimizer(model_smep.parameters(), mode='ep')
    opt_sdmep = SDMEPOptimizer(model_sdmep.parameters(), mode='ep', dion_thresh=1000000)
    
    x = torch.randn(4, 5).to(device)
    target = torch.tensor([0, 1, 2, 1]).to(device)
    
    # Single step
    opt_smep.zero_grad()
    opt_smep.step(x, target, model_smep)
    
    opt_sdmep.zero_grad()
    opt_sdmep.step(x, target, model_sdmep)
    
    # Check weights are similar (should be identical for small matrices)
    for (n1, p1), (n2, p2) in zip(model_smep.named_parameters(), model_sdmep.named_parameters()):
        diff = (p1 - p2).abs().max().item()
        assert diff < 1e-5, f"SMEP and SDMEP diverged on {n1}: max diff = {diff}"

@pytest.mark.slow
@pytest.mark.xfail(reason="FP16 support requires deeper changes to state settling mechanism")
def test_mixed_precision_stability(device):
    """Test that optimizer works with half precision (FP16)."""
    if device == 'cpu':
        pytest.skip("FP16 not well supported on CPU")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device).half()
    
    optimizer = SMEPOptimizer(model.parameters(), mode='ep', beta=0.5, settle_steps=10)
    
    x = torch.randn(8, 10).to(device).half()
    target = torch.randint(0, 5, (8,)).to(device)
    
    # Multiple steps to check stability
    for _ in range(5):
        optimizer.zero_grad()
        try:
            optimizer.step(x, target, model)
        except Exception as e:
            pytest.fail(f"FP16 training failed: {e}")
        
        # Check for NaN/Inf
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Non-finite gradients in FP16 mode"
            assert torch.isfinite(p).all(), "Non-finite parameters in FP16 mode"
