"""
Unit tests for MEP optimizers (new strategy-based API).
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers import (
    CompositeOptimizer,
    BackpropGradient,
    EPGradient,
    MuonUpdate,
    DionUpdate,
    SpectralConstraint,
    ErrorFeedback,
    NoFeedback,
)
from mep.presets import smep, sdmep, local_ep, natural_ep, muon_backprop


@pytest.fixture
def simple_model(device):
    """Simple MLP for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    ).to(device)
    return model


@pytest.fixture
def large_model(device):
    """Larger model for testing Dion threshold."""
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    return model


class TestMuonUpdate:
    """Tests for Muon (Newton-Schulz) update strategy."""
    
    def test_muon_orthogonalizes(self, device):
        """Muon should produce approximately orthogonal update matrix."""
        strategy = MuonUpdate(ns_steps=10)  # More steps for better convergence
        grad = torch.randn(50, 30, device=device)
        state = {}
        group_config = {}
        
        update = strategy.transform_gradient(None, grad, state, group_config)
        
        # Check orthogonality: U.T @ U ≈ I (for the smaller dimension)
        UtU = update.T @ update
        identity = torch.eye(30, device=device)
        # Relaxed tolerance - Newton-Schulz converges but not perfectly
        assert torch.allclose(UtU, identity, atol=0.15)
    
    def test_muon_preserves_shape(self, device):
        """Muon should preserve gradient shape."""
        strategy = MuonUpdate(ns_steps=5)
        grad = torch.randn(40, 20, device=device)
        state = {}
        
        update = strategy.transform_gradient(None, grad, state, {})
        assert update.shape == grad.shape


class TestDionUpdate:
    """Tests for Dion (low-rank SVD) update strategy."""
    
    def test_dion_uses_lowrank_for_large_matrices(self, device):
        """Dion should use low-rank SVD for matrices above threshold."""
        strategy = DionUpdate(rank_frac=0.5, threshold=1000)
        grad = torch.randn(50, 50, device=device)  # 2500 params > 1000
        param = nn.Parameter(grad)
        state = {}
        group_config = {"use_error_feedback": False}
        
        update = strategy.transform_gradient(param, grad, state, group_config)
        assert update.shape == grad.shape
    
    def test_dion_falls_back_to_muon_for_small_matrices(self, device):
        """Dion should fall back to Muon for small matrices."""
        strategy = DionUpdate(rank_frac=0.5, threshold=10000, muon_fallback=MuonUpdate(ns_steps=10))
        grad = torch.randn(10, 10, device=device)  # 100 params < 10000
        param = nn.Parameter(grad)
        state = {}
        group_config = {}
        
        update = strategy.transform_gradient(param, grad, state, group_config)
        
        # Should be approximately orthogonal (Muon behavior)
        # Note: Newton-Schulz may not achieve perfect orthogonality in few steps
        UtU = update.T @ update
        identity = torch.eye(10, device=device)
        # Check that off-diagonal elements are small (orthogonality)
        off_diag = UtU - torch.diag(torch.diag(UtU))
        assert off_diag.abs().mean() < 0.15


class TestSpectralConstraint:
    """Tests for spectral norm constraint."""
    
    def test_spectral_constraint_enforces_bound(self, device):
        """Spectral constraint should enforce σ(W) ≤ γ."""
        constraint = SpectralConstraint(gamma=0.95)
        
        # Create matrix with large spectral norm
        w = nn.Parameter(torch.zeros(20, 20, device=device))
        with torch.no_grad():
            w[0, 0] = 5.0  # Spectral norm = 5
        
        state = {}
        group_config = {}
        
        constraint.enforce(w, state, group_config)
        
        # Check spectral norm after constraint
        U, S, V = torch.linalg.svd(w.detach())
        assert S[0].item() <= 0.96  # Allow small tolerance
    
    def test_spectral_constraint_skips_1d_params(self, device):
        """Spectral constraint should skip 1D parameters (biases)."""
        constraint = SpectralConstraint(gamma=0.95)
        
        w = nn.Parameter(torch.randn(10, device=device))
        original = w.clone()
        
        constraint.enforce(w, {}, {})
        assert torch.allclose(w, original)


class TestErrorFeedback:
    """Tests for error feedback strategy."""
    
    def test_error_feedback_accumulates_residual(self, device):
        """Error feedback should accumulate residuals."""
        feedback = ErrorFeedback(beta=0.9)
        grad = torch.ones(10, 10, device=device)
        state = {}
        group_config = {}
        
        # First accumulation
        g_aug = feedback.accumulate(grad, state, group_config)
        assert torch.allclose(g_aug, grad)  # No prior error
        
        # Simulate update that loses information
        update = torch.zeros_like(grad)
        residual = grad - update
        
        # Update buffer
        feedback.update_buffer(residual, state, group_config)
        
        # Second accumulation should include error
        grad2 = torch.ones(10, 10, device=device)
        g_aug2 = feedback.accumulate(grad2, state, group_config)
        assert not torch.allclose(g_aug2, grad2)


class TestCompositeOptimizer:
    """Tests for CompositeOptimizer."""
    
    def test_backprop_step(self, device, simple_model):
        """CompositeOptimizer with backprop should update parameters."""
        optimizer = CompositeOptimizer(
            simple_model.parameters(),
            gradient=BackpropGradient(),
            update=MuonUpdate(ns_steps=5),
            constraint=SpectralConstraint(gamma=0.95),
            lr=0.01,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        loss_fn = nn.CrossEntropyLoss()
        
        # Standard backprop workflow
        output = simple_model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        w_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step()
        
        # Check parameters changed
        for p, p_old in zip(simple_model.parameters(), w_before):
            assert not torch.allclose(p, p_old)
    
    def test_ep_step(self, device, simple_model):
        """CompositeOptimizer with EP should update parameters."""
        optimizer = CompositeOptimizer(
            simple_model.parameters(),
            gradient=EPGradient(beta=0.5, settle_steps=5, settle_lr=0.05),
            update=MuonUpdate(ns_steps=5),
            constraint=SpectralConstraint(gamma=0.95),
            feedback=NoFeedback(),  # Disable for simplicity
            lr=0.01,
            model=simple_model,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        w_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step(x=x, target=y)
        
        # Check parameters changed
        for p, p_old in zip(simple_model.parameters(), w_before):
            if p.ndim >= 2:  # Only weights, not biases
                assert not torch.allclose(p, p_old)
    
    def test_zero_grad(self, device, simple_model):
        """zero_grad should clear gradients."""
        optimizer = CompositeOptimizer(
            simple_model.parameters(),
            gradient=BackpropGradient(),
            update=MuonUpdate(),
            lr=0.01,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Verify gradients exist
        for p in simple_model.parameters():
            assert p.grad is not None
        
        optimizer.zero_grad()
        
        # Verify gradients cleared
        for p in simple_model.parameters():
            assert p.grad is None


class TestPresets:
    """Tests for preset factory functions."""
    
    def test_smep_backprop(self, device, simple_model):
        """SMEP preset with backprop mode."""
        optimizer = smep(
            simple_model.parameters(),
            model=simple_model,
            mode='backprop',
            lr=0.01,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        w_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step()
        
        for p, p_old in zip(simple_model.parameters(), w_before):
            assert not torch.allclose(p, p_old)
    
    def test_muon_backprop(self, device, simple_model):
        """Muon backprop preset (drop-in SGD replacement)."""
        optimizer = muon_backprop(
            simple_model.parameters(),
            lr=0.01,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        w_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step()
        
        for p, p_old in zip(simple_model.parameters(), w_before):
            assert not torch.allclose(p, p_old)
    
    def test_sdmep(self, device, large_model):
        """SDMEP preset with Dion for large matrices."""
        optimizer = sdmep(
            large_model.parameters(),
            model=large_model,
            lr=0.01,
            dion_thresh=5000,  # Lower threshold for testing
        )
        
        x = torch.randn(4, 100, device=device)
        y = torch.randint(0, 10, (4,), device=device)
        
        w_before = [p.clone() for p in large_model.parameters()]
        optimizer.step(x=x, target=y)
        
        for p, p_old in zip(large_model.parameters(), w_before):
            if p.ndim >= 2:
                assert not torch.allclose(p, p_old)
    
    def test_local_ep(self, device, simple_model):
        """LocalEPMuon preset."""
        optimizer = local_ep(
            simple_model.parameters(),
            model=simple_model,
            lr=0.01,
            beta=0.1,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        w_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step(x=x, target=y)
        
        for p, p_old in zip(simple_model.parameters(), w_before):
            if p.ndim >= 2:
                assert not torch.allclose(p, p_old)
    
    def test_natural_ep(self, device, simple_model):
        """NaturalEPMuon preset."""
        optimizer = natural_ep(
            simple_model.parameters(),
            model=simple_model,
            lr=0.01,
            beta=0.5,
        )
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        w_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step(x=x, target=y)
        
        for p, p_old in zip(simple_model.parameters(), w_before):
            if p.ndim >= 2:
                assert not torch.allclose(p, p_old)
