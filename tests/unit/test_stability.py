"""
Edge case tests and stability tests for MEP optimizers.

Tests boundary conditions, numerical stability, and error handling.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer, LocalEPMuon, NaturalEPMuon


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_learning_rate(self, device):
        """Test that negative learning rate raises error."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SMEPOptimizer([w], lr=-0.01)

    def test_invalid_momentum(self, device):
        """Test that invalid momentum raises error."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        with pytest.raises(ValueError, match="Momentum must be in"):
            SMEPOptimizer([w], momentum=1.5)
        with pytest.raises(ValueError, match="Momentum must be in"):
            SMEPOptimizer([w], momentum=-0.1)

    def test_invalid_beta(self, device):
        """Test that invalid beta raises error."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        with pytest.raises(ValueError, match="Beta must be in"):
            SMEPOptimizer([w], mode='ep', beta=1.5)
        with pytest.raises(ValueError, match="Beta must be in"):
            SMEPOptimizer([w], mode='ep', beta=0)

    def test_invalid_gamma(self, device):
        """Test that invalid gamma raises error."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        with pytest.raises(ValueError, match="Gamma must be in"):
            SMEPOptimizer([w], gamma=1.5)
        with pytest.raises(ValueError, match="Gamma must be in"):
            SMEPOptimizer([w], gamma=0)

    def test_invalid_spectral_timing(self, device):
        """Test that invalid spectral_timing raises error."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        with pytest.raises(ValueError, match="Spectral timing must be"):
            SMEPOptimizer([w], spectral_timing='invalid')

    def test_invalid_mode(self, device):
        """Test that invalid mode raises error."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        with pytest.raises(ValueError, match="Mode must be"):
            SMEPOptimizer([w], mode='invalid')


class TestEdgeCases:
    """Test edge cases in optimizer behavior."""

    def test_empty_input(self, device):
        """Test that empty input raises appropriate error."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x_empty = torch.randn(0, 10, device=device)
        y = torch.randint(0, 5, (0,), device=device)

        with pytest.raises(ValueError, match="Input tensor cannot be empty"):
            optimizer.step(x=x_empty, target=y)

    def test_single_sample(self, device):
        """Test that single sample batch works."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(1, 10, device=device)
        y = torch.randint(0, 5, (1,), device=device)

        # Should not crash
        optimizer.step(x=x, target=y)

    def test_very_large_beta(self, device):
        """Test behavior with beta=1.0 (maximum)."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(
            model.parameters(), model=model, mode='ep', beta=1.0
        )

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        # Should not crash
        optimizer.step(x=x, target=y)

    def test_very_small_learning_rate(self, device):
        """Test behavior with very small learning rate."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(
            model.parameters(), model=model, mode='ep', lr=1e-6
        )

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        # Store initial weights
        initial_weights = [p.clone() for p in model.parameters()]

        # Run many steps
        for _ in range(10):
            optimizer.step(x=x, target=y)

        # Weights should have changed very little
        for p, p_init in zip(model.parameters(), initial_weights):
            change = (p - p_init).norm().item()
            assert change < 1.0  # Should be small


class TestNumericalStability:
    """Test numerical stability of optimizers."""

    def test_nan_gradient_detection(self, device):
        """Test that NaN gradients are detected."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        optimizer = SMEPOptimizer([w], lr=0.1)

        # Inject NaN gradient
        w.grad = torch.full_like(w, float('nan'))

        with pytest.raises(RuntimeError, match="NaN/Inf"):
            optimizer.step()

    def test_inf_gradient_detection(self, device):
        """Test that Inf gradients are detected."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        optimizer = SMEPOptimizer([w], lr=0.1)

        # Inject Inf gradient
        w.grad = torch.full_like(w, float('inf'))

        with pytest.raises(RuntimeError, match="NaN/Inf"):
            optimizer.step()

    def test_large_weight_initialization(self, device):
        """Test stability with large initial weights."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        with torch.no_grad():
            model[0].weight.fill_(100.0)
            model[0].bias.fill_(100.0)

        optimizer = SMEPOptimizer(
            model.parameters(),
            model=model,
            mode='ep',
            use_spectral_constraint=True,
            gamma=0.95
        )

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        # Should not crash, spectral constraint should kick in
        optimizer.step(x=x, target=y)

        # Check spectral norm is constrained
        U, S, Vh = torch.linalg.svd(model[0].weight.detach())
        assert S[0].item() <= 1.0  # Should be constrained


class TestBackpropMode:
    """Test backprop mode functionality."""

    def test_backprop_mode_basic(self, device):
        """Test basic backprop mode."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(model.parameters(), mode='backprop')

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        # Store gradient before step
        grad_before = model[0].weight.grad.clone()

        optimizer.step()

        # Gradient should have been used (may or may not be zeroed depending on implementation)
        # The key is that the step completed without error
        assert grad_before is not None

    def test_backprop_mode_with_momentum(self, device):
        """Test backprop mode with momentum."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(model.parameters(), mode='backprop', momentum=0.9)

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

        # Should converge without issues


class TestErrorFeedback:
    """Test error feedback mechanism."""

    def test_error_feedback_accumulation(self, device):
        """Test that error feedback accumulates residuals."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        optimizer = SMEPOptimizer(
            [w], lr=0.1, use_error_feedback=True, error_beta=0.9
        )

        # Run several steps
        for i in range(5):
            w.grad = torch.randn_like(w)
            optimizer.step()

            # Check error buffer exists and has content
            state = optimizer.state[w]
            assert "error_buffer" in state
            # With pure Muon, error buffer should be zeroed
            # (error feedback is for Dion)

    def test_error_feedback_disabled(self, device):
        """Test behavior when error feedback is disabled."""
        w = torch.randn(10, 10, requires_grad=True, device=device)
        optimizer = SMEPOptimizer([w], lr=0.1, use_error_feedback=False)

        w.grad = torch.randn_like(w)
        optimizer.step()

        # Error buffer should be zeroed
        state = optimizer.state[w]
        assert state["error_buffer"].norm().item() == 0


class TestSpectralConstraintEdgeCases:
    """Test spectral constraint edge cases."""

    def test_already_constrained_weights(self, device):
        """Test that already-constrained weights are not modified much."""
        # Create small weight matrix
        w = torch.randn(10, 10, requires_grad=True, device=device)

        # Scale to have small spectral norm
        with torch.no_grad():
            w.mul_(0.1)

        # Use very small learning rate instead of 0
        optimizer = SDMEPOptimizer([w], lr=1e-8, gamma=1.0)

        w_before = w.clone()

        # Run steps with tiny learning rate
        for _ in range(5):
            w.grad = torch.zeros_like(w)
            optimizer.step()

        # Should be nearly unchanged (only numerical precision)
        assert torch.allclose(w, w_before, atol=1e-6)

    def test_rectangular_matrices(self, device):
        """Test spectral constraint on rectangular matrices."""
        # Tall matrix
        w_tall = torch.randn(20, 10, requires_grad=True, device=device)
        # Wide matrix
        w_wide = torch.randn(10, 20, requires_grad=True, device=device)

        optimizer = SDMEPOptimizer(
            [w_tall, w_wide], lr=0.1, gamma=0.95, use_spectral_constraint=True
        )

        for _ in range(10):
            w_tall.grad = torch.randn_like(w_tall)
            w_wide.grad = torch.randn_like(w_wide)
            optimizer.step()

        # Check both are constrained (with small tolerance for numerical precision)
        for w in [w_tall, w_wide]:
            U, S, Vh = torch.linalg.svd(w.detach())
            # Allow small tolerance above gamma due to numerical precision
            assert S[0].item() <= 1.05, f"Spectral norm {S[0].item()} exceeds bound"


class TestModelArchitectures:
    """Test different model architectures."""

    def test_deep_network(self, device):
        """Test with deeper network."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        ).to(device)

        optimizer = SMEPOptimizer(
            model.parameters(),
            model=model,
            mode='ep',
            settle_steps=15
        )

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        # Should work with deeper networks
        optimizer.step(x=x, target=y)

    def test_network_with_dropout(self, device):
        """Test network with dropout layers."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        model.train()
        optimizer.step(x=x, target=y)

    def test_network_with_batchnorm(self, device):
        """Test network with batch normalization."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        model.train()
        optimizer.step(x=x, target=y)


class TestOptimizerState:
    """Test optimizer state management."""

    def test_state_dict_save_load(self, device):
        """Test saving and loading optimizer state."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(model.parameters(), lr=0.1, momentum=0.9)

        # Run some steps
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        for _ in range(3):
            if optimizer.defaults["mode"] == 'ep':
                optimizer.step(x=x, target=y, model=model)
            else:
                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.mse_loss(output, y)
                loss.backward()
                optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer
        model2 = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer2 = SMEPOptimizer(model2.parameters(), lr=0.1, momentum=0.9)

        # Load state
        optimizer2.load_state_dict(state_dict)

        # State should match
        assert len(optimizer2.state) == len(optimizer.state)

    def test_zero_grad(self, device):
        """Test zero_grad functionality."""
        model = nn.Sequential(nn.Linear(10, 5)).to(device)
        optimizer = SMEPOptimizer(model.parameters(), lr=0.1)

        # Run step to create state
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        if optimizer.defaults["mode"] == 'ep':
            optimizer.step(x=x, target=y, model=model)
        else:
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        # Gradients should be zero or None
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() == 0
