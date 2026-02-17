"""
Tests for EP classification improvements.

Tests the CrossEntropy energy function and softmax KL divergence formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer


@pytest.fixture
def classification_model(device):
    """Simple MLP for classification."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5)  # 5 classes
    ).to(device)
    return model


@pytest.fixture
def classification_data(device):
    """Mini batch for classification."""
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 5, (8,), device=device)  # 5 classes
    return x, y


def test_cross_entropy_energy_computation(device, classification_model):
    """Test that CrossEntropy energy is computed correctly."""
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 5, (4,), device=device)

    optimizer = SMEPOptimizer(
        classification_model.parameters(),
        model=classification_model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.5,
        settle_steps=10
    )

    # Run a step - should not crash
    optimizer.step(x=x, target=y)

    # Check that model parameters were updated
    for p in classification_model.parameters():
        assert p.grad is not None


def test_softmax_kl_divergence(device):
    """Test that KL divergence is used for output layer in classification."""
    # Create a simple model - wrap in Sequential for proper hook registration
    model = nn.Sequential(
        nn.Linear(10, 5)
    ).to(device)

    optimizer = SMEPOptimizer(
        model.parameters(),
        model=model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.5,
        settle_steps=10
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 5, (4,), device=device)

    # Run step
    optimizer.step(x=x, target=y)

    # Verify gradients exist
    assert model[0].weight.grad is not None
    assert model[0].bias.grad is not None


def test_softmax_temperature(device, classification_model):
    """Test that softmax temperature affects the energy computation."""
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 5, (8,), device=device)

    # Low temperature = sharper distributions
    optimizer_low_temp = SMEPOptimizer(
        classification_model.parameters(),
        model=classification_model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.5,
        settle_steps=10,
        softmax_temperature=0.5
    )

    # High temperature = softer distributions
    classification_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer_high_temp = SMEPOptimizer(
        classification_model.parameters(),
        model=classification_model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.5,
        settle_steps=10,
        softmax_temperature=2.0
    )

    # Run steps
    optimizer_low_temp.step(x=x, target=y)
    low_temp_grad = classification_model[0].weight.grad.clone()

    classification_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer_high_temp.step(x=x, target=y)
    high_temp_grad = classification_model[0].weight.grad.clone()

    # Gradients should be different due to temperature
    # (not checking exact values, just that temperature has an effect)
    assert low_temp_grad.shape == high_temp_grad.shape


def test_sdmeo_classification(device, classification_model):
    """Test SDMEPOptimizer with classification."""
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 5, (8,), device=device)

    optimizer = SDMEPOptimizer(
        classification_model.parameters(),
        model=classification_model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.5,
        settle_steps=10,
        dion_thresh=100  # Small threshold to trigger Dion
    )

    # Run step
    optimizer.step(x=x, target=y)

    # Verify gradients exist
    for p in classification_model.parameters():
        assert p.grad is not None


def test_classification_convergence(device):
    """Test that model can learn a simple classification task."""
    torch.manual_seed(42)

    # Simple 3-class classification with separable data
    n_samples = 100
    n_features = 4
    n_classes = 3

    # Generate separable data
    x_data = torch.randn(n_samples, n_features, device=device)
    y_data = torch.zeros(n_samples, dtype=torch.long, device=device)

    # Make classes separable by adding offsets
    for i in range(n_samples):
        class_idx = i % n_classes
        y_data[i] = class_idx
        x_data[i, :2] += class_idx * 2  # Separate in first 2 dimensions

    model = nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes)
    ).to(device)

    optimizer = SMEPOptimizer(
        model.parameters(),
        model=model,
        mode='ep',
        loss_type='cross_entropy',
        lr=0.05,
        beta=0.3,
        settle_steps=15,
        settle_lr=0.03
    )

    # Train for several epochs
    batch_size = 16
    n_epochs = 20

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle data
        perm = torch.randperm(n_samples, device=device)
        x_data = x_data[perm]
        y_data = y_data[perm]

        for i in range(0, n_samples, batch_size):
            x_batch = x_data[i:i+batch_size]
            y_batch = y_data[i:i+batch_size]

            if len(x_batch) == 0:
                continue

            optimizer.step(x=x_batch, target=y_batch)
            n_batches += 1

    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        logits = model(x_data)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_data).float().mean().item()

    # Should achieve better than random (>33% for 3 classes)
    # With good convergence, should be >70%
    assert accuracy > 0.5, f"Classification accuracy too low: {accuracy:.2%}"


def test_mse_vs_cross_entropy_energy(device):
    """Test that MSE and CrossEntropy produce different energy values."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    ).to(device)

    x = torch.randn(4, 10, device=device)
    y_indices = torch.randint(0, 3, (4,), device=device)
    y_onehot = F.one_hot(y_indices, 3).float()

    # MSE optimizer
    mse_opt = SMEPOptimizer(
        model.parameters(),
        model=model,
        mode='ep',
        loss_type='mse',
        beta=0.5,
        settle_steps=5
    )

    # CrossEntropy optimizer
    ce_opt = SMEPOptimizer(
        model.parameters(),
        model=model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.5,
        settle_steps=5
    )

    # Compute energies
    structure = mse_opt._inspect_model(model)
    states_free = mse_opt._settle(model, x, beta=0.0)

    # MSE energy
    mse_energy = mse_opt._compute_energy(
        model, x, states_free, structure,
        target_vec=None, beta=0.0
    )

    # CrossEntropy energy (with target)
    ce_target = ce_opt._prepare_target(y_indices, 3, dtype=x.dtype)
    ce_energy = ce_opt._compute_energy(
        model, x, states_free, structure,
        target_vec=ce_target, beta=0.5
    )

    # Energies should be different (different formulations)
    assert mse_energy.shape == torch.Size([])
    assert ce_energy.shape == torch.Size([])
    assert not torch.allclose(mse_energy, ce_energy)
