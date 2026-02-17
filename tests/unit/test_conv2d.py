"""
Tests for Conv2d support in EP optimizers.

Verifies that convolutional layers work correctly with:
- State capture during settling
- Energy computation with 4D tensors
- EP gradient computation
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer


class SimpleCNN(nn.Module):
    """Simple CNN for testing Conv2d support."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


@pytest.fixture
def cnn_model(device):
    """Create a simple CNN model."""
    return SimpleCNN(num_classes=10).to(device)


@pytest.fixture
def cnn_data(device):
    """Create sample CNN input data (MNIST-like)."""
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    return x, y


def test_cnn_model_inspection(cnn_model):
    """Test that CNN structure is correctly inspected."""
    from mep.optimizers import SMEPOptimizer
    
    optimizer = SMEPOptimizer(cnn_model.parameters())
    structure = optimizer._inspect_model(cnn_model)
    
    # Count layer types
    layer_types = [item["type"] for item in structure]
    
    # Should have conv layers, linear layer, and other components
    assert layer_types.count("layer") >= 3, "Should have at least 3 layers (2 conv + 1 linear)"
    assert any(item["type"] == "pool" for item in structure), "Should have pooling layers"
    assert any(item["type"] == "act" for item in structure), "Should have activation layers"


def test_cnn_settling(cnn_model, cnn_data):
    """Test that settling works with CNN architectures."""
    x, y = cnn_data
    
    optimizer = SMEPOptimizer(
        cnn_model.parameters(),
        model=cnn_model,
        mode='ep',
        beta=0.3,
        settle_steps=10,
        settle_lr=0.02
    )
    
    # Run settling
    states = optimizer._settle(cnn_model, x, beta=0.0)
    
    # Should have states for each layer
    assert len(states) >= 3, f"Expected at least 3 states, got {len(states)}"
    
    # Check state shapes match layer outputs
    # conv1 output: (4, 16, 28, 28)
    # conv2 output: (4, 32, 14, 14) after pooling
    # fc output: (4, 10)
    assert states[0].ndim == 4, "Conv layer states should be 4D"
    assert states[-1].ndim == 2, "FC layer state should be 2D"


def test_cnn_ep_step(cnn_model, cnn_data):
    """Test full EP step with CNN."""
    x, y = cnn_data
    
    optimizer = SMEPOptimizer(
        cnn_model.parameters(),
        model=cnn_model,
        mode='ep',
        beta=0.3,
        settle_steps=10,
        settle_lr=0.02
    )
    
    # Store initial weights
    initial_weights = {name: p.clone() for name, p in cnn_model.named_parameters()}
    
    # Run EP step
    optimizer.step(x=x, target=y)
    
    # Check that weights changed
    weights_changed = False
    for name, p in cnn_model.named_parameters():
        if not torch.allclose(p, initial_weights[name]):
            weights_changed = True
            break
    
    assert weights_changed, "At least some weights should have changed after EP step"


def test_cnn_energy_computation(cnn_model, cnn_data):
    """Test energy computation with CNN states."""
    x, y = cnn_data
    
    optimizer = SMEPOptimizer(
        cnn_model.parameters(),
        model=cnn_model,
        mode='ep',
        loss_type='cross_entropy'
    )
    
    # Get model structure
    structure = optimizer._inspect_model(cnn_model)
    
    # Run forward to get states
    states = optimizer._settle(cnn_model, x, beta=0.0)
    
    # Compute energy
    energy = optimizer._compute_energy(
        cnn_model, x, states, structure, target_vec=None, beta=0.0
    )
    
    # Energy should be a scalar
    assert energy.ndim == 0, "Energy should be a scalar"
    assert not torch.isnan(energy), "Energy should not be NaN"
    assert not torch.isinf(energy), "Energy should not be Inf"
    # Energy should be approximately non-negative (allow small numerical error)
    assert energy.item() > -1e-6, f"Energy should be non-negative, got {energy.item()}"


def test_cnn_classification_energy(cnn_model, cnn_data):
    """Test classification energy with CNN."""
    x, y = cnn_data
    
    optimizer = SMEPOptimizer(
        cnn_model.parameters(),
        model=cnn_model,
        mode='ep',
        loss_type='cross_entropy',
        beta=0.3
    )
    
    structure = optimizer._inspect_model(cnn_model)
    
    # Free phase
    states_free = optimizer._settle(cnn_model, x, beta=0.0)
    
    # Nudged phase
    states_nudged = optimizer._settle(cnn_model, x, target=y, beta=0.3)
    
    # Compute energies
    E_free = optimizer._compute_energy(
        cnn_model, x, states_free, structure, target_vec=None, beta=0.0
    )
    
    target_vec = optimizer._prepare_target(y, states_free[-1].shape[-1], states_free[-1].dtype)
    E_nudged = optimizer._compute_energy(
        cnn_model, x, states_nudged, structure, target_vec=target_vec, beta=0.3
    )
    
    # Both energies should be valid
    assert not torch.isnan(E_free) and not torch.isinf(E_free)
    assert not torch.isnan(E_nudged) and not torch.isinf(E_nudged)
    
    # Nudged energy should typically be lower (better fit to target)
    # Note: This isn't guaranteed but is a reasonable expectation


def test_cnn_sdmeo(cnn_model, cnn_data):
    """Test SDMEPOptimizer with CNN."""
    x, y = cnn_data
    
    optimizer = SDMEPOptimizer(
        cnn_model.parameters(),
        model=cnn_model,
        mode='ep',
        beta=0.3,
        settle_steps=10,
        dion_thresh=50000  # Lower threshold to trigger Dion for larger layers
    )
    
    # Run EP step
    optimizer.step(x=x, target=y)
    
    # Check weights changed
    weights_changed = False
    for p in cnn_model.parameters():
        if p.grad is not None:
            weights_changed = True
            break
    
    assert weights_changed, "Gradients should be computed for CNN"


def test_cnn_backward_mode(cnn_model, cnn_data):
    """Test CNN with backprop mode (sanity check)."""
    x, y = cnn_data
    
    optimizer = SMEPOptimizer(
        cnn_model.parameters(),
        model=cnn_model,
        mode='backprop'
    )
    
    cnn_model.train()
    output = cnn_model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()
    
    # Check gradients were computed
    has_grad = any(p.grad is not None for p in cnn_model.parameters())
    assert has_grad, "Gradients should be computed in backprop mode"
