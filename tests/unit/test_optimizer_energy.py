"""
Tests for energy computation and settling dynamics.
"""

import torch
import torch.nn as nn
import pytest
from mep import smep
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector
from mep.optimizers.settling import Settler


def test_settle_energy_reduction(device):
    """Test that settling reduces energy compared to initial state."""
    # Simple MLP
    dims = [10, 20, 5]
    model = nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        nn.ReLU(),
        nn.Linear(dims[1], dims[2])
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    batch_size = 4
    x = torch.randn(batch_size, dims[0]).to(device)
    
    # Capture initial states
    initial_states = []
    handles = []
    
    def capture_hook(module, inp, output):
        initial_states.append(output.detach().clone())
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            handles.append(m.register_forward_hook(capture_hook))
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()
    
    # Compute initial energy
    E_initial = energy_fn(model, x, initial_states, structure, target_vec=None, beta=0.0)
    
    # Settle
    settler = Settler(steps=10, lr=0.05)
    settled_states = settler.settle(model, x, target=None, beta=0.0, energy_fn=energy_fn, structure=structure)
    
    # Compute settled energy
    E_settled = energy_fn(model, x, settled_states, structure, target_vec=None, beta=0.0)
    
    # Energy should decrease (or stay similar)
    assert E_settled.item() <= E_initial.item() + 0.1, "Settling should reduce energy"


def test_energy_with_nudge(device):
    """Test energy computation with nudging."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    ).to(device)
    
    energy_fn = EnergyFunction(loss_type='mse')
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)
    target_vec = nn.functional.one_hot(y, num_classes=2).float()
    
    # Get states
    states = []
    handles = []
    
    def capture_hook(module, inp, output):
        states.append(output.detach().clone())
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            handles.append(m.register_forward_hook(capture_hook))
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()
    
    # Energy without nudge
    E_free = energy_fn(model, x, states, structure, target_vec=None, beta=0.0)
    
    # Energy with nudge
    E_nudged = energy_fn(model, x, states, structure, target_vec=target_vec, beta=0.5)
    
    # Nudged energy should be different
    assert E_nudged.item() != E_free.item(), "Nudge should change energy"


def test_classification_energy(device):
    """Test energy computation for classification."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    ).to(device)
    
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 3, (4,), device=device)
    
    # Get states
    states = []
    handles = []
    
    def capture_hook(module, inp, output):
        states.append(output.detach().clone())
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            handles.append(m.register_forward_hook(capture_hook))
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()
    
    # Energy with classification target
    E = energy_fn(model, x, states, structure, target_vec=y, beta=0.5)
    
    # Should be finite
    assert torch.isfinite(E), f"Energy should be finite, got {E}"
