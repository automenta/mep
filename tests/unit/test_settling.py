"""
Tests for settling dynamics, including adaptive early stopping.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.settling import Settler
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector

def test_settle_adaptive_stopping(device):
    """Test that settler stops early when energy converges."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 2)
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, 10, device=device)

    # Set a very large max_steps but reasonable tolerance
    max_steps = 1000
    tol = 1e-3
    patience = 5

    settler = Settler(
        steps=max_steps,
        lr=0.05,
        tol=tol,
        patience=patience
    )

    # We can't easily count steps inside settle, but we can check if it returns.
    # To verify it stopped early, we can subclass or mock energy_fn to count calls.

    call_count = 0
    original_call = energy_fn.__call__

    def mocked_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(*args, **kwargs)

    energy_fn_mock = mocked_call

    # Settle
    settled = settler.settle(model, x, target=None, beta=0.0, energy_fn=energy_fn_mock, structure=structure)

    assert call_count < max_steps, f"Settler should have stopped early (steps: {call_count} vs max: {max_steps})"
    assert call_count > patience, f"Settler should run at least patience steps (steps: {call_count})"


def test_settle_divergence(device):
    """Test that settler raises RuntimeError on divergence (NaN/Inf)."""
    model = nn.Sequential(
        nn.Linear(10, 10)
    ).to(device)

    # Create an energy function that returns NaN
    def bad_energy_fn(*args, **kwargs):
        return torch.tensor(float('nan'), device=device)

    inspector = ModelInspector()
    structure = inspector.inspect(model)
    x = torch.randn(4, 10, device=device)

    settler = Settler(steps=10)

    with pytest.raises(RuntimeError, match="Energy diverged"):
        settler.settle(model, x, target=None, beta=0.0, energy_fn=bad_energy_fn, structure=structure)

def test_settle_with_graph_adaptive(device):
    """Test adaptive stopping for settle_with_graph."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 2)
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, 10, device=device)

    max_steps = 1000
    tol = 1e-3
    patience = 5

    settler = Settler(
        steps=max_steps,
        lr=0.05,
        tol=tol,
        patience=patience
    )

    call_count = 0
    original_call = energy_fn.__call__

    def mocked_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(*args, **kwargs)

    settled_states = settler.settle_with_graph(
        model, x, target=None, beta=0.0, energy_fn=mocked_call, structure=structure
    )

    assert call_count < max_steps, f"Settler should have stopped early (steps: {call_count})"

    # Check that states require grad (except they are returned detached)
    # Wait, settle_with_graph returns detached states?
    # docstring says: "Return detached states."
    # But internally it maintains graph.
    for s in settled_states:
        assert not s.requires_grad # Because returned list is detached
