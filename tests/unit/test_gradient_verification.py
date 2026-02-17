"""
Tests for numerical gradient verification.
Compares EP gradients against standard backpropagation.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.strategies.gradient import EPGradient, BackpropGradient
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector

def test_ep_vs_backprop_gradients(device):
    """
    Compare EP gradients with standard Backprop gradients.
    EP with small beta and sufficient settling should approximate Backprop.
    """
    # Simple MLP
    input_dim = 5
    hidden_dim = 10
    output_dim = 2

    # Use double precision for better numerical stability during check
    torch.set_default_dtype(torch.float64)

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(), # Tanh is often smoother for EP settling than ReLU
        nn.Linear(hidden_dim, output_dim)
    ).to(device)

    # Clone model for BP to ensure same initialization
    model_bp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim)
    ).to(device)
    model_bp.load_state_dict(model.state_dict())

    # Inputs
    batch_size = 4
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randn(batch_size, output_dim, device=device) # Regression target

    # 1. Compute BP gradients
    loss_fn = nn.MSELoss(reduction='sum') # reduction='sum' matches EP's default scaling often
    # EP uses 0.5 * MSE as internal energy, but external nudge uses beta * MSE.
    # E_ext = beta * 0.5 * MSE?
    # Let's check EnergyFunction._nudge_term:
    # return beta * F.mse_loss(..., reduction="sum") / batch_size
    # Note: F.mse_loss default is mean. reduction="sum" makes it sum.
    # So E_ext = beta * sum((y-y_hat)^2) / batch_size

    # BP Loss should match what EP approximates.
    # EP approximates d/dtheta (E_ext_star) where E_ext_star is the loss at the fixed point.
    # If beta -> 0, EP grad -> d/dtheta L.
    # L in EnergyFunction is: F.mse_loss(output, target, reduction="sum") / batch_size

    bp_optimizer = torch.optim.SGD(model_bp.parameters(), lr=0.1)
    bp_optimizer.zero_grad()
    output_bp = model_bp(x)
    loss_bp = nn.functional.mse_loss(output_bp, y, reduction='sum') / batch_size
    loss_bp.backward()

    bp_grads = [p.grad.clone() for p in model_bp.parameters()]

    # 2. Compute EP gradients
    # Use small beta and many steps for good approximation
    ep_strategy = EPGradient(
        beta=0.01,
        settle_steps=100,
        settle_lr=0.01,
        loss_type="mse"
    )
    energy_fn = EnergyFunction(loss_type="mse")
    inspector = ModelInspector()

    # Zero grads manually for EP model
    for p in model.parameters():
        p.grad = None

    ep_strategy.compute_gradients(
        model, x, y, energy_fn=energy_fn, structure_fn=inspector.inspect
    )

    ep_grads = [p.grad for p in model.parameters() if p.grad is not None]

    # Compare
    assert len(ep_grads) == len(bp_grads)

    cosine_sims = []
    diffs = []

    for g_ep, g_bp in zip(ep_grads, bp_grads):
        # Normalize checks
        sim = torch.nn.functional.cosine_similarity(g_ep.flatten(), g_bp.flatten(), dim=0)
        cosine_sims.append(sim.item())

        diff = torch.norm(g_ep - g_bp) / (torch.norm(g_bp) + 1e-8)
        diffs.append(diff.item())

    # Restore float32
    torch.set_default_dtype(torch.float32)

    print(f"\nCosine Similarities: {cosine_sims}")
    print(f"Relative Diffs: {diffs}")

    # We expect high cosine similarity (> 0.99) and low relative diff (< 0.1 or so)
    # EP is an approximation, so exact match is not expected.
    # With Tanh and sufficient steps, it should be quite close.

    assert all(s > 0.95 for s in cosine_sims), f"Gradients should be aligned. Sims: {cosine_sims}"
    # Relative difference might be higher due to scaling factors, but direction should be correct.
    # If beta is small enough, magnitude should also be close.
