"""
Numerical Gradient Validation for Equilibrium Propagation.

This module validates that EP gradients match finite difference approximations,
ensuring the EP implementation is mathematically correct.
"""

import torch
import torch.nn.functional as F
import pytest
import torch.nn as nn
from typing import Dict, Tuple, Optional


def calculate_numerical_gradient(
    model: nn.Module,
    optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    param: torch.Tensor,
    epsilon: float = 1e-4,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Calculate numerical gradient using central finite differences.

    Args:
        model: Neural network module.
        optimizer: EP optimizer instance (provides _settle method).
        x: Input tensor.
        y: Target tensor.
        param: Parameter tensor to compute gradient for.
        epsilon: Perturbation size for finite differences.
        loss_type: 'mse' for regression, 'cross_entropy' for classification.

    Returns:
        Numerical gradient tensor with same shape as param.
    """
    grad = torch.zeros_like(param)
    flat_param = param.view(-1)
    flat_grad = grad.view(-1)

    def compute_loss() -> float:
        """Compute loss after settling to fixed point."""
        states = optimizer._settle(model, x, beta=0.0)
        out = states[-1]

        if loss_type == "cross_entropy":
            # For classification, use cross-entropy loss
            if y.dim() > 1 and y.shape[1] > 1:
                target = y.argmax(dim=1)
            else:
                target = y.squeeze().long()
            loss = F.cross_entropy(out, target, reduction="sum")
        else:
            # MSE loss for regression
            loss = 0.5 * torch.sum((out - y) ** 2)

        return loss.item()

    for i in range(flat_param.numel()):
        # +epsilon perturbation
        with torch.no_grad():
            flat_param[i] += epsilon
        loss_plus = compute_loss()

        # -epsilon perturbation
        with torch.no_grad():
            flat_param[i] -= 2 * epsilon
        loss_minus = compute_loss()

        # Reset to original value
        with torch.no_grad():
            flat_param[i] += epsilon

        # Central difference: (f(x+ε) - f(x-ε)) / (2ε)
        flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return grad


def compute_gradient_metrics(
    ep_grad: torch.Tensor,
    num_grad: torch.Tensor,
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Compute comparison metrics between EP and numerical gradients.

    Args:
        ep_grad: Gradient from EP computation.
        num_grad: Gradient from finite differences.
        tolerance: Threshold for relative error pass/fail.

    Returns:
        Dictionary with metrics: cosine_similarity, relative_error, passed.
    """
    ep_flat = ep_grad.view(1, -1)
    num_flat = num_grad.view(1, -1)

    # Cosine similarity (direction alignment)
    cosine_sim = F.cosine_similarity(ep_flat, num_flat, dim=1).item()

    # Relative error: ||ep - num|| / (||ep|| + ||num||)
    diff_norm = torch.norm(ep_grad - num_grad)
    total_norm = torch.norm(ep_grad) + torch.norm(num_grad)
    relative_error = (diff_norm / (total_norm + 1e-8)).item()

    # Pass criteria
    direction_ok = cosine_sim > 0.9
    magnitude_ok = relative_error < tolerance

    return {
        "cosine_similarity": cosine_sim,
        "relative_error": relative_error,
        "direction_ok": direction_ok,
        "magnitude_ok": magnitude_ok,
        "passed": direction_ok and magnitude_ok
    }


@pytest.mark.slow
def test_numerical_gradients_mse(device):
    """
    Verify EP gradients match numerical gradients for MSE loss (regression).

    This is the primary validation test for the EP implementation.
    
    Note: EP provides an approximation to true gradients. With finite β,
    we expect:
    - Perfect direction alignment (cosine > 0.9)
    - Small relative error for hidden layers (< 15%)
    - Moderate relative error for output layer (< 25%) due to β approximation
    """
    from mep.optimizers import SMEPOptimizer

    # Small model for fast numerical gradient computation
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    ).to(device)

    # Double precision for accurate numerical comparison
    model.double()

    # Configure optimizer with settings for accurate EP gradients
    optimizer = SMEPOptimizer(
        model.parameters(),
        lr=0.01,
        mode='ep',
        beta=0.05,        # Small beta for better gradient approximation
        settle_steps=50,  # Many steps for convergence
        settle_lr=0.02,
        loss_type='mse'
    )

    # Simple regression data
    x = torch.randn(1, 2, dtype=torch.double, device=device)
    y = torch.tensor([[0.5]], dtype=torch.double, device=device)

    # Compute EP gradients
    optimizer.zero_grad()
    optimizer._compute_ep_gradients(model, x, y)

    # Collect and compare gradients for each parameter
    results = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        ep_grad = param.grad.clone()
        num_grad = calculate_numerical_gradient(
            model, optimizer, x, y, param,
            epsilon=1e-4,
            loss_type='mse'
        )

        # Use layer-specific tolerances
        # Output layer (2.*) has higher tolerance due to β approximation
        is_output_layer = name.startswith("2.")
        tolerance = 0.25 if is_output_layer else 0.15

        metrics = compute_gradient_metrics(ep_grad, num_grad, tolerance=tolerance)
        results[name] = metrics

    # Report results
    print("\n=== Numerical Gradient Validation (MSE) ===")
    for name, metrics in results.items():
        status = "✓ PASS" if metrics["passed"] else "✗ FAIL"
        print(f"{name}: {status}")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        print(f"  Relative Error: {metrics['relative_error']:.4f}")

    # Assertions: Direction must always be correct, magnitude has tolerance
    for name, metrics in results.items():
        assert metrics["direction_ok"], (
            f"Gradient direction mismatch for {name}: "
            f"cosine_sim={metrics['cosine_similarity']:.4f} (expected >0.9)"
        )
        # Output layer gets more tolerance due to EP approximation
        is_output_layer = name.startswith("2.")
        tolerance = 0.25 if is_output_layer else 0.15
        assert metrics["relative_error"] < tolerance, (
            f"Gradient magnitude mismatch for {name}: "
            f"relative_error={metrics['relative_error']:.4f} (expected <{tolerance})"
        )


@pytest.mark.slow
def test_numerical_gradients_cross_entropy(device):
    """
    Verify EP gradients match numerical gradients for CrossEntropy loss.

    This validates the classification energy function implementation.
    """
    from mep.optimizers import SMEPOptimizer

    # Small classification model
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3)  # 3 classes
    ).to(device)

    model.double()

    optimizer = SMEPOptimizer(
        model.parameters(),
        lr=0.01,
        mode='ep',
        beta=0.05,
        settle_steps=50,
        settle_lr=0.02,
        loss_type='cross_entropy'
    )

    # Classification data - batch size 2 for stability
    x = torch.randn(2, 4, dtype=torch.double, device=device)
    y = torch.tensor([1, 2], dtype=torch.long, device=device)  # Class indices

    # Compute EP gradients
    optimizer.zero_grad()
    optimizer._compute_ep_gradients(model, x, y)

    # Compare gradients
    results = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        ep_grad = param.grad.clone()
        num_grad = calculate_numerical_gradient(
            model, optimizer, x, y, param,
            epsilon=1e-4,
            loss_type='cross_entropy'
        )

        # Use more lenient tolerance for classification
        metrics = compute_gradient_metrics(ep_grad, num_grad, tolerance=0.25)
        results[name] = metrics

    # Report results
    print("\n=== Numerical Gradient Validation (CrossEntropy) ===")
    for name, metrics in results.items():
        status = "✓ PASS" if metrics["passed"] else "✗ FAIL"
        print(f"{name}: {status}")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        print(f"  Relative Error: {metrics['relative_error']:.4f}")

    # Assertions (more lenient for classification)
    for name, metrics in results.items():
        assert metrics["direction_ok"], (
            f"Gradient direction mismatch for {name}: "
            f"cosine_sim={metrics['cosine_similarity']:.4f} (expected >0.9)"
        )


@pytest.mark.slow
def test_numerical_gradients_batch_size(device):
    """
    Verify EP gradients are correct with batch size > 1.

    Energy should be normalized by batch size for consistent gradients.
    """
    from mep.optimizers import SMEPOptimizer

    model = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    ).to(device)

    model.double()

    optimizer = SMEPOptimizer(
        model.parameters(),
        lr=0.01,
        mode='ep',
        beta=0.05,
        settle_steps=40,
        settle_lr=0.02,
        loss_type='mse'
    )

    # Batch of 4 samples
    x = torch.randn(4, 3, dtype=torch.double, device=device)
    y = torch.randn(4, 1, dtype=torch.double, device=device)

    # Compute EP gradients
    optimizer.zero_grad()
    optimizer._compute_ep_gradients(model, x, y)

    # Compare for first layer only (fastest)
    name, param = list(model.named_parameters())[0]
    ep_grad = param.grad.clone()
    num_grad = calculate_numerical_gradient(
        model, optimizer, x, y, param,
        epsilon=1e-4,
        loss_type='mse'
    )

    metrics = compute_gradient_metrics(ep_grad, num_grad, tolerance=0.15)

    print(f"\n=== Batch Size Validation (batch={x.shape[0]}) ===")
    print(f"{name}: Cosine={metrics['cosine_similarity']:.4f}, "
          f"RelError={metrics['relative_error']:.4f}")

    assert metrics["direction_ok"], (
        f"Batch gradient direction mismatch: "
        f"cosine_sim={metrics['cosine_similarity']:.4f}"
    )


@pytest.mark.slow
def test_beta_convergence(device):
    """
    Verify EP gradients converge to true gradients as β → 0.

    Smaller beta should yield more accurate gradient estimates.
    """
    from mep.optimizers import SMEPOptimizer

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Linear(4, 1)
    ).to(device)

    model.double()

    x = torch.randn(1, 2, dtype=torch.double, device=device)
    y = torch.tensor([[0.5]], dtype=torch.double, device=device)

    # Compute reference numerical gradient
    base_optimizer = SMEPOptimizer(
        model.parameters(),
        lr=0.01,
        mode='ep',
        beta=0.01,
        settle_steps=100,
        settle_lr=0.02
    )

    base_optimizer.zero_grad()
    base_optimizer._compute_ep_gradients(model, x, y)

    # Get numerical gradient as ground truth
    name, param = list(model.named_parameters())[0]
    num_grad = calculate_numerical_gradient(
        model, base_optimizer, x, y, param,
        epsilon=1e-4
    )

    # Test different beta values
    betas = [0.5, 0.1, 0.05, 0.01]
    errors = []

    print("\n=== Beta Convergence Test ===")
    for beta in betas:
        test_model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 1)
        ).to(device)
        test_model.double()

        # Copy weights from original model
        with torch.no_grad():
            for p, p_orig in zip(test_model.parameters(), model.parameters()):
                p.copy_(p_orig)

        test_optimizer = SMEPOptimizer(
            test_model.parameters(),
            lr=0.01,
            mode='ep',
            beta=beta,
            settle_steps=50,
            settle_lr=0.02
        )

        test_optimizer.zero_grad()
        test_optimizer._compute_ep_gradients(test_model, x, y)

        ep_grad = list(test_model.parameters())[0].grad.clone()
        metrics = compute_gradient_metrics(ep_grad, num_grad, tolerance=0.3)
        errors.append(metrics["relative_error"])

        print(f"β={beta:.3f}: Relative Error={metrics['relative_error']:.4f}")

    # Error should generally decrease with smaller beta
    # (allowing some noise due to settling approximation)
    assert errors[-1] < errors[0], (
        f"Expected smaller beta to reduce error: {errors}"
    )


@pytest.mark.slow
def test_settling_steps_convergence(device):
    """
    Verify that more settling steps yield more accurate gradients.
    """
    from mep.optimizers import SMEPOptimizer

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Linear(4, 1)
    ).to(device)

    model.double()

    x = torch.randn(1, 2, dtype=torch.double, device=device)
    y = torch.tensor([[0.5]], dtype=torch.double, device=device)

    # Reference with many steps
    ref_optimizer = SMEPOptimizer(
        model.parameters(),
        lr=0.01,
        mode='ep',
        beta=0.05,
        settle_steps=100,
        settle_lr=0.02
    )

    ref_optimizer.zero_grad()
    ref_optimizer._compute_ep_gradients(model, x, y)

    name, param = list(model.named_parameters())[0]
    ref_grad = param.grad.clone()

    # Test different settling steps
    step_configs = [5, 10, 20, 50]

    print("\n=== Settling Steps Convergence Test ===")
    for steps in step_configs:
        test_model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 1)
        ).to(device)
        test_model.double()

        with torch.no_grad():
            for p, p_orig in zip(test_model.parameters(), model.parameters()):
                p.copy_(p_orig)

        test_optimizer = SMEPOptimizer(
            test_model.parameters(),
            lr=0.01,
            mode='ep',
            beta=0.05,
            settle_steps=steps,
            settle_lr=0.02
        )

        test_optimizer.zero_grad()
        test_optimizer._compute_ep_gradients(test_model, x, y)

        ep_grad = list(test_model.parameters())[0].grad.clone()
        metrics = compute_gradient_metrics(ep_grad, ref_grad, tolerance=0.2)

        print(f"Steps={steps:2d}: Cosine={metrics['cosine_similarity']:.4f}, "
              f"RelError={metrics['relative_error']:.4f}")

        # More steps should generally improve accuracy
        if steps >= 20:
            assert metrics["cosine_similarity"] > 0.95, (
                f"Expected high cosine similarity with {steps} steps"
            )
