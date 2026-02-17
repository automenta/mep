import torch
import torch.nn as nn
from mep import (
    CompositeOptimizer,
    EPGradient,
    MuonUpdate,
    SpectralConstraint,
    ErrorFeedback,
    NoFeedback,
    NoConstraint
)

# Custom Optimizer Factory
def my_custom_optimizer(model, lr=0.01):
    """
    Example of composing a custom optimizer strategy.

    This optimizer combines:
    1. Equilibrium Propagation for gradients (biologically plausible)
    2. Muon updates (orthogonalization for conditioning)
    3. Spectral Normalization (stability)
    4. Error Feedback (memory retention for continual learning)
    """

    # 1. Gradient Strategy: How to compute gradients?
    # using Equilibrium Propagation with 10 settling steps
    gradient_strategy = EPGradient(
        beta=0.5,           # Nudging strength
        settle_steps=10,    # Number of steps to reach equilibrium
        settle_lr=0.05,     # Learning rate for the settling phase
        loss_type="mse"     # Energy function loss term
    )

    # 2. Update Strategy: How to update weights?
    # using Muon (Newton-Schulz) orthogonalization
    update_strategy = MuonUpdate(
        ns_steps=5          # Newton-Schulz iterations
    )

    # 3. Constraint Strategy: How to constrain weights?
    # using Spectral Normalization to ensure contractive dynamics
    constraint_strategy = SpectralConstraint(
        gamma=0.95          # Upper bound on spectral radius
    )

    # 4. Feedback Strategy: How to handle residuals?
    # using Error Feedback to accumulate unused gradient information
    feedback_strategy = ErrorFeedback(
        beta=0.9            # Decay rate for accumulated error
    )

    return CompositeOptimizer(
        model.parameters(),
        model=model,        # Required for EP to access structure
        lr=lr,
        gradient=gradient_strategy,
        update=update_strategy,
        constraint=constraint_strategy,
        feedback=feedback_strategy
    )

def main():
    print("--- Custom Optimizer Example ---")

    # simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # Create optimizer
    optimizer = my_custom_optimizer(model)
    print(f"Optimizer created: {optimizer}")

    # Dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))

    print("Running training step...")
    optimizer.zero_grad()

    # Step with EP (requires target)
    optimizer.step(x=x, target=y)

    print(f"Step completed.")
    print("Success!")

if __name__ == "__main__":
    main()
