"""
Benchmark regression tests for MEP optimizers.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytest
import torch.nn as nn
from mep import sdmep


@pytest.mark.slow
def test_mnist_accuracy_regression(device):
    """
    Verify that the model can achieve reasonable accuracy on MNIST subset.
    This is a regression test for basic functionality.
    """
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    except Exception as e:
        pytest.skip(f"Could not download MNIST: {e}")

    # Create subsets for speed
    train_indices = range(500)
    test_indices = range(100)
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)

    # Model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    optimizer = sdmep(
        model.parameters(),
        model=model,
        lr=0.05,
        momentum=0.9,
        mode='ep',
        settle_steps=5
    )

    # Train 1 epoch (reduced for speed)
    for epoch in range(1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            
            optimizer.step(x=x, target=y)
            optimizer.zero_grad()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    
    # Should be better than random (10%)
    assert accuracy > 0.15, f"Accuracy too low: {accuracy}"
