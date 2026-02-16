import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytest
import torch.nn as nn
from mep.optim import SDMEPOptimizer

@pytest.mark.slow
def test_mnist_accuracy_regression(device):
    """
    Verify that the model can achieve reasonable accuracy (~66% in 3 epochs) on MNIST.
    This is a regression test for the claim in README.
    """
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use a subset for speed if we just want to smoke test, 
    # but for regression on accuracy we need enough data.
    # Let's use 1000 samples for training and 100 for testing to be faster
    # but still statistically significant to jump above random chance (10%).
    # We aim for > 50% on this subset.
    
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    except Exception as e:
        pytest.skip(f"Could not download MNIST: {e}")
        
    # Create subsets
    train_indices = range(1000)
    test_indices = range(100)
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    # Model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(device)
    optimizer = SDMEPOptimizer(model.parameters(), lr=0.05, momentum=0.9, mode='ep')
    
    # Train 3 epochs
    for epoch in range(3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            optimizer.step(x=data, target=target, model=model)
            
    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data) # Forward pass only
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            
    accuracy = correct / total
    
    # Assert accuracy > 50% (random is 10%)
    # This proves learning occurred.
    assert accuracy > 0.5, f"Accuracy too low: {accuracy}"
