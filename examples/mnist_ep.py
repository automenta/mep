import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from mep import smep

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 1
LR = 0.05
BETA = 0.5  # Nudging strength
SETTLE_STEPS = 15 # Number of steps for free/nudged phase

def main():
    # 1. Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # 2. Model (Standard PyTorch Sequential)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # 3. Optimizer (SMEP in EP mode)
    # Note: We pass model=model so the optimizer can access structure
    optimizer = smep(
        model.parameters(),
        model=model,
        mode='ep',       # Enable Equilibrium Propagation
        lr=LR,
        beta=BETA,
        settle_steps=SETTLE_STEPS,
        loss_type="cross_entropy"
    )

    print("Training with Equilibrium Propagation...")
    model.train()

    for epoch in range(EPOCHS):
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # No .to(device) for simplicity, can add if needed

            # Standard forward pass (optional, but good for tracking metrics)
            # The optimizer.step() does its own internal forward/settling
            optimizer.zero_grad()

            # EP Step: Handles free phase, nudged phase, and parameter updates
            # Returns the loss computed during the free phase
            optimizer.step(x=data, target=target)

            # Calculate accuracy on the fly (using current weights)
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            if batch_idx % 100 == 0:
                acc = 100. * correct / total
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Acc: {acc:.2f}%")
                correct = 0
                total = 0

if __name__ == '__main__':
    main()
