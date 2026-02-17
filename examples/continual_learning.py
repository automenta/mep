import torch
import torch.nn as nn
from torchvision import datasets, transforms
from mep import smep
import copy

# Hyperparameters
BATCH_SIZE = 64
LR = 0.02
EPOCHS_PER_TASK = 1 # Keep it fast for example
BETA = 0.5

def get_task_data(task_id, root='./data'):
    """
    Returns train/test loaders for Split MNIST.
    Task 1: digits 0-4
    Task 2: digits 5-9
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root, train=False, download=True, transform=transform)

    if task_id == 1:
        valid_labels = [0, 1, 2, 3, 4]
    else:
        valid_labels = [5, 6, 7, 8, 9]

    # Filter function
    def filter_dataset(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label in valid_labels]
        return torch.utils.data.Subset(dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        filter_dataset(train_set), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        filter_dataset(test_set), batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

def train_task(model, optimizer, loader, task_name):
    print(f"  Training on {task_name}...")
    model.train()
    for epoch in range(EPOCHS_PER_TASK):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            optimizer.step(x=data, target=target)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

def run_experiment(use_feedback):
    print(f"\n=== Running Experiment: Error Feedback = {use_feedback} ===")

    # Setup
    model = create_model()

    # Configure optimizer
    # For continual learning, we usually want high error retention (beta close to 1)
    optimizer = smep(
        model.parameters(),
        model=model,
        mode='ep',
        lr=LR,
        beta=BETA,
        use_error_feedback=use_feedback,
        error_beta=0.9 if use_feedback else 0.0
    )

    # Load Data
    t1_train, t1_test = get_task_data(1)
    t2_train, t2_test = get_task_data(2)

    # Train Task 1
    train_task(model, optimizer, t1_train, "Task 1 (0-4)")
    acc_t1_after_t1 = evaluate(model, t1_test)
    print(f"  Accuracy on Task 1 after Task 1: {acc_t1_after_t1:.2f}%")

    # Train Task 2
    train_task(model, optimizer, t2_train, "Task 2 (5-9)")
    acc_t2_after_t2 = evaluate(model, t2_test)
    print(f"  Accuracy on Task 2 after Task 2: {acc_t2_after_t2:.2f}%")

    # Check forgetting
    acc_t1_after_t2 = evaluate(model, t1_test)
    print(f"  Accuracy on Task 1 after Task 2: {acc_t1_after_t2:.2f}%")

    forgetting = acc_t1_after_t1 - acc_t1_after_t2
    print(f"  Forgetting: {forgetting:.2f}%")

    return forgetting

def main():
    print("Running Continual Learning Benchmark (Split MNIST)")
    print("Comparing Standard EP vs EP with Error Feedback")

    f_baseline = run_experiment(use_feedback=False)
    f_feedback = run_experiment(use_feedback=True)

    print("\n=== Summary ===")
    print(f"Baseline Forgetting: {f_baseline:.2f}%")
    print(f"Feedback Forgetting: {f_feedback:.2f}%")

    if f_feedback < f_baseline:
        print("SUCCESS: Error feedback reduced catastrophic forgetting.")
    else:
        print("NOTE: Error feedback did not reduce forgetting in this short run.")

if __name__ == '__main__':
    main()
