"""
MEP Benchmark Runner

Comprehensive benchmarking framework for comparing optimizers with:
- Multiple repeats for statistical robustness
- Time-based epoch calculation
- Statistical significance testing
- Detailed metrics collection
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import random
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats

import torch.nn as nn
from mep.benchmarks.baselines import get_optimizer
from mep.benchmarks.metrics import MetricsTracker
from mep.benchmarks.visualization import BenchmarkVisualizer

try:
    import wandb
except ImportError:
    wandb = None


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    train: bool = True,
    subset_size: Optional[int] = None
) -> DataLoader:
    """
    Create data loader for specified dataset.

    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        batch_size: Batch size for data loader
        train: Whether to load training or test set
        subset_size: Optional limit on dataset size (for quick tests)
    """
    if dataset_name.lower() in ['mnist', 'fashion_mnist']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_cls = {
        'mnist': datasets.MNIST,
        'fashion_mnist': datasets.FashionMNIST,
        'cifar10': datasets.CIFAR10
    }[dataset_name.lower()]

    dataset = dataset_cls(
        './data',
        train=train,
        download=True,
        transform=transform
    )

    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )


def create_model(model_type: str, architecture: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Create neural network model.

    Args:
        model_type: Type of model ('MLP', 'CNN')
        architecture: Architecture configuration dict
        device: Device to create model on

    Returns:
        PyTorch model
    """
    if model_type == 'MLP':
        dims = architecture['dims']
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers).to(device)

    elif model_type == 'CNN':
        # Simple CNN for CIFAR-10
        layers = [
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, architecture.get('num_classes', 10))
        ]
        return nn.Sequential(*layers).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def calculate_epochs_for_time(
    model: nn.Module,
    optimizer: Any,
    train_loader: DataLoader,
    device: torch.device,
    is_ep: bool,
    target_time_seconds: float = 60.0
) -> int:
    """
    Estimate number of epochs that would take approximately target_time.

    Runs a single epoch as a pilot to estimate time per epoch.
    """
    model.train()
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if hasattr(data, 'view'):
            data = data.view(data.size(0), -1)

        optimizer.zero_grad()

        if is_ep:
            optimizer.step(x=data, target=target, model=model)
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    pilot_time = time.time() - start_time
    estimated_epochs = max(1, int(target_time_seconds / pilot_time))

    print(f"Pilot epoch time: {pilot_time:.2f}s → Estimated epochs for ~{target_time_seconds}s: {estimated_epochs}")

    return estimated_epochs


def run_single_trial(
    optimizer_name: str,
    model: nn.Module,
    optimizer: Any,
    is_ep: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single training trial.

    Returns:
        Dictionary with training history and metrics
    """
    history = {
        'epoch_loss': [],
        'epoch_accuracy': [],
        'epoch_time': [],
        'time_per_step': [],
        'test_accuracy': []
    }

    tracker = MetricsTracker()

    for epoch in range(epochs):
        model.train()
        tracker.start_epoch()
        epoch_start = time.time()
        step_times = []

        for batch_idx, (data, target) in enumerate(train_loader):
            step_start = time.time()
            data, target = data.to(device), target.to(device)

            if hasattr(data, 'view'):
                data = data.view(data.size(0), -1)

            optimizer.zero_grad()

            if is_ep:
                optimizer.step(x=data, target=target, model=model)
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            step_times.append(time.time() - step_start)

            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / data.size(0)
                tracker.log_step(loss.item(), accuracy)

        tracker.end_epoch()
        epoch_metrics = tracker.compute_epoch_metrics()

        history['epoch_loss'].append(epoch_metrics['epoch_loss'])
        history['epoch_accuracy'].append(epoch_metrics['epoch_accuracy'])
        history['epoch_time'].append(time.time() - epoch_start)
        history['time_per_step'].append(float(np.mean(step_times)))

        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device)
        history['test_accuracy'].append(test_acc)

    return history


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if hasattr(data, 'view'):
                data = data.view(data.size(0), -1)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return correct / total


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }


def statistical_test(
    baseline_values: List[float],
    treatment_values: List[float],
    metric_name: str = 'accuracy'
) -> Dict[str, Any]:
    """
    Perform statistical significance test (Welch's t-test).

    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    if len(baseline_values) < 2 or len(treatment_values) < 2:
        return {
            't_statistic': None,
            'p_value': None,
            'significant': False,
            'note': 'Insufficient samples for statistical test'
        }

    t_stat, p_value = stats.ttest_ind(
        baseline_values,
        treatment_values,
        equal_var=False  # Welch's t-test
    )

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': (
            f"{'Significantly better' if t_stat < 0 else 'Significantly worse'} "
            f"than baseline (p={p_value:.4f})"
            if p_value < 0.05
            else "No significant difference from baseline"
        )
    }


def run_benchmarks(
    config: Dict[str, Any],
    optimizers_to_run: List[str],
    repeats: int = 2,
    target_time_per_trial: float = 60.0,
    output_dir: Optional[str] = None,
    wandb_enabled: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive benchmarks with repeats.

    Args:
        config: Configuration dictionary
        optimizers_to_run: List of optimizer names
        repeats: Number of repeats per optimizer
        target_time_per_trial: Target time in seconds per trial
        output_dir: Output directory for results
        wandb_enabled: Whether to enable WandB logging

    Returns:
        Complete results dictionary with statistics
    """
    # Setup
    if output_dir is None:
        output_dir = config.get('output_dir', 'benchmarks/results')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {repeats} repeats per optimizer, ~{target_time_per_trial}s each")
    print(f"Optimizers: {', '.join(optimizers_to_run)}")
    print("=" * 60)

    # Create data loaders
    train_loader = get_dataloader(
        config['dataset'],
        config['batch_size'],
        train=True,
        subset_size=config.get('subset_size')
    )
    test_loader = get_dataloader(
        config['dataset'],
        config['batch_size'],
        train=False
    )

    # Results storage
    all_results = {
        'config': config,
        'optimizers': {},
        'summary': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'repeats': repeats,
            'target_time_per_trial': target_time_per_trial
        }
    }

    # Run benchmarks for each optimizer
    for opt_name in optimizers_to_run:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {opt_name.upper()}")
        print(f"{'='*60}")

        repeat_results = []

        for repeat in range(repeats):
            print(f"\n--- Repeat {repeat + 1}/{repeats} ---")

            # Reset seed for each repeat
            seed = config.get('seed', 42) + repeat
            seed_everything(seed)

            # Create fresh model
            model = create_model(
                config['model'],
                config['architecture'],
                device
            )

            # Create optimizer
            opt_kwargs = {
                k: v for k, v in config.items()
                if k in ['lr', 'momentum', 'weight_decay', 'gamma', 'rank_frac',
                         'error_beta', 'beta', 'settle_steps', 'ns_steps',
                         'dion_thresh', 'use_spectral_constraint']
            }

            optimizer, is_ep = get_optimizer(
                opt_name,
                model,
                **opt_kwargs
            )

            # Calculate epochs for target time (on first repeat)
            if repeat == 0:
                epochs = calculate_epochs_for_time(
                    model, optimizer, train_loader, device, is_ep,
                    target_time_per_trial
                )
            else:
                # Use same epochs for subsequent repeats
                pass

            # Run trial
            trial_history = run_single_trial(
                opt_name, model, optimizer, is_ep,
                train_loader, test_loader, device, epochs, config
            )

            # Store final test accuracy for this repeat
            final_acc = trial_history['test_accuracy'][-1] if trial_history['test_accuracy'] else 0
            repeat_results.append({
                'repeat': repeat + 1,
                'seed': seed,
                'epochs': epochs,
                'final_test_accuracy': final_acc,
                'history': trial_history
            })

            print(f"  Final test accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

        # Aggregate results for this optimizer
        final_accuracies = [r['final_test_accuracy'] for r in repeat_results]
        acc_stats = compute_statistics(final_accuracies)

        all_results['optimizers'][opt_name] = {
            'repeats': repeat_results,
            'statistics': {
                'final_test_accuracy': acc_stats
            }
        }

        print(f"\n{opt_name} Summary:")
        print(f"  Final Test Accuracy: {acc_stats['mean']*100:.2f}% ± {acc_stats['std']*100:.2f}%")

    # Compute summary with statistical comparisons
    baseline_name = 'sgd'  # Use SGD as baseline
    if baseline_name in all_results['optimizers']:
        baseline_accs = [
            r['final_test_accuracy']
            for r in all_results['optimizers'][baseline_name]['repeats']
        ]

        for opt_name in optimizers_to_run:
            if opt_name == baseline_name:
                continue

            treatment_accs = [
                r['final_test_accuracy']
                for r in all_results['optimizers'][opt_name]['repeats']
            ]

            stat_result = statistical_test(baseline_accs, treatment_accs)
            all_results['optimizers'][opt_name]['statistics']['comparison_to_baseline'] = stat_result

    # Save results
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        # Convert any non-serializable objects
        json_results = json.loads(json.dumps(all_results, default=str))
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate visualizations
    visualizer = BenchmarkVisualizer(output_dir)
    visualizer.plot_optimizer_comparison_with_stats(all_results, metric='final_test_accuracy')
    visualizer.plot_training_curves_all(all_results)

    print(f"\nVisualizations saved to: {output_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MEP Comprehensive Benchmark Runner")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--baselines', nargs='+', help='List of optimizers to run (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory (overrides config)')
    parser.add_argument('--repeats', type=int, default=2, help='Number of repeats per optimizer')
    parser.add_argument('--time-per-trial', type=float, default=60.0,
                        help='Target time in seconds per trial (default: 60s)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Fixed number of epochs (overrides time-based calculation)')

    args = parser.parse_args()
    config = load_config(args.config)

    if args.output:
        config['output_dir'] = args.output
    if args.wandb:
        config['wandb'] = True

    # Determine optimizers to run
    if args.baselines:
        optimizers_to_run = args.baselines
    elif 'baselines' in config:
        optimizers_to_run = config['baselines']
    else:
        optimizers_to_run = [config.get('optimizer', 'sgd')]

    # Run benchmarks
    results = run_benchmarks(
        config=config,
        optimizers_to_run=optimizers_to_run,
        repeats=args.repeats,
        target_time_per_trial=args.time_per_trial,
        output_dir=config.get('output_dir'),
        wandb_enabled=config.get('wandb', False)
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Optimizer':<15} {'Mean Acc (%)':<15} {'Std (%)':<12} {'vs SGD':<25}")
    print("-" * 70)

    for opt_name, opt_data in results['optimizers'].items():
        stats = opt_data['statistics']['final_test_accuracy']
        mean_acc = stats['mean'] * 100
        std_acc = stats['std'] * 100

        comparison = ""
        if 'comparison_to_baseline' in opt_data['statistics']:
            comp = opt_data['statistics']['comparison_to_baseline']
            if comp['significant']:
                comparison = comp['interpretation'][:24]
            else:
                comparison = "No sig. difference"

        print(f"{opt_name:<15} {mean_acc:<15.2f} {std_acc:<12.2f} {comparison:<25}")

    print("=" * 70)
    print(f"\nFull results saved to: {config.get('output_dir', 'benchmarks/results')}/benchmark_results.json")


if __name__ == "__main__":
    main()
