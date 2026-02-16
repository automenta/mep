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

import torch.nn as nn
from mep.benchmarks.baselines import get_optimizer
from mep.benchmarks.metrics import MetricsTracker
from mep.benchmarks.visualization import BenchmarkVisualizer

try:
    import wandb
except ImportError:
    wandb = None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataloader(dataset_name, batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name.lower() == 'mnist':
        dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    elif dataset_name.lower() == 'fashion_mnist':
        dataset = datasets.FashionMNIST('./data', train=train, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        # Different normalization for CIFAR
        transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10('./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)

def main():
    parser = argparse.ArgumentParser(description="MEP Benchmark Runner")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--baselines', nargs='+', help='List of baselines to run (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory (overrides config)')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.output:
        config['output_dir'] = args.output
    if args.wandb:
        config['wandb'] = True
        
    os.makedirs(config['output_dir'], exist_ok=True)
    seed_everything(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run multiple baselines if specified
    optimizers_to_run = args.baselines if args.baselines else [config['optimizer']]
    
    results = {}
    
    for opt_name in optimizers_to_run:
        print(f"Running benchmark with optimizer: {opt_name}")
        
        # Reset seed for fair comparison
        seed_everything(config.get('seed', 42))
        
        # Init Model
        if config['model'] == 'MLP':
            dims = config['architecture']['dims']
            layers = []
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:
                    layers.append(nn.ReLU())
            model = nn.Sequential(*layers).to(device)
        else:
            raise ValueError(f"Unknown model: {config['model']}")
            
        # Init Optimizer
        # Extract optimizer specific args from config
        opt_kwargs = {}
        for k in ['momentum', 'weight_decay', 'gamma', 'rank_frac', 'error_beta']:
            if k in config:
                opt_kwargs[k] = config[k]
                
        optimizer, is_ep = get_optimizer(
            opt_name, 
            model, 
            lr=config['learning_rate'], 
            **opt_kwargs
        )
        
        # Metric Tracker
        tracker = MetricsTracker()
        
        # WandB Init
        if config.get('wandb') and wandb:
            wandb.init(
                project="mep-research", 
                config=config, 
                name=f"{config['dataset']}-{opt_name}",
                reinit=True
            )
            
        # Training Loop
        epochs = config['epochs']
        train_loader = get_dataloader(config['dataset'], config['batch_size'], train=True)
        # val_loader = get_dataloader(config['dataset'], config['batch_size'], train=False)
        
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            model.train()
            tracker.start_epoch()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{opt_name}]")
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                # Flatten for MLP
                if config['model'] == 'MLP':
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                
                if is_ep:
                    # EP Step via Optimizer
                    optimizer.step(x=data, target=target, model=model) 
                else:
                    # Backprop Step
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                
                if not is_ep:
                    optimizer.step()
                
                # Metrics (Need separate forward pass for EP usually if we want accurate loss/acc)
                # Or use what we have. compute_ep_gradients doesn't return loss.
                with torch.no_grad():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    accuracy = correct / data.size(0)
                    
                    tracker.log_step(loss.item(), accuracy)
                    pbar.set_postfix({'loss': loss.item(), 'acc': accuracy})
                    
            tracker.end_epoch()
            epoch_metrics = tracker.compute_epoch_metrics()
            
            history['loss'].append(epoch_metrics['epoch_loss'])
            history['accuracy'].append(epoch_metrics['epoch_accuracy'])
            
            if config.get('wandb') and wandb:
                wandb.log(epoch_metrics)
                
            print(f"Epoch {epoch+1}: Loss={epoch_metrics['epoch_loss']:.4f}, Acc={epoch_metrics['epoch_accuracy']:.4f}")
            
        if config.get('wandb') and wandb:
            wandb.finish()
            
        results[opt_name] = history
        
    # Visualization
    visualizer = BenchmarkVisualizer(config['output_dir'])
    visualizer.plot_optimizer_comparison(results, metric='loss')
    visualizer.plot_optimizer_comparison(results, metric='accuracy')
    
    print(f"Benchmark complete. Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
