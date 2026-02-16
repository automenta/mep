import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict

class BenchmarkVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
    def plot_training_curves(self, history: Dict[str, List[float]], title: str = "Training Curves"):
        """Plot loss and accuracy over epochs."""
        epochs = range(1, len(history['loss']) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Loss
        sns.lineplot(x=epochs, y=history['loss'], ax=ax1, label='Loss', color='tab:blue')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Accuracy
        if 'accuracy' in history:
            ax2 = ax1.twinx()
            sns.lineplot(x=epochs, y=history['accuracy'], ax=ax2, label='Accuracy', color='tab:orange')
            ax2.set_ylabel('Accuracy', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()
        
    def plot_spectral_norm(self, spectral_norms: List[float], gamma: float = None):
        """Plot spectral norm evolution."""
        if not spectral_norms:
            return
            
        epochs = range(1, len(spectral_norms) + 1)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=epochs, y=spectral_norms, label='Spectral Norm')
        
        if gamma:
            plt.axhline(y=gamma, color='r', linestyle='--', label=f'Gamma ({gamma})')
            
        plt.xlabel('Epochs')
        plt.ylabel('Spectral Norm')
        plt.title('Spectral Norm Evolution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'spectral_norm.png'))
        plt.close()
        
    def plot_optimizer_comparison(self, results: Dict[str, Dict[str, List[float]]], metric: str = 'loss'):
        """Compare different optimizers on a metric."""
        plt.figure(figsize=(12, 8))
        
        for opt_name, hist in results.items():
            if metric in hist:
                epochs = range(1, len(hist[metric]) + 1)
                sns.lineplot(x=epochs, y=hist[metric], label=opt_name)
                
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'Optimizer Comparison: {metric.capitalize()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'optimizer_comparison_{metric}.png'))
        plt.close()
