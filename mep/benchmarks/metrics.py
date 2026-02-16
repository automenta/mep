import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_start_time = None
        
    def start_epoch(self):
        self.metrics['epoch_time_accum'] = 0.0
        self.epoch_start_time = time.time()
        
    def end_epoch(self):
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.metrics['epoch_time'].append(duration)
            self.epoch_start_time = None
            
    def log_step(
        self, 
        loss: float, 
        accuracy: Optional[float] = None,
        spectral_norm: Optional[float] = None,
        energy_free: Optional[float] = None,
        energy_nudged: Optional[float] = None,
        settling_steps: Optional[int] = None,
        grad_norm: Optional[float] = None
    ):
        """Log metrics for a single step (average over batch)."""
        self.metrics['step_loss'].append(loss)
        if accuracy is not None:
            self.metrics['step_accuracy'].append(accuracy)
        if spectral_norm is not None:
            self.metrics['step_spectral_norm'].append(spectral_norm)
        if energy_free is not None:
            self.metrics['step_energy_free'].append(energy_free)
        if energy_nudged is not None:
            self.metrics['step_energy_nudged'].append(energy_nudged)
        if settling_steps is not None:
            self.metrics['step_settling_steps'].append(settling_steps)
        if grad_norm is not None:
            self.metrics['step_grad_norm'].append(grad_norm)
            
    def compute_epoch_metrics(self) -> Dict[str, float]:
        """Compute average metrics for the epoch based on logged steps."""
        epoch_metrics = {}
        
        for key in self.metrics:
            if key.startswith('step_'):
                values = self.metrics[key]
                if values:
                    epoch_key = key.replace('step_', 'epoch_')
                    epoch_metrics[epoch_key] = float(np.mean(values))
                    # Clear step metrics for next epoch? Or keep history?
                    # Usually we want to clear step metrics to avoid unbounded growth
                    # But we might want full history for detailed plots.
                    # Let's keep them but tracking index?
                    # For simplicty, let's just average and create a new list for epoch history in a separate structure if needed.
                    # Or just return expected dict.
                    self.metrics[key] = [] 
                    
        # Add epoch time
        if 'epoch_time' in self.metrics and self.metrics['epoch_time']:
            epoch_metrics['epoch_time'] = self.metrics['epoch_time'][-1]
            
        return epoch_metrics

    def check_nan(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
        return False
