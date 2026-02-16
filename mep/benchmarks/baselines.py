from typing import Tuple, Dict, Any, Optional
import torch.optim as optim
from mep.optim import SMEPOptimizer, SDMEPOptimizer

def get_optimizer(
    name: str, 
    model: Any, 
    lr: float = 0.01, 
    momentum: float = 0.9, 
    weight_decay: float = 0.0005,
    **kwargs
) -> Tuple[optim.Optimizer, bool]:
    """
    Get optimizer by name.
    Returns: (optimizer_instance, is_ep_optimizer)
    
    is_ep_optimizer: True if it requires EP gradients (compute_ep_gradients),
                     False if it uses standard Backprop (.backward())
    """
    name = name.lower()
    params = model.parameters()
    
    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay), False
        
    elif name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay), False
        
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay), False
        
    elif name == 'eqprop':
        # Vanilla EP: Use SMEP with ns_steps=0 (Effective SGD) in EP mode
        return SMEPOptimizer(params, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', ns_steps=0), True
        
    elif name == 'muon':
        # Standalone Muon: SMEP in backprop mode
        return SMEPOptimizer(params, lr=lr, momentum=momentum, wd=weight_decay, mode='backprop'), False
        
    elif name == 'smep':
        # SMEP: Muon + EP
        return SMEPOptimizer(params, lr=lr, momentum=momentum, wd=weight_decay, mode='ep'), True
        
    elif name == 'sdmep':
        # Full SDMEP: Dion + Muon + EP
        return SDMEPOptimizer(params, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', **kwargs), True
        
    else:
        raise ValueError(f"Unknown optimizer: {name}")
