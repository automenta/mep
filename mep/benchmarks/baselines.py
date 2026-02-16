from typing import Tuple, Dict, Any, Optional
import torch.optim as optim
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer, LocalEPMuon, NaturalEPMuon

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

    # Helper to filter kwargs for specific optimizers
    def filter_kwargs(kwargs, exclude_keys):
        return {k: v for k, v in kwargs.items() if k not in exclude_keys}

    # Keys specific to SDMEP that SMEP/LocalEP don't support
    sdmep_keys = ['rank_frac', 'dion_thresh']
    
    # Common kwargs for SMEP-based optimizers (excluding SDMEP specifics)
    smep_kwargs = filter_kwargs(kwargs, sdmep_keys)

    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay), False
        
    elif name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay), False
        
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay), False
        
    elif name == 'eqprop':
        # Vanilla EP: Use SMEP with ns_steps=0 (Effective SGD) in EP mode
        # Ensure ns_steps is not in kwargs to avoid conflict/override
        eq_kwargs = filter_kwargs(smep_kwargs, ['ns_steps'])
        return SMEPOptimizer(params, model=model, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', ns_steps=0, **eq_kwargs), True
        
    elif name == 'muon':
        # Standalone Muon: SMEP in backprop mode
        # Ensure mode is not in kwargs
        muon_kwargs = filter_kwargs(smep_kwargs, ['mode'])
        return SMEPOptimizer(params, model=model, lr=lr, momentum=momentum, wd=weight_decay, mode='backprop', **muon_kwargs), False
        
    elif name == 'smep':
        # SMEP: Muon + EP
        return SMEPOptimizer(params, model=model, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', **smep_kwargs), True
        
    elif name == 'sdmep':
        # Full SDMEP: Dion + Muon + EP
        # SDMEP accepts all kwargs
        return SDMEPOptimizer(params, model=model, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', **kwargs), True

    elif name == 'local_ep':
        # Local EP: Layer-local updates
        return LocalEPMuon(params, model=model, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', **smep_kwargs), True

    elif name == 'natural_ep':
        # Natural EP: Fisher whitening
        return NaturalEPMuon(params, model=model, lr=lr, momentum=momentum, wd=weight_decay, mode='ep', **smep_kwargs), True
        
    else:
        raise ValueError(f"Unknown optimizer: {name}")
