"""
MEP Benchmark Baselines

Provides factory function to get optimizer instances by name.
Includes both standard PyTorch optimizers (baselines) and EP-based optimizers.
"""

from typing import Tuple, Any
import torch.optim as optim
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer, LocalEPMuon, NaturalEPMuon


def get_optimizer(
    name: str,
    model: Any,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    **kwargs
) -> Tuple[Any, bool]:
    """
    Get optimizer by name.

    Args:
        name: Optimizer name (case-insensitive). Options:
            - 'sgd': SGD with momentum (baseline)
            - 'adam': Adam optimizer (baseline)
            - 'adamw': AdamW optimizer with decoupled weight decay (baseline)
            - 'muon': Standalone Muon optimizer (backprop mode)
            - 'eqprop': Vanilla Equilibrium Propagation (no spectral/Muon)
            - 'smep': Spectral Muon EP (Muon + EP gradients)
            - 'sdmep': Full SDMEP (Dion + Muon + EP)
            - 'local_ep': Local EP with layer-local updates
            - 'natural_ep': Natural EP with Fisher whitening
        model: PyTorch model to optimize
        lr: Learning rate
        momentum: Momentum for SGD-based optimizers
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Tuple of (optimizer_instance, is_ep_optimizer) where:
            - optimizer_instance: The optimizer object
            - is_ep_optimizer: True if EP gradients are required
    """
    name = name.lower()
    params = model.parameters()

    def _exclude(kwargs: dict, exclude_keys: tuple) -> dict:
        """Filter out excluded keys from kwargs."""
        return {k: v for k, v in kwargs.items() if k not in exclude_keys}

    # SDMEP-specific keys that other optimizers don't use
    _SDMEP_KEYS = ('rank_frac', 'dion_thresh')

    # Common kwargs for SMEP-based optimizers
    _SMEP_KWARGS = _exclude(kwargs, _SDMEP_KEYS)

    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay), False

    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay), False

    if name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay), False

    if name == 'eqprop':
        # Vanilla EP: SMEP with ns_steps=0 (no Newton-Schulz orthogonalization)
        eq_kwargs = _exclude(_SMEP_KWARGS, ('ns_steps',))
        return SMEPOptimizer(
            params, model=model, lr=lr, momentum=momentum, wd=weight_decay,
            mode='ep', ns_steps=0, **eq_kwargs
        ), True

    if name == 'muon':
        # Standalone Muon: SMEP in backprop mode
        muon_kwargs = _exclude(_SMEP_KWARGS, ('mode',))
        return SMEPOptimizer(
            params, model=model, lr=lr, momentum=momentum, wd=weight_decay,
            mode='backprop', **muon_kwargs
        ), False

    if name == 'smep':
        # SMEP: Muon + EP gradients
        return SMEPOptimizer(
            params, model=model, lr=lr, momentum=momentum, wd=weight_decay,
            mode='ep', **_SMEP_KWARGS
        ), True

    if name == 'sdmep':
        # Full SDMEP: Dion + Muon + EP
        return SDMEPOptimizer(
            params, model=model, lr=lr, momentum=momentum, wd=weight_decay,
            mode='ep', **kwargs
        ), True

    if name == 'local_ep':
        # Local EP: Layer-local updates only
        return LocalEPMuon(
            params, model=model, lr=lr, momentum=momentum, wd=weight_decay,
            mode='ep', **_SMEP_KWARGS
        ), True

    if name == 'natural_ep':
        # Natural EP: Fisher Information whitening
        return NaturalEPMuon(
            params, model=model, lr=lr, momentum=momentum, wd=weight_decay,
            mode='ep', **_SMEP_KWARGS
        ), True

    raise ValueError(f"Unknown optimizer: {name}. Available: sgd, adam, adamw, muon, "
                     f"eqprop, smep, sdmep, local_ep, natural_ep")
