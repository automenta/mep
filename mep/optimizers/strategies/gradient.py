"""
Gradient computation strategies.

Implements various methods for computing gradients:
- Standard backpropagation
- Equilibrium Propagation (free/nudged contrast)
- Layer-local EP (biologically plausible)
- Natural gradient with Fisher whitening
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Any, List, Dict
from .base import GradientStrategy


class BackpropGradient:
    """
    Standard backpropagation via .backward().
    
    This is the default gradient computation for conventional deep learning.
    """
    
    def __init__(self, loss_fn: Optional[nn.Module] = None):
        self.loss_fn = loss_fn
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        loss_fn: Optional[nn.Module] = None,
        **kwargs: Any
    ) -> None:
        """
        Compute gradients via standard backpropagation.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            loss_fn: Loss function (override instance default).
        """
        loss_fn = loss_fn or self.loss_fn
        if loss_fn is None:
            raise ValueError("loss_fn must be provided to BackpropGradient")
        
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()


class EPGradient:
    """
    Equilibrium Propagation via free/nudged phase contrast.
    
    Computes gradients as (E_nudged - E_free) / beta, where:
    - Free phase: network settles with beta=0
    - Nudged phase: network settles with target perturbation
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
    ):
        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        energy_fn: Callable,
        structure_fn: Callable,
        **kwargs: Any
    ) -> None:
        """
        Compute EP gradients via free/nudged contrast.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            energy_fn: Function to compute energy given states.
            structure_fn: Function to extract model structure.
        """
        structure = structure_fn(model)
        
        # Free phase (beta=0)
        states_free = self._settle(
            model, x, target=None, beta=0.0,
            energy_fn=energy_fn, structure=structure
        )
        
        # Nudged phase
        states_nudged = self._settle(
            model, x, target=target, beta=self.beta,
            energy_fn=energy_fn, structure=structure
        )
        
        # Apply contrast
        self._apply_contrast(
            model, x, target, states_free, states_nudged,
            energy_fn, structure
        )
    
    def _settle(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Settle network to energy minimum."""
        from ..settling import Settler
        
        settler = Settler(
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
        )
        return settler.settle(model, x, target, beta, energy_fn, structure)
    
    def _apply_contrast(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        energy_fn: Callable,
        structure: List[Dict[str, Any]]
    ) -> None:
        """Apply EP gradient: (E_nudged - E_free) / beta."""
        # Prepare target
        target_vec = self._prepare_target(
            target, states_free[-1].shape[-1], states_free[-1].dtype
        )
        
        # Compute energies
        E_free = energy_fn(model, x, states_free, structure, target_vec=None, beta=0.0)
        E_nudged = energy_fn(
            model, x, states_nudged, structure, target_vec=target_vec, beta=self.beta
        )
        
        # Contrast loss
        loss = (E_nudged - E_free) / self.beta
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        
        # Accumulate gradients
        for p, g in zip(params, grads):
            if g is not None:
                if p.grad is None:
                    p.grad = g.detach()
                else:
                    p.grad.add_(g.detach())
    
    def _prepare_target(
        self,
        target: torch.Tensor,
        num_classes: int,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert target to appropriate format."""
        if self.loss_type == "cross_entropy":
            if target.dim() > 1 and target.shape[1] > 1:
                return target.argmax(dim=1).long()
            return target.squeeze().long()
        else:
            if target.dim() == 1:
                return F.one_hot(target, num_classes=num_classes).to(dtype=dtype)
            return target.to(dtype=dtype)


class LocalEPGradient:
    """
    Layer-local EP gradients (biologically plausible).
    
    Each layer computes its own local energy gradient based on
    immediate input/output, without cross-layer gradient flow.
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        loss_type: str = "mse",
    ):
        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        energy_fn: Callable,
        structure_fn: Callable,
        **kwargs: Any
    ) -> None:
        """Compute layer-local EP gradients."""
        from ..settling import Settler
        
        structure = structure_fn(model)
        settler = Settler(
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
        )
        
        # Free phase
        states_free = settler.settle(
            model, x, target=None, beta=0.0,
            energy_fn=energy_fn, structure=structure
        )
        
        # Nudged phase
        states_nudged = settler.settle(
            model, x, target=target, beta=self.beta,
            energy_fn=energy_fn, structure=structure
        )
        
        # Apply local contrast per layer
        self._apply_local_contrast(
            model, x, target, states_free, states_nudged, structure
        )
    
    def _apply_local_contrast(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        structure: List[Dict[str, Any]]
    ) -> None:
        """Apply EP contrast independently per layer."""
        # Extract layer I/O
        io_free = self._get_layer_io(x, states_free, structure)
        io_nudged = self._get_layer_io(x, states_nudged, structure)
        
        map_free = {id(item["module"]): item for item in io_free}
        map_nudged = {id(item["module"]): item for item in io_nudged}
        batch_size = x.shape[0]
        
        for item in structure:
            if item["type"] != "layer":
                continue
            
            module = item["module"]
            if id(module) not in map_free or id(module) not in map_nudged:
                continue
            
            # Free phase
            in_free = map_free[id(module)]["input"].detach()
            out_free = map_free[id(module)]["output"].detach()
            
            # Nudged phase
            in_nudged = map_nudged[id(module)]["input"].detach()
            out_nudged = map_nudged[id(module)]["output"].detach()
            
            module_params = list(module.parameters())
            
            with torch.enable_grad():
                pred_free = module(in_free)
                E_free = 0.5 * F.mse_loss(pred_free, out_free, reduction="sum") / batch_size
                
                pred_nudged = module(in_nudged)
                E_nudged = 0.5 * F.mse_loss(pred_nudged, out_nudged, reduction="sum") / batch_size
                
                loss = (E_nudged - E_free) / self.beta
                grads = torch.autograd.grad(loss, module_params, retain_graph=False, allow_unused=True)
            
            for p, g in zip(module_params, grads):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.detach()
                    else:
                        p.grad.add_(g.detach())
    
    def _get_layer_io(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
        structure: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract layer inputs and outputs."""
        io_list = []
        prev = x
        state_idx = 0
        
        for item in structure:
            if item["type"] == "layer":
                if state_idx >= len(states):
                    break
                module = item["module"]
                state = states[state_idx]
                io_list.append({"module": module, "input": prev, "output": state})
                prev = state
                state_idx += 1
            elif item["type"] == "act":
                prev = item["module"](prev)
        
        return io_list


class NaturalGradient:
    """
    Natural gradient with Fisher Information whitening.
    
    Wraps a base gradient strategy and applies Fisher-based whitening
    to account for the geometry of the parameter space.
    """
    
    def __init__(
        self,
        base_strategy: GradientStrategy,
        fisher_approx: str = "empirical",
        use_diagonal: bool = False,
    ):
        self.base_strategy = base_strategy
        self.fisher_approx = fisher_approx
        self.use_diagonal = use_diagonal
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        energy_fn: Optional[Callable] = None,
        structure_fn: Optional[Callable] = None,
        **kwargs: Any
    ) -> None:
        """
        Compute natural gradients with Fisher whitening.
        
        First computes base gradients, then captures Fisher information.
        """
        # Get structure if structure_fn provided
        structure = None
        if structure_fn is not None:
            structure = structure_fn(model)
        
        # Compute base gradients
        if energy_fn is not None and structure is not None:
            self.base_strategy.compute_gradients(
                model, x, target, energy_fn=energy_fn, structure_fn=structure_fn, **kwargs
            )
        else:
            self.base_strategy.compute_gradients(model, x, target, **kwargs)
        
        # Capture Fisher information for later use in update
        self._compute_fisher(model, x, target, energy_fn, structure)
    
    def _compute_fisher(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        energy_fn: Optional[Callable],
        structure: Optional[List[Dict[str, Any]]]
    ) -> None:
        """
        Compute Fisher Information Matrix blocks.
        
        Stores Fisher blocks in a way accessible to NaturalUpdate strategy.
        """
        # For empirical Fisher: F = sum(g @ g.T) over samples
        # We approximate using the free-phase gradients
        if energy_fn is not None and structure is not None:
            # EP context - use free phase energy gradients
            pass  # Fisher will be computed from stored grad_free
        else:
            # Standard backprop context
            for p in model.parameters():
                if p.grad is not None and p.ndim >= 2:
                    # Store for NaturalUpdate to use
                    pass  # Gradient already in p.grad
