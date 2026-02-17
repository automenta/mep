"""
Settling dynamics for Equilibrium Propagation.

This module handles the iterative settling of network activations
to minimize the energy function during free and nudged phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable


class Settler:
    """
    Settles network activations to minimize energy.
    
    Uses gradient-based optimization to find fixed points of the
    energy function during EP free and nudged phases.
    """
    
    MOMENTUM = 0.5
    
    def __init__(
        self,
        steps: int = 20,
        lr: float = 0.05,
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
    ):
        self.steps = steps
        self.lr = lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
    
    def settle(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """
        Settle network activations to energy minimum.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor (None for free phase).
            beta: Nudging strength.
            energy_fn: Function to compute energy.
            structure: Model structure from inspector.
        
        Returns:
            List of settled state tensors for each layer.
        
        Raises:
            ValueError: If input is invalid.
            RuntimeError: If settling diverges.
        """
        if x.numel() == 0:
            raise ValueError(f"Input tensor cannot be empty, got shape {x.shape}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        
        # Capture initial states
        states = self._capture_states(model, x, structure)
        
        if not states:
            layer_count = sum(1 for item in structure if item["type"] == "layer")
            raise RuntimeError(
                f"No activations captured. Expected {layer_count} layer(s).\n"
                f"Model: {type(model).__name__}, Structure: {len(structure)} items"
            )
        
        # Prepare target
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, states[-1].shape[-1], states[-1].dtype)
        
        # Momentum buffers
        momentum_buffers = [torch.zeros_like(s) for s in states]
        
        # Settling loop
        prev_energy: Optional[float] = None
        
        for step in range(self.steps):
            with torch.enable_grad():
                E = energy_fn(model, x, states, structure, target_vec, beta)
                
                # Check for divergence
                if torch.isnan(E) or torch.isinf(E):
                    raise RuntimeError(
                        f"Energy diverged at step {step}: E={E.item()}. "
                        f"Try reducing settle_lr, beta, or learning rate."
                    )
                
                grads = torch.autograd.grad(E, states, retain_graph=False, allow_unused=True)
            
            # SGD step with momentum
            with torch.no_grad():
                for i, (state, g) in enumerate(zip(states, grads)):
                    if g is None:
                        continue
                    
                    buf = momentum_buffers[i]
                    buf.mul_(self.MOMENTUM).add_(g)
                    state.sub_(buf, alpha=self.lr)
        
        return [s.detach() for s in states]
    
    def settle_with_graph(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """
        Settle network keeping computation graph intact for gradient flow.
        
        This variant doesn't detach states, allowing gradients to flow
        back to model parameters through the settling trajectory.
        """
        if x.numel() == 0:
            raise ValueError(f"Input tensor cannot be empty, got shape {x.shape}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        
        # Capture initial states (with grad) - use fresh tensors that track graph
        states = self._capture_states_fresh(model, x, structure)
        
        if not states:
            layer_count = sum(1 for item in structure if item["type"] == "layer")
            raise RuntimeError(
                f"No activations captured. Expected {layer_count} layer(s)."
            )
        
        # Prepare target
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, states[-1].shape[-1], states[-1].dtype)
        
        # Momentum buffers
        momentum_buffers = [torch.zeros_like(s) for s in states]
        
        # Settling loop - states are optimized but remain connected to graph
        # through the energy computation which uses module(prev)
        for step in range(self.steps):
            # For settling, we need states that can receive gradients
            # Create working copies that require grad
            working_states = [s.detach().requires_grad_(True) for s in states]
            
            E = energy_fn(model, x, working_states, structure, target_vec, beta)
            
            if torch.isnan(E) or torch.isinf(E):
                raise RuntimeError(f"Energy diverged at step {step}: E={E.item()}")
            
            grads = torch.autograd.grad(E, working_states, retain_graph=False, allow_unused=True)
            
            # Update working states
            for i, (state, g) in enumerate(zip(working_states, grads)):
                if g is None:
                    continue
                buf = momentum_buffers[i]
                buf.mul_(self.MOMENTUM).add_(g)
                state = state - buf * self.lr
                working_states[i] = state
            
            # Copy back to states (detach to break graph through time)
            with torch.no_grad():
                for i, s in enumerate(working_states):
                    states[i] = s.detach().requires_grad_(False)
        
        # For EP gradient computation, we need states that when used in energy
        # computation will connect to parameters. Return detached states.
        return [s.detach() for s in states]
    
    def _capture_states_fresh(
        self,
        model: nn.Module,
        x: torch.Tensor,
        structure: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Capture states as fresh tensors that will be optimized during settling."""
        states: List[torch.Tensor] = []
        handles: List[Any] = []
        
        def capture_hook(module: nn.Module, inp: Any, output: Any) -> None:
            # Capture as detached tensor - will be used as initial point for settling
            if isinstance(output, tuple):
                states.append(output[0].detach().clone())
            else:
                states.append(output.detach().clone())
        
        for item in structure:
            if item["type"] in ("layer", "attention"):
                handles.append(item["module"].register_forward_hook(capture_hook))
        
        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()
        
        return states

    def _capture_states(
        self,
        model: nn.Module,
        x: torch.Tensor,
        structure: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Capture initial layer states via forward pass."""
        states: List[torch.Tensor] = []
        handles: List[Any] = []
        
        def capture_hook(module: nn.Module, inp: Any, output: Any) -> None:
            if isinstance(output, tuple):
                states.append(output[0].detach().clone().requires_grad_(True))
            else:
                states.append(output.detach().clone().requires_grad_(True))
        
        for item in structure:
            if item["type"] in ("layer", "attention"):
                handles.append(item["module"].register_forward_hook(capture_hook))
        
        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()
        
        return states
    
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
