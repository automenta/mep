"""
Energy function computation for Equilibrium Propagation.

This module defines the energy function used in EP:
    E = E_internal + E_external

where:
    E_internal = 0.5 * Σ ||s_i - f_i(s_{i-1})||²  (state consistency)
    E_external = β * L(s_last, y)                 (task loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


class EnergyFunction:
    """
    Computes the EP energy function.
    
    The energy measures how well the network states satisfy:
    1. Internal consistency (each layer matches its prediction)
    2. External constraint (output matches target, when nudged)
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
    ):
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
    
    def __call__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        states: List[torch.Tensor],
        structure: List[Dict[str, Any]],
        target_vec: Optional[torch.Tensor] = None,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute total energy: E = E_int + E_ext.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            states: List of layer states.
            structure: Model structure.
            target_vec: Target for nudge term (None for free phase).
            beta: Nudging strength.
        
        Returns:
            Scalar energy tensor.
        """
        batch_size = x.shape[0]
        if batch_size == 0:
            raise ValueError(f"Batch size cannot be zero, got input shape {x.shape}")
        
        use_classification = self.loss_type == "cross_entropy"
        
        E = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        prev = x
        state_idx = 0
        
        # Find layer modules for identifying last layer
        layer_modules = [item["module"] for item in structure if item["type"] == "layer"]
        last_layer_idx = len(layer_modules) - 1
        
        for item in structure:
            item_type = item["type"]
            module = item["module"]
            
            if item_type == "layer":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                is_last_layer = (state_idx == last_layer_idx)
                h = module(prev)
                
                if use_classification and is_last_layer:
                    # KL divergence for classification output
                    E = E + self._kl_energy(state, h, batch_size)
                else:
                    # MSE for hidden layers and regression
                    E = E + 0.5 * F.mse_loss(h, state, reduction="sum") / batch_size
                
                prev = state
                state_idx += 1
            
            elif item_type == "norm":
                prev = module(prev)
            
            elif item_type == "pool":
                prev = module(prev)
            
            elif item_type == "attention":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                
                if isinstance(module, nn.MultiheadAttention):
                    try:
                        h = module(prev, prev, prev, need_weights=False)[0]
                    except (RuntimeError, AssertionError):
                        prev = state
                        state_idx += 1
                        continue
                else:
                    h = module(prev)
                
                E = E + 0.5 * F.mse_loss(h, state, reduction="sum") / batch_size
                prev = h
                state_idx += 1
            
            elif item_type == "act":
                prev = module(prev)
        
        # Nudge term
        if target_vec is not None and beta > 0:
            E = E + self._nudge_term(prev, target_vec, beta, batch_size)
        
        # Stability check
        if torch.isnan(E) or torch.isinf(E):
            raise RuntimeError(
                f"Energy computation produced NaN/Inf. "
                f"Input: {x.shape}, States: {len(states)}, "
                f"Target: {target_vec.shape if target_vec is not None else None}"
            )
        
        return E
    
    def _kl_energy(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute KL divergence energy for classification output.
        
        Uses softmax-aware formulation:
            E = D_KL(softmax(state) || softmax(prediction))
        """
        eps = 1e-8
        state_softmax = F.softmax(state / self.softmax_temperature, dim=-1)
        h_softmax = F.softmax(prediction / self.softmax_temperature, dim=-1)
        
        kl_div = F.kl_div(
            torch.log(state_softmax + eps), h_softmax, reduction="sum"
        )
        return kl_div / batch_size
    
    def _nudge_term(
        self,
        output: torch.Tensor,
        target_vec: torch.Tensor,
        beta: float,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute external nudge term.
        
        For classification: CrossEntropy with label smoothing
        For regression: MSE
        """
        if self.loss_type == "cross_entropy":
            # target_vec contains class indices
            return beta * F.cross_entropy(
                output, target_vec, reduction="sum", label_smoothing=0.1
            ) / batch_size
        else:
            # MSE for regression
            return beta * F.mse_loss(output, target_vec, reduction="sum") / batch_size
