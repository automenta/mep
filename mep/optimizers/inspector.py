"""
Model structure inspection utilities.

Extracts layer structure from PyTorch models for EP state tracking.
"""

import torch.nn as nn
from typing import List, Dict, Any


class ModelInspector:
    """
    Extracts sequence of layers and activations from a model.
    
    Caches structure to avoid repeated introspection.
    """
    
    def __init__(self):
        self._cache: Dict[int, List[Dict[str, Any]]] = {}
    
    def inspect(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Extract model structure.
        
        Args:
            model: Neural network to inspect.
        
        Returns:
            List of structure items with 'type' and 'module' keys.
        """
        model_id = id(model)
        if model_id in self._cache:
            return self._cache[model_id]
        
        structure: List[Dict[str, Any]] = []
        
        for m in model.modules():
            # Convolutional and linear layers
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                structure.append({"type": "layer", "module": m})
            
            # Transformer attention
            elif isinstance(m, nn.MultiheadAttention):
                structure.append({"type": "attention", "module": m})
            
            # Normalization layers
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
            )):
                structure.append({"type": "norm", "module": m})
            
            # Activations
            elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax,
                nn.Flatten, nn.Dropout, nn.GELU, nn.SiLU, nn.ELU,
                nn.CELU, nn.GLU, nn.Hardswish, nn.Mish
            )):
                structure.append({"type": "act", "module": m})
            
            # Pooling layers
            elif isinstance(m, (
                nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d
            )):
                structure.append({"type": "pool", "module": m})
        
        self._cache[model_id] = structure
        return structure
    
    def clear_cache(self) -> None:
        """Clear the structure cache."""
        self._cache.clear()
    
    def get_layers(
        self,
        structure: List[Dict[str, Any]]
    ) -> List[nn.Module]:
        """Extract only layer modules from structure."""
        return [item["module"] for item in structure if item["type"] == "layer"]
