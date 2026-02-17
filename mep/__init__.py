from .optimizers import SMEPOptimizer, SDMEPOptimizer

__version__ = "0.2.0"
__all__ = [
    'SMEPOptimizer',
    'SDMEPOptimizer'
]

# Optional CUDA module
try:
    from . import cuda
    __all__.append('cuda')
except ImportError:
    pass
