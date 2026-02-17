"""
Integration modules for third-party frameworks.
"""

# Try to import lightning integration, but fail gracefully if not installed
try:
    from .lightning import MEPLightningModule
    __all__ = ["MEPLightningModule"]
except ImportError:
    __all__ = []
