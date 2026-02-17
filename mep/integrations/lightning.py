from typing import Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
try:
    import lightning.pytorch as pl
except ImportError:
    # Fallback to older package name or mocks if needed,
    # but normally this module is only imported if lightning is present.
    try:
        import pytorch_lightning as pl
    except ImportError:
        raise ImportError("MEPLightningModule requires 'lightning' or 'pytorch-lightning' to be installed.")

from mep.optimizers import CompositeOptimizer

class MEPLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning Module optimized for Equilibrium Propagation (MEP).

    This class handles the manual optimization loop required for EP training,
    where `optimizer.step()` takes input `x` and target `target` arguments
    instead of a loss closure.

    Usage:
        class MyModel(MEPLightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(...)

            def forward(self, x):
                return self.model(x)

            def configure_optimizers(self):
                return smep(self.parameters(), model=self, mode='ep')

    Attributes:
        automatic_optimization (bool): set to False to enable manual control.
    """
    def __init__(self) -> None:
        super().__init__()
        # EP requires manual optimization to pass x, target to step()
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int):
        """
        Standard training step for EP.

        Expects `batch` to be a tuple (x, y).
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError(f"MEPLightningModule expects batch to be (x, y), got {type(batch)}")

        # Get optimizer
        # Note: self.optimizers() returns a single optimizer or a list.
        # We assume a single optimizer for standard EP.
        optimizer = self.optimizers()

        # Zero gradients
        optimizer.zero_grad()

        # EP Step
        # The step method in CompositeOptimizer handles the free/nudged phases.
        # It does NOT return the loss by default unless a closure is passed,
        # but passing closure is not the standard way for EP (which needs x, target).
        optimizer.step(x=x, target=y)

        # Logging / Metrics
        # Since optimizer.step() updates weights but doesn't return loss,
        # we perform a forward pass to compute metrics for logging.
        # This adds computational overhead but is necessary for monitoring.
        with torch.no_grad():
            output = self(x)
            loss = self.compute_loss(output, y)

            self.log("train_loss", loss, prog_bar=True)

            # Optional accuracy logging
            if hasattr(self, "compute_accuracy"):
                acc = self.compute_accuracy(output, y)
                self.log("train_acc", acc, prog_bar=True)

        return None # Manual optimization training_step doesn't need to return loss

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for logging purposes.
        Override this if you use a custom loss.
        """
        # Guess loss function based on target dtype
        if target.dtype in (torch.long, torch.int64):
            return nn.functional.cross_entropy(output, target)
        else:
            return nn.functional.mse_loss(output, target)

    def compute_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute accuracy for classification tasks.
        Override or remove if not applicable.
        """
        if target.dtype in (torch.long, torch.int64):
             pred = output.argmax(dim=1)
             correct = (pred == target).sum().item()
             return correct / target.size(0)
        return 0.0
