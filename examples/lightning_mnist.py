import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

# Import integration module
from mep.integrations import MEPLightningModule
from mep import smep

class MNISTClassifier(MEPLightningModule):
    """
    Example of using MEPLightningModule for MNIST classification.
    """
    def __init__(self, data_dir='./data', hidden_dim=128, learning_rate=0.02):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir

        # Define model architecture
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Return SMEP optimizer in EP mode
        # Note: We must pass 'model=self' so EP can inspect/manipulate it.
        # But 'self' is the LightningModule which wraps the actual model.
        # The optimizer needs to know the graph structure.
        # If we pass `self`, the graph might be weird due to Lightning wrappers.
        # It's safer to pass `self.model` (the nn.Sequential).
        # However, `self.parameters()` comes from `self`.

        return smep(
            self.parameters(),
            model=self.model, # Use the inner sequential model for structure inspection
            mode='ep',
            lr=self.hparams.learning_rate,
            beta=0.5,
            settle_steps=10
        )

    # Optional: Override training_step if you need custom logic beyond default EP
    # But MEPLightningModule provides a default one.

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_full = MNIST(self.data_dir, train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64, num_workers=4)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

if __name__ == '__main__':
    # Initialize model
    model = MNISTClassifier()

    # Initialize trainer
    # fast_dev_run runs 1 batch of train/val/test to verify setup
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True
    )

    print("Training with PyTorch Lightning + Equilibrium Propagation...")
    trainer.fit(model)

    print("Testing...")
    trainer.test(model)
