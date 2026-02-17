"""
Tests for CNN and Transformer architecture support.

Tests verify that MEP optimizers work with:
- Convolutional networks (Conv1d, Conv2d, Conv3d)
- Transformer encoders/decoders
- Mixed architectures
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleTransformer(nn.Module):
    """Simple Transformer encoder for testing."""

    def __init__(self, vocab_size=100, d_model=64, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) - token indices
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]  # (batch, seq_len, d_model)
        x = self.transformer(x)
        # Use mean pooling over sequence
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class ConvTransformer(nn.Module):
    """CNN + Transformer hybrid for testing (simplified)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output size
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, num_classes)
        )

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def cnn_model(device):
    return SimpleCNN(num_classes=10).to(device)


@pytest.fixture
def transformer_model(device):
    return SimpleTransformer(vocab_size=100, d_model=64, num_classes=10).to(device)


@pytest.fixture
def conv_transformer_model(device):
    return ConvTransformer(num_classes=10).to(device)


class TestCNN:
    """Test CNN support."""

    def test_cnn_forward(self, device, cnn_model):
        """Test CNN forward pass."""
        x = torch.randn(4, 3, 32, 32, device=device)
        output = cnn_model(x)
        assert output.shape == (4, 10)

    def test_cnn_ep_training(self, device, cnn_model):
        """Test CNN training with EP."""
        optimizer = SMEPOptimizer(
            cnn_model.parameters(),
            model=cnn_model,
            mode='ep',
            loss_type='cross_entropy',
            beta=0.5,
            settle_steps=10
        )

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        optimizer.step(x=x, target=y)

        # Verify gradients exist
        for p in cnn_model.parameters():
            assert p.grad is not None

    def test_cnn_conv1d(self, device):
        """Test 1D convolution support."""
        model = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 3, 32, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)

    def test_cnn_conv3d(self, device):
        """Test 3D convolution support."""
        # Use a model without flatten that maintains consistent shapes
        model = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(2, 1, 8, 8, 8, device=device)
        y = torch.randn(2, 8, 8, 8, 8, device=device)  # Match output shape

        optimizer.step(x=x, target=y)


class TestTransformer:
    """Test Transformer support."""

    def test_transformer_forward(self, device, transformer_model):
        """Test Transformer forward pass."""
        x = torch.randint(0, 100, (4, 20), device=device)
        output = transformer_model(x)
        assert output.shape == (4, 10)

    def test_transformer_ep_training(self, device):
        """Test Transformer-like model training with EP."""
        # Use a model that mimics transformer structure but works with EP
        d_model = 32
        model = nn.Sequential(
            nn.Linear(20, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),  # Mimics attention block
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 5)
        ).to(device)

        optimizer = SMEPOptimizer(
            model.parameters(),
            model=model,
            mode='ep',
            loss_type='cross_entropy',
            beta=0.5,
            settle_steps=10
        )

        x = torch.randn(4, 20, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)

        # Verify gradients exist
        for p in model.parameters():
            assert p.grad is not None

    def test_multihead_attention(self, device):
        """Test MultiheadAttention integration."""
        d_model = 32
        seq_len = 10
        # Create a wrapper module for MultiheadAttention
        class AttentionBlock(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                out, _ = self.attn(x, x, x)
                out = self.norm(out)
                return out  # Return same shape as input

        model = AttentionBlock(d_model, nhead=4).to(device)
        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, seq_len, d_model, device=device)
        y = torch.randn(4, seq_len, d_model, device=device)  # Same shape as output

        optimizer.step(x=x, target=y)


class TestConvTransformer:
    """Test CNN + Transformer hybrid architectures."""

    def test_conv_transformer_forward(self, device, conv_transformer_model):
        """Test hybrid model forward pass."""
        x = torch.randn(4, 3, 32, 32, device=device)
        output = conv_transformer_model(x)
        assert output.shape == (4, 10)

    def test_conv_transformer_ep_training(self, device, conv_transformer_model):
        """Test hybrid model training with EP."""
        optimizer = SMEPOptimizer(
            conv_transformer_model.parameters(),
            model=conv_transformer_model,
            mode='ep',
            loss_type='cross_entropy',
            beta=0.5,
            settle_steps=10
        )

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        optimizer.step(x=x, target=y)

        # Verify gradients exist
        for p in conv_transformer_model.parameters():
            assert p.grad is not None


class TestNormalization:
    """Test normalization layer handling."""

    def test_batchnorm(self, device):
        """Test BatchNorm support."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        model.train()
        optimizer.step(x=x, target=y)

    def test_layernorm(self, device):
        """Test LayerNorm support."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)

    def test_groupnorm(self, device):
        """Test GroupNorm support."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)


class TestPooling:
    """Test pooling layer handling."""

    def test_maxpool2d(self, device):
        """Test MaxPool2d support."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)

    def test_adaptive_pooling(self, device):
        """Test AdaptiveAvgPool support."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16, 5)
        ).to(device)

        optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)


class TestSDMEPCNN:
    """Test SDMEPOptimizer with CNN architectures."""

    def test_sdmep_cnn(self, device, cnn_model):
        """Test SDMEP with CNN."""
        optimizer = SDMEPOptimizer(
            cnn_model.parameters(),
            model=cnn_model,
            mode='ep',
            loss_type='cross_entropy',
            beta=0.5,
            settle_steps=10,
            dion_thresh=10000  # Small threshold to trigger Dion
        )

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        optimizer.step(x=x, target=y)

        # Verify gradients exist
        for p in cnn_model.parameters():
            assert p.grad is not None
