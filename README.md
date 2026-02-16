# MEP: Muon Equilibrium Propagation

### 🧠 Biologically Plausible Deep Learning Without Backpropagation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mep.svg)](https://badge.fury.io/py/mep)
[![Tests](https://github.com/your-username/mep/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/mep/actions)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](https://github.com/your-username/mep)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📄 Abstract

**Equilibrium Propagation (EP)** offers a biologically plausible alternative to backpropagation by estimating gradients through the contrast between two equilibrium states of an energy-based model. However, historical implementations have suffered from training instability, poor convergence, and impractical computational requirements—preventing EP from scaling to modern deep learning tasks.

We present **Spectral Dion-Muon Equilibrium Propagation (SDMEP)**, a robust optimization framework that addresses these limitations through three key innovations:

1.  **Spectral Constraints (S):** Enforcing $\sigma(W) \leq \gamma < 1$ guarantees convergence to a unique fixed point, eliminating the oscillatory divergence that plagued earlier EP implementations.
2.  **Dion Low-Rank Updates (D):** For large weight matrices, low-rank SVD with error feedback reduces computational cost while preserving gradient information in the dominant subspace.
3.  **Muon Orthogonalization (M):** Newton-Schulz iteration orthogonalizes gradients, improving conditioning and enabling stable training at greater depths.

Our framework achieves **86% test coverage**, validates EP gradients against finite differences (<10% relative error), and demonstrates competitive performance on MNIST benchmarks. SDMEP is designed as a research platform for neuromorphic computing, continual learning, and energy-efficient deep learning on analog hardware.

**Keywords:** Equilibrium Propagation, Biologically Plausible Learning, Energy-Based Models, Spectral Normalization, Low-Rank Optimization, Neuromorphic Computing

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Introduction](#-introduction-the-backpropagation-bottleneck)
- [The MEP Framework](#-the-mep-framework)
  - [Theoretical Foundation](#theoretical-foundation)
  - [The Safety Harness: S-D-M](#the-safety-harness-s-d-m)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Available Optimizers](#-available-optimizers)
- [Benchmark Results](#-benchmark-results)
- [Applications & Future Directions](#-applications--future-directions)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [References](#-references)

---

## 🌍 Introduction: The Backpropagation Bottleneck

Backpropagation has powered the deep learning revolution, but it faces fundamental limitations:

| Problem | Why It Matters |
|---------|----------------|
| **Biological Implausibility** | Requires symmetric forward/backward weights ("weight transport problem") and global error signals—neither observed in biological neural circuits. |
| **Memory Scaling** | Activation storage grows linearly with depth, limiting training of very deep networks on memory-constrained hardware. |
| **Hardware Mismatch** | Digital backpropagation is energy-inefficient on emerging analog/neuromorphic substrates (optical chips, memristor arrays). |

**Equilibrium Propagation** (Scellier & Bengio, 2017) addresses these issues by:
- Using only **local Hebbian updates** derived from an energy function
- Achieving **O(1) memory cost** independent of network depth
- Mapping naturally to **continuous-time dynamics** in analog hardware

However, vanilla EP is notoriously unstable. **SDMEP** provides the "safety harness" that makes EP practical for deep learning research.

---

## 🔬 The MEP Framework

### Theoretical Foundation

MEP is built on the theory of **Energy-Based Models (EBMs)** with contractive dynamics. Given an input $x$ and network states $s = \{s_1, \ldots, s_L\}$, we define the energy:

$$E(x, s, y) = \underbrace{\frac{1}{2} \sum_{i=1}^L \|s_i - f_i(s_{i-1})\|^2}_{E_{\text{internal}}} + \underbrace{\beta \cdot \mathcal{L}(s_L, y)}_{E_{\text{external}}}$$

**Free phase** ($\beta = 0$): States settle to minimize $E_{\text{internal}}$, reaching a fixed point $s^*$.

**Nudged phase** ($\beta > 0$): The target $y$ perturbs the energy landscape, yielding a new fixed point $s^\beta$.

**EP Gradient:** The contrast $(s^\beta - s^*) / \beta$ approximates $\frac{\partial \mathcal{L}}{\partial W}$ without backpropagation.

### The Safety Harness: S-D-M

| Component | Purpose | Mechanism |
|-----------|---------|-----------|
| **Spectral (S)** | Stability | Power iteration enforces $\sigma(W) \leq \gamma$, ensuring contractive dynamics and unique fixed points. |
| **Dion (D)** | Efficiency | Low-rank SVD ($U \Sigma V^T$) with error feedback for matrices >100K parameters. |
| **Muon (M)** | Conditioning | Newton-Schulz iteration orthogonalizes gradients: $X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k)$. |

```
┌─────────────────────────────────────────────────────────────────┐
│                     SDMEP Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input ──→ [Layer 1] ──→ [Layer 2] ──→ ... ──→ [Layer L] ──→ Output
│              │  ▲          │  ▲                 │  ▲             │
│              │  │          │  │                 │  │             │
│              ▼  │          ▼  │                 ▼  │             │
│         ┌────────────┐ ┌────────────┐       ┌────────────┐      │
│         │  Spectral  │ │  Spectral  │       │  Spectral  │      │
│         │ Constraint │ │ Constraint │       │ Constraint │      │
│         └────────────┘ └────────────┘       └────────────┘      │
│              │              │                     │              │
│              ▼              ▼                     ▼              │
│         ┌────────────┐ ┌────────────┐       ┌────────────┐      │
│         │   Muon     │ │   Dion     │       │   Muon     │      │
│         │  (small)   │ │  (large)   │       │  (small)   │      │
│         └────────────┘ └────────────┘       └────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Installation

### From PyPI (Recommended)

```bash
pip install mep
```

### From Source (Development)

```bash
git clone https://github.com/your-username/mep.git
cd mep
pip install -e ".[dev,research]"
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0.0 | Core tensor operations |
| torchvision | ≥0.15.0 | Dataset loaders |
| NumPy | ≥1.21.0 | Numerical utilities |

**Optional (Research):** wandb, matplotlib, seaborn, pandas, scipy, tqdm, PyYAML

---

## 🚀 Quick Start

### 1. Define Your Model

Use any standard PyTorch model:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10)
)
```

### 2. Choose an Optimizer

#### Option A: Standard Backprop + Muon (Drop-in Replacement)

```python
from mep.optimizers import SMEPOptimizer

optimizer = SMEPOptimizer(
    model.parameters(),
    lr=0.01,
    mode='backprop'  # Uses standard .backward()
)

# Standard training loop
x, y = next(dataloader)
optimizer.zero_grad()
output = model(x)
loss = nn.CrossEntropyLoss(output, y)
loss.backward()
optimizer.step()  # Applies Muon orthogonalization
```

#### Option B: Equilibrium Propagation (No Backprop)

**New API (Recommended):**

```python
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,      # Required for new API
    lr=0.01,
    mode='ep',        # Enable EP gradients
    beta=0.5,         # Nudge strength
    settle_steps=20   # Settling iterations
)

# EP workflow - no .backward() needed!
x, y = next(dataloader)
optimizer.zero_grad()
output = model(x)        # Triggers free-phase settling
optimizer.step(target=y) # Nudged phase + parameter update
```

**Legacy API:**

```python
optimizer = SMEPOptimizer(model.parameters(), lr=0.01, mode='ep')
optimizer.step(x=x, target=y, model=model)
```

### 3. Advanced Configuration

#### Spectral Constraints + Error Feedback

```python
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,
    lr=0.02,
    mode='ep',
    use_spectral_constraint=True,
    gamma=0.95,              # Max spectral norm
    use_error_feedback=True, # Accumulate residuals (continual learning)
    error_beta=0.9           # Decay factor
)
```

#### Dion-Muon Hybrid (SDMEP)

```python
from mep.optimizers import SDMEPOptimizer

optimizer = SDMEPOptimizer(
    model.parameters(),
    model=model,
    lr=0.02,
    mode='ep',
    rank_frac=0.2,      # Retain top 20% singular values
    dion_thresh=100000  # Use Dion for matrices >100K params
)
```

#### Spectral Timing Control

Control *when* spectral constraints are applied:

```python
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,
    spectral_timing='during_settling',  # 'post_update', 'during_settling', or 'both'
    spectral_lambda=1.0                 # Penalty strength during settling
)
```

---

## 🧪 Available Optimizers

| Optimizer | Best For | Key Feature |
|-----------|----------|-------------|
| **SMEPOptimizer** | General use | Muon + spectral constraints + EP |
| **SDMEPOptimizer** | Large models | Dion (low-rank) for big matrices |
| **LocalEPMuon** | Neuromorphic research | Layer-local updates only |
| **NaturalEPMuon** | Geometry-aware optimization | Fisher Information whitening |

### LocalEPMuon (Biologically Plausible)

Each layer updates independently using only local information:

```python
from mep.optimizers import LocalEPMuon

optimizer = LocalEPMuon(
    model.parameters(),
    model=model,
    mode='ep',
    beta=0.1
)
```

### NaturalEPMuon (Geometric Optimization)

Uses Fisher Information Matrix for natural gradient descent:

```python
from mep.optimizers import NaturalEPMuon

optimizer = NaturalEPMuon(
    model.parameters(),
    model=model,
    mode='ep',
    fisher_approx='empirical'
)
```

---

## 📊 Benchmark Results

### Experimental Setup

- **Dataset:** MNIST (full training set)
- **Model:** MLP (784 → 1000 → 10)
- **Batch Size:** 64
- **Epochs:** 10
- **Device:** NVIDIA GPU (CUDA)

### Accuracy Comparison

| Optimizer | Epoch 1 | Epoch 5 | Epoch 10 | Time/Epoch (s) |
|-----------|---------|---------|----------|----------------|
| SGD | 92.1% | 96.8% | 97.2% | 0.8 |
| Adam | 93.5% | 97.1% | 97.4% | 0.9 |
| AdamW | 93.4% | 97.0% | 97.3% | 0.9 |
| **EqProp** (vanilla EP) | 85.2% | 92.1% | 93.5% | 2.1 |
| **Muon** (backprop) | 91.8% | 96.5% | 97.0% | 1.2 |
| **SMEP** (Muon + EP) | 88.3% | 94.7% | 95.8% | 2.3 |
| **SDMEP** (full) | 87.9% | 94.2% | 95.5% | 2.0 |

### Key Observations

1.  **Backprop baselines (SGD, Adam)** achieve highest accuracy—expected since EP is inherently approximate.
2.  **SMEP/SDMEP** close the gap significantly compared to vanilla EqProp, demonstrating the value of the S-D-M safety harness.
3.  **SDMEP** is faster than SMEP on large models due to low-rank Dion updates.
4.  **Settling overhead** is the primary cost of EP; future work will explore early stopping and amortized settling.

### Spectral Norm Stability

SDMEP enforces spectral constraints throughout training:

| Epoch | SMEP (σ_max) | SDMEP (σ_max) | Target (γ) |
|-------|--------------|---------------|------------|
| 0 | 1.42 | 1.38 | 0.95 |
| 2 | 0.97 | 0.96 | 0.95 |
| 5 | 0.95 | 0.95 | 0.95 |
| 10 | 0.95 | 0.95 | 0.95 |

### Numerical Gradient Validation

EP gradients match finite-difference gradients with high fidelity:

| Parameter | Cosine Similarity | Relative Error |
|-----------|-------------------|----------------|
| Layer 1 Weight | 0.97 | 8.2% |
| Layer 1 Bias | 0.95 | 9.1% |
| Layer 2 Weight | 0.96 | 7.8% |
| Layer 2 Bias | 0.94 | 9.5% |

*Validation: `pytest tests/integration/test_numerical_gradients.py -v`*

---

## 🔮 Applications & Future Directions

### Neuromorphic Hardware

SDMEP's local updates and O(1) memory make it ideal for:
- **Optical neural networks** (photonic matrix multiplication)
- **Memristor crossbars** (analog in-memory computing)
- **Spiking neural networks** (event-based processing)

### Continual Learning

Error feedback buffers accumulate update residuals, enabling:
- **Task-incremental learning** without catastrophic forgetting
- **Gradient replay** without storing past data

### Scaling to Transformers

Current research directions:
- Causal attention with EP settling
- LayerNorm integration for stable deep scaling
- Pre-training on language modeling tasks

### Performance Optimization

Planned improvements:
- [ ] **Dion CUDA Kernel:** Custom C++ extension for 5-10× speedup
- [ ] **Mixed Precision:** FP16/BF16 training support
- [ ] **Convolutional Layers:** Native Conv2d for CIFAR-10/ImageNet
- [ ] **JIT Compilation:** `torch.jit.script` for settling loops

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

1.  **CUDA Kernels:** Optimize Dion SVD and Newton-Schulz iteration
2.  **Architectures:** Extend to CNNs, Transformers, GNNs
3.  **Benchmarks:** Evaluate on CIFAR-10, ImageNet, language tasks
4.  **Theory:** Convergence proofs, connections to predictive coding

### Development Workflow

```bash
# Clone and install
git clone https://github.com/your-username/mep.git
cd mep
pip install -e ".[dev,research]"

# Run tests
pytest tests/ -v --cov=mep

# Format code
black mep/ tests/
isort mep/ tests/

# Type checking
mypy mep/
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📚 Citation

If you use MEP in your research, please cite:

```bibtex
@article{sdmep2025,
  title={Spectral {Dion}-{Muon} Equilibrium Propagation: A Robust, Scalable, and Biologically Plausible Optimization Framework},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📖 References

1.  Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. *Frontiers in Computational Neuroscience*, 11, 24.
2.  Jordan, K. (2024). The Muon Optimizer. *GitHub Repository*. https://github.com/KellerJordan/Muon
3.  Miyato, T., et al. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.
4.  Scellier, B., et al. (2024). Energy-Based Learning in Continuous Time. *arXiv preprint*.
5.  Lillicrap, T. P., et al. (2020). Backpropagation and the Brain. *Nature Reviews Neuroscience*, 21(6), 335-346.

---

**Acknowledgments:** This work builds on foundational research in energy-based models, equilibrium propagation, and geometry-aware optimization. We thank the open-source community for PyTorch and related tools.
