# MEP: Muon Equilibrium Propagation

### 🧠 Biologically Plausible Deep Learning Without Backpropagation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/your-username/mep/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/mep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📄 Abstract

**Equilibrium Propagation (EP)** offers a biologically plausible alternative to backpropagation by estimating gradients through the contrast between two equilibrium states of an energy-based model. However, historical implementations have suffered from training instability, poor convergence, and impractical computational requirements—preventing EP from scaling to modern deep learning tasks.

We present **Spectral Dion-Muon Equilibrium Propagation (SDMEP)**, a refactored optimization framework that addresses these limitations through three key innovations:

1.  **Spectral Constraints (S):** Enforcing σ(W) ≤ γ < 1 guarantees convergence to a unique fixed point, eliminating the oscillatory divergence that plagued earlier EP implementations.
2.  **Dion Low-Rank Updates (D):** For large weight matrices, low-rank SVD with error feedback reduces computational cost while preserving gradient information in the dominant subspace.
3.  **Muon Orthogonalization (M):** Newton-Schulz iteration orthogonalizes gradients, improving conditioning and enabling stable training at greater depths.

This framework is designed as a research platform for exploring biologically plausible learning, neuromorphic computing, continual learning, and energy-efficient deep learning on analog hardware.

**Keywords:** Equilibrium Propagation, Biologically Plausible Learning, Energy-Based Models, Spectral Normalization, Low-Rank Optimization, Neuromorphic Computing, Continual Learning

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Introduction](#-introduction-the-backpropagation-bottleneck)
- [The MEP Framework](#-the-mep-framework)
- [Quick Start](#-quick-start)
- [Optimizer Selection Guide](#-optimizer-selection-guide)
- [Architecture: Strategy Pattern](#-architecture-strategy-pattern)
- [Understanding EP](#-understanding-ep-a-visual-guide)
- [Open Research Questions](#-open-research-questions)
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

MEP is built on the theory of **Energy Based Models (EBMs)** with contractive dynamics. Given an input x and network states s = {s₁, ..., sₗ}, we define the energy:

```
E(x, s, y) = E_internal + E_external

E_internal = 0.5 × Σ ||sᵢ - fᵢ(sᵢ₋₁)||²     (state consistency)
E_external = β × L(s_last, y)                (task loss)
```

**Free phase** (β = 0): States settle to minimize E_internal, reaching a fixed point s*.

**Nudged phase** (β > 0): The target y perturbs the energy landscape, yielding a new fixed point s^β.

**EP Gradient:** The contrast (s^β - s*) / β approximates ∂L/∂W without backpropagation.

### The Safety Harness: S-D-M

| Component | Purpose | Mechanism |
|-----------|---------|-----------|
| **Spectral (S)** | Stability | Power iteration enforces σ(W) ≤ γ, ensuring contractive dynamics and unique fixed points. |
| **Dion (D)** | Efficiency | Low-rank SVD (U Σ V^T) with error feedback for matrices >100K parameters. |
| **Muon (M)** | Conditioning | Newton-Schulz iteration orthogonalizes gradients: X_{k+1} = ½ X_k (3I - X_k^T X_k). |

---

## 🔧 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
import torch.nn as nn
from mep import smep, sdmep, muon_backprop

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Option 1: EP mode (biologically plausible)
optimizer = smep(model.parameters(), model=model, mode='ep')
optimizer.step(x=x, target=y)  # No .backward() needed!

# Option 2: Backprop mode (drop-in SGD replacement)
optimizer = muon_backprop(model.parameters())
loss.backward()
optimizer.step()
```

### Recommended Configuration

```python
from mep import smep

# For classification
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    beta=0.5,
    settle_steps=10,
    settle_lr=0.05,
    loss_type='mse',
    use_error_feedback=False,  # Critical for stability
    ns_steps=5,
    gamma=0.95,
)

# For continual learning
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    beta=0.5,
    settle_steps=10,
    settle_lr=0.05,
    loss_type='mse',
    use_error_feedback=True,   # Enables memory retention
    error_beta=0.95,           # High retention
)
```

---

## 🎯 Optimizer Selection Guide

| Use Case | Recommended | Configuration Notes |
|----------|-------------|---------------------|
| Standard classification | **Adam/SGD** | Default settings |
| Biological plausibility research | **SMEP** | `use_error_feedback=False` |
| Continual/lifelong learning | **SMEP+EF** | `use_error_feedback=True, error_beta=0.95` |
| Memory-constrained (deep nets) | **EP** | O(1) memory vs O(depth) for backprop |
| Neuromorphic hardware | **SMEP/LocalEP** | Local learning rules |
| Very deep networks | **Muon** | Backprop + orthogonalization |
| Large models (>1M params/layer) | **SDMEP** | `dion_thresh=200000` |

### Key Observations from Benchmarking

- **Classification**: EP-based methods show competitive performance with careful hyperparameter tuning
- **Continual Learning**: Error feedback dramatically reduces catastrophic forgetting
- **Regression**: EP shows instability despite natural MSE alignment—an open research problem
- **Speed**: EP is ~3× slower than backprop due to settling iterations

---

## 🏗️ Architecture: Strategy Pattern

The refactored MEP uses a **strategy pattern** for maximum flexibility and extensibility:

```
CompositeOptimizer
├── GradientStrategy    (how to compute ∇L)
│   ├── BackpropGradient    # Standard .backward()
│   ├── EPGradient          # Free/nudged phase contrast
│   ├── LocalEPGradient     # Layer-local updates only
│   └── NaturalGradient     # Fisher Information whitening
├── UpdateStrategy      (how to transform ∇L → ΔW)
│   ├── PlainUpdate         # Vanilla SGD
│   ├── MuonUpdate          # Newton-Schulz orthogonalization
│   ├── DionUpdate          # Low-rank SVD for large matrices
│   └── FisherUpdate        # Natural gradient descent
├── ConstraintStrategy  (how to enforce constraints)
│   ├── NoConstraint        # Unconstrained
│   └── SpectralConstraint  # σ(W) ≤ γ
└── FeedbackStrategy    (how to accumulate residuals)
    ├── NoFeedback          # Standard optimization
    └── ErrorFeedback       # Accumulate residuals (continual learning)
```

### Custom Composition

```python
from mep.optimizers import (
    CompositeOptimizer,
    EPGradient, MuonUpdate, SpectralConstraint, ErrorFeedback
)

# Custom optimizer for continual learning
optimizer = CompositeOptimizer(
    model.parameters(),
    gradient=EPGradient(beta=0.5, settle_steps=10),
    update=MuonUpdate(ns_steps=5),
    constraint=SpectralConstraint(gamma=0.95),
    feedback=ErrorFeedback(beta=0.95),
    lr=0.01,
    model=model,
)
```

### Debugging with EPMonitor

```python
from mep import smep, EPMonitor

monitor = EPMonitor()
optimizer = smep(model.parameters(), model=model, mode='ep')

for epoch in range(epochs):
    monitor.start_epoch()
    
    for x, y in train_loader:
        optimizer.step(x=x, target=y)
    
    metrics = monitor.end_epoch(model, optimizer)
    print(f"Epoch {epoch}: Energy gap = {metrics.energy_gap:.4f}")
    
    if not monitor.check_convergence():
        print("Warning: Settling may not have converged!")

print(monitor.summary())
```

---

## 🔮 Understanding EP: A Visual Guide

### Free Phase vs Nudged Phase

```
Free Phase (β = 0):
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
         │  ▲        │  ▲        │  ▲
         │  │        │  │        │  │
         └──┴────────┴──┴────────┴──┘
              States settle to minimize E_internal

Nudged Phase (β > 0):
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
         │  ▲        │  ▲        │  ▲         │
         │  │        │  │        │  │         │ (target nudges)
         └──┴────────┴──┴────────┴──┴─────────┘
              Target perturbs energy landscape

EP Gradient = (nudged_states - free_states) / β
```

### Energy Function

```
E = E_internal + E_external

E_internal = 0.5 × Σ ||sᵢ - fᵢ(sᵢ₋₁)||²   (state consistency)
E_external = β × L(s_last, y)             (task loss)

For classification with MSE:
  L = ||s_last - one_hot(y)||²

For classification with CrossEntropy:
  L = CrossEntropy(softmax(s_last), y)
```

---

## 🔮 Open Research Questions

### 1. Why Does Regression Fail?

Despite EP's energy function naturally matching MSE loss, we observed severe instability during training.

**Hypotheses:**
- Settling dynamics create positive feedback loop
- Error feedback accumulates in wrong direction
- Energy landscape has poor local minima

**Potential fixes:**
- Lower settling learning rate
- Gradient clipping during settling
- Energy-based early stopping
- Different energy function formulation

**This is an open problem—contributions welcome!**

### 2. Closing the Classification Gap

EP-based methods show competitive but not state-of-the-art performance on standard benchmarks.

**Potential improvements:**
- Adaptive settling (stop when energy converges)
- Better energy functions for classification
- Layer-wise learning rates
- Batch normalization integration
- Deeper architecture studies

### 3. SDMEP for Large Models

Dion (low-rank SVD) should shine for large matrices but requires careful tuning.

**Needed:**
- Better rank selection heuristics (adaptive based on gradient spectrum)
- Automatic threshold selection
- Performance characterization for different model sizes

**Promise:** For models with very large layers, Dion could provide significant speedup.

### 4. Neuromorphic Hardware Integration

EP's local learning rules are a natural fit for analog hardware, but public implementations are limited.

**Potential targets:**
- Optical neural networks (continuous-time dynamics)
- Memristor crossbars (local Hebbian updates)
- Spiking neural networks (event-based processing)
- Analog chips (natural fit for settling dynamics)

**This is a prime research opportunity!**

### 5. Continual Learning Mechanisms

Error feedback shows promise for reducing catastrophic forgetting, but the mechanism requires deeper understanding.

**Questions:**
- How much history does the buffer retain?
- What is the optimal error_beta for different task sequences?
- Can we combine with explicit replay for even better results?
- Does this work for domain-incremental (not just task-incremental) learning?

### 6. Application Domain Discovery

We are actively exploring where EP-based optimizers excel:

- **Continual/lifelong learning**: Error feedback as implicit replay
- **Memory-constrained scenarios**: O(1) memory for very deep networks
- **Neuromorphic hardware**: Natural fit for local learning rules
- **Energy-based modeling**: Direct alignment with EP's energy formulation
- **Online learning**: Event-based dynamics for streaming data

**The full potential of EP is still being discovered.**

---

## 📚 References

1.  Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. *Frontiers in Computational Neuroscience*, 11, 24.

2.  Jordan, K. (2024). The Muon Optimizer. *GitHub Repository*. https://github.com/KellerJordan/Muon

3.  Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.

4.  Scellier, B., Franceschi, L., & Bengio, Y. (2024). Energy-Based Learning in Continuous Time. *arXiv preprint*.

5.  Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the Brain. *Nature Reviews Neuroscience*, 21(6), 335-346.

6.  Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

7.  Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*, 114(13), 3521-3526.

---

## 📁 Module Structure

```
mep/
├── optimizers/
│   ├── composite.py       # Main CompositeOptimizer
│   ├── strategies/
│   │   ├── gradient.py    # Backprop, EP, LocalEP, Natural
│   │   ├── update.py      # Muon, Dion, Fisher
│   │   ├── constraint.py  # Spectral norm
│   │   └── feedback.py    # Error feedback
│   ├── energy.py          # Energy function
│   ├── settling.py        # Settling dynamics
│   ├── monitor.py         # EP debugging utilities
│   └── inspector.py       # Model structure extraction
├── presets/
│   └── __init__.py        # Factory functions (smep, sdmep, etc.)
├── benchmarks/
│   ├── tuned_compare.py   # Classification benchmarks
│   └── niche_benchmarks.py # Regression, continual learning
├── cuda/
│   └── kernels.py         # CUDA-accelerated operations
└── optimizers_legacy.py   # Archived original implementation
```

---

## 🤝 Contributing

Contributions welcome! High-priority areas:

1.  **Fix regression instability** - EP should excel here
2.  **Adaptive settling** - Early stopping for speedup
3.  **SDMEP tuning** - Better rank selection for large models
4.  **Continual learning benchmarks** - More task sequences, domains
5.  **Hardware demos** - Neuromorphic chip implementations

```bash
# Development setup
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
