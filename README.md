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

# Default settings work well for most classification tasks
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    beta=0.3,               # Nudging strength
    settle_steps=15,        # Settling iterations
    settle_lr=0.1,          # Settling learning rate
    loss_type='cross_entropy',
    # use_error_feedback=False by default
)

# For continual learning (with error feedback)
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    use_error_feedback=True,  # Enable for CL
    error_beta=0.95,          # High retention
)
```

---

## 🎯 Optimizer Selection Guide

| Use Case | Recommended | Configuration Notes |
|----------|-------------|---------------------|
| Standard classification | **Adam/SGD** | Default settings |
| Biological plausibility research | **SMEP** | `mode='ep'` |
| Memory-constrained (deep nets) | **EP** | O(1) memory vs O(depth) for backprop |
| Neuromorphic hardware | **SMEP/LocalEP** | Local learning rules |
| Very deep networks | **Muon** | Backprop + orthogonalization |
| Large models (>1M params/layer) | **SDMEP** | `dion_thresh=200000` |

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
    └── ErrorFeedback       # Accumulate residuals
```

### Custom Composition

```python
from mep.optimizers import (
    CompositeOptimizer,
    EPGradient, MuonUpdate, SpectralConstraint, ErrorFeedback
)

# Custom optimizer with error feedback
optimizer = CompositeOptimizer(
    model.parameters(),
    gradient=EPGradient(beta=0.3, settle_steps=15),
    update=MuonUpdate(ns_steps=5),
    constraint=SpectralConstraint(gamma=0.95),
    feedback=ErrorFeedback(beta=0.9),
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

### What Makes EP Different?

EP is not just "backprop without the backward pass." It has qualitatively different properties:

| Property | Backpropagation | Equilibrium Propagation |
|----------|----------------|------------------------|
| **Gradient flow** | Chain rule through computation graph | Contrast between equilibrium states |
| **Memory** | O(depth) - stores all activations | O(1) - only current states |
| **Weight transport** | Requires symmetric forward/backward weights | Uses same weights for both directions |
| **Update locality** | Global error signal | Layer-local energy minimization |
| **Temporal dynamics** | Instant gradient computation | Iterative settling process |
| **Biological plausibility** | Low (weight transport problem) | Higher (local Hebbian updates) |

### When EP Might Matter

1. **Memory-constrained training**: EP's O(1) activation storage could enable training deeper networks on limited hardware

2. **Continual learning**: Error feedback with EP may reduce catastrophic forgetting (research direction)

3. **Neuromorphic hardware**: EP's local learning rules map naturally to analog substrates

4. **Energy-based interpretation**: EP provides an energy-based view of learning, which may offer theoretical insights

5. **Language modeling**: Sequential prediction tasks where EP's dynamics differ from BPTT

### When to Use Backprop

- Standard classification/regression tasks
- Production training pipelines
- When training speed is critical
- When maximum accuracy is the goal

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

## 💡 When to Use EP: Honest Positioning

### EP's Unique Value

**EP is NOT designed to replace backpropagation for standard deep learning.** Backprop excels at production training, maximum accuracy, and speed-critical applications.

**EP's value lies in specific research niches:**

| Use Case | Why EP Matters | Status |
|----------|---------------|--------|
| **Biological plausibility research** | No weight transport problem; local Hebbian updates match neural circuits | ✅ Working |
| **Neuromorphic hardware** | Event-based dynamics map naturally to analog substrates | 🔬 Research needed |
| **Energy-based model research** | Direct energy minimization framework | ✅ Working |
| **Educational tool** | Demonstrates alternatives to backpropagation | ✅ Working |
| **Continual learning** | Error feedback may reduce forgetting | 🔬 Needs validation |
| **Memory-constrained training** | O(1) activation storage theoretical | 🔬 Needs validation |

### When to Use Backprop Instead

- **Standard classification/regression**: Backprop is faster and more accurate
- **Production pipelines**: Backprop has mature tooling
- **Training speed critical**: EP is ~1.5-3× slower due to settling
- **Maximum accuracy goals**: Backprop remains state-of-the-art

### EP's Qualitative Differences

EP exhibits fundamentally different learning dynamics:

1. **No backward pass through computation graph** - Gradients emerge from energy minimization
2. **Local learning rules** - Each layer updates based on local state contrasts
3. **Iterative settling** - Network reaches equilibrium before updating
4. **Contrastive learning** - Compares free vs nudged states
5. **Energy-based interpretation** - Learning minimizes energy function

### Current Validation Status

| Claim | Theoretical Basis | Empirical Status |
|-------|------------------|------------------|
| O(1) activation memory | ✓ Sound | ➔ Needs proper validation |
| Reduced catastrophic forgetting | ✓ Plausible | ➔ Needs proper benchmark |
| Biologically plausible | ✓ Yes | ✓ Local rules confirmed |
| Works on sequential tasks | ✓ Yes | ➔ Character LM example added |
| Competitive accuracy | ✓ Possible | ✓ Classification works (~89% MNIST) |

### Research Contributions

Even if EP doesn't "beat" backprop on standard benchmarks, it provides:

- **Research tool** for studying biologically plausible learning mechanisms
- **Educational value** demonstrating alternative learning paradigms
- **Neuromorphic foundation** for future analog hardware deployment
- **Theoretical insights** from energy-based learning perspective

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

Contributions welcome! High-priority areas based on [ROADMAP.md](ROADMAP.md):

### Immediate Priorities (Week 1-2)

1.  **Memory validation study** - Run `examples/validate_memory_scaling.py` at extreme depths (1000+ layers)
2.  **Continual learning benchmark** - Proper sequential training with forgetting metrics
3.  **Character LM tuning** - Improve text generation quality with EP
4.  **Documentation** - Document EP's qualitative learning dynamics

### Medium-Term (Month 1-2)

5.  **Add baselines** - EWC, GEM for continual learning comparison
6.  **Deep scaling study** - Test at 1000+ layers with gradient checkpointing
7.  **Publish results** - Honest assessment of EP's tradeoffs

### Long-Term (Month 3+)

8.  **Find EP's niches** - Identify domains where EP excels
9.  **Neuromorphic demos** - Partner with hardware groups
10. **PyTorch Lightning integration** - If/when adoption warrants it

```bash
# Development setup
pip install -e ".[dev]"
pytest tests/ -v
```

### How to Contribute

- **Validation studies**: Run benchmarks and report results (positive or negative)
- **Bug reports**: Include minimal reproduction examples
- **Feature requests**: Explain use case and why it matters for EP research
- **Documentation**: Improve examples, fix typos, add tutorials

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
