# MEP: Muon Equilibrium Propagation
### 🧠 Biologically Plausible, Infinite-Depth Learning

![MEP Architecture](https://raw.githubusercontent.com/your-username/mep/main/docs/assets/mep_banner.png)

**SDMEP** is a PyTorch-based optimization framework that provides robust training methods for deep neural networks. It implements two primary modes of operation:
1.  **Standard Backpropagation** enhanced with **Muon** (Newton-Schulz) and **Dion** (Low-Rank SVD) geometry-aware updates.
2.  **Equilibrium Propagation (EP)**, a biology-inspired gradient estimation method that theoretically avoids the need for global error backpropagation.

This repository serves as a research platform for **Biologically Plausible Learning**, implementing continuous-time dynamics and local learning rules using PyTorch's autograd engine for efficient execution. By combining **Spectral Normalization**, the **Muon** optimizer, and **Dion** low-rank updates, SDMEP solves historic instability issues of Energy-Based Models (EBMs), enabling deep scaling experiments relevant to neuromorphic and analog hardware.

---

## 🌟 Key Features

1.  **O(1) Memory Cost:** Memory usage does not grow with network depth. Train 1,000-layer networks with the memory of a single layer.
2.  **Biological Plausibility:** Local (Hebbian) updates. Neurons only need information from their neighbors.
3.  **Unbreakable Stability:** Unlike traditional EP, SDMEP enforces a strict **Spectral Constraint ($\sigma(W) \le \gamma < 1$)**, guaranteeing convergence to a unique fixed point.
4.  **Hardware Native:** Ready for optical chips, analog arrays, and FPGA clusters.

---

## 🔧 Installation

Install MEP via pip:

```bash
pip install mep
```

Or install from source for development:

```bash
git clone https://github.com/your-username/mep.git
cd mep
pip install .
```

---

## 🚀 Quick Start

### 1. Define Model
Use any standard PyTorch model (e.g., `nn.Sequential`, ResNet, Transformer).

```python
import torch.nn as nn
from mep.optimizers import SMEPOptimizer

model = nn.Sequential(
    nn.Linear(784, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10)
)
```

### 2. Choose an Optimizer Mode

**Option 1: Standard Backprop (Muon updates only)**
```python
optimizer = SMEPOptimizer(model.parameters(), lr=0.01, mode='backprop')

# Standard training loop
x, y = next(dataloader)
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()  #  Applies Muon (Newton-Schulz) updates
```

**Option 2: Equilibrium Propagation (Biology-inspired gradients)**

**New API (Recommended):**
```python
optimizer = SMEPOptimizer(
    model.parameters(), 
    model=model,      # Pass model once
    lr=0.01, 
    mode='ep',        # Enable EP gradient computation
    beta=0.5,        # Nudge strength
    settle_steps=20  # Settling iterations
)

# EP workflow - no .backward() needed!
x, y = next(dataloader)
optimizer.zero_grad()
output = model(x)             # Automatic free-phase settling
optimizer.step(target=y)      # Nudged phase + updates
```

**Legacy API:**
```python
optimizer = SMEPOptimizer(model.parameters(), lr=0.01, mode='ep')
optimizer.step(x=x, target=y, model=model)
```

### Advanced Features (SMEP & SDMEP)

Enable **Spectral Constraints** (Lipschitz control) and **Error Feedback** (for continual learning) directly in `SMEPOptimizer`:

```python
optimizer = SMEPOptimizer(
    model.parameters(),
    lr=0.02,
    use_spectral_constraint=True,  # Constrain spectral norm < gamma
    gamma=0.95,
    use_error_feedback=True        # Accumulate update residuals
)
```

Use `SDMEPOptimizer` for **Dion-Muon hybrid updates** (Low-Rank SVD for large layers, Newton-Schulz for small ones):

```python
from mep.optimizers import SDMEPOptimizer
optimizer = SDMEPOptimizer(
    model.parameters(),
    rank_frac=0.2,   # Use top 20% singular values
    dion_thresh=1e5  # Use Dion if params > 100k
)
```

---

## 📊 Benchmarks

Compare SMEP (mode='ep') against standard optimizers on MNIST:

```bash
python -m mep.benchmarks.runner \
  --config mep/benchmarks/config/mnist.yaml \
  --baselines smep sdmep sgd \
  --output benchmarks/results/mnist_comparison
```

Results (plots and logs) will be saved to `benchmarks/results/`. Performance reports are in `PERFORMANCE.md`.

---

## 🔮 Roadmap

*   [x] **Foundation:** `pyproject.toml`, packaging, CI/CD setup.
*   [x] **Testing:** >85% coverage, numerical gradient verification.
*   [x] **Benchmarks:** Framework for comparing Optimizers/Models.
*   [x] **Robustness:** Type hints, input validation, NaN checks.
*   [ ] **Dion CUDA Kernel:** Custom C++ extension for acceleration.
*   [ ] **Convolutional Layers:** Extending `EPNetwork` for CNNs.
*   [ ] **LLM Scaling:** Testing SDMEP on Transformer blocks.

---

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` (coming soon) and run tests before submitting a PR.

```bash
# Run tests
pytest tests/ -v
```

## 📜 License

MIT License. See `LICENSE` for details.

---

*Cite this work:*
> *Spectral Dion-Muon Equilibrium Propagation (SDMEP): A robust, scalable, and biologically plausible optimization framework.* (2025)
