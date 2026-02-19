# MEP: Muon Equilibrium Propagation

### 🧠 Biologically Plausible Deep Learning Without Backpropagation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/your-username/mep/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/mep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
import torch.nn as nn
from mep import smep, muon_backprop

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Option 1: EP mode (biologically plausible)
optimizer = smep(model.parameters(), model=model, mode='ep')
optimizer.step(x=x, target=y)  # No .backward() needed!

# Option 2: Backprop mode (drop-in replacement)
optimizer = muon_backprop(model.parameters())
loss.backward()
optimizer.step()
```

### Optimal EP Configuration

```python
from mep import smep

optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    beta=0.5,           # Nudging strength
    settle_steps=30,    # Settling iterations
    settle_lr=0.15,     # Settling learning rate
    loss_type='mse',    # Stable energy
    use_error_feedback=False,
)
```

---

## Performance Summary

| Benchmark | EP | SGD | Adam |
|-----------|-----|-----|------|
| MNIST (3 epoch) | **91.4%** | 91.0% | 90.2% |
| MNIST (10 epoch) | 95.37% | 93.80% | **95.75%** |
| XOR (100 step) | 100% | 100% | 100% |

**Key Findings:**
- ✅ EP achieves performance parity with backpropagation
- ⚠️ EP is ~2× slower (fundamental algorithmic cost)
- ⚠️ EP uses more memory than backprop+checkpointing
- ❌ Dropout incompatible with EP settling

📊 **Full results:** [docs/benchmarks/VALIDATION_RESULTS.md](docs/benchmarks/VALIDATION_RESULTS.md)

---

## When to Use EP

### ✅ Use EP For:
- Biological plausibility research
- Neuromorphic hardware deployment
- Energy-based model research
- Educational demonstrations
- Studying alternative learning mechanisms

### ✅ Use Backprop For:
- Standard classification/regression
- Production training pipelines
- Speed-critical applications
- Maximum accuracy goals

📋 **Detailed guidance:** [docs/research/ROADMAP_RESEARCH.md](docs/research/ROADMAP_RESEARCH.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/index.md](docs/index.md) | **Start here** - Full documentation index |
| [docs/benchmarks/PERFORMANCE_BASELINES.md](docs/benchmarks/PERFORMANCE_BASELINES.md) | Performance thresholds, optimal config |
| [docs/benchmarks/VALIDATION_RESULTS.md](docs/benchmarks/VALIDATION_RESULTS.md) | Full validation study |
| [docs/research/ROADMAP_RESEARCH.md](docs/research/ROADMAP_RESEARCH.md) | Research trajectory, partnerships |
| [docs/methods_paper.md](docs/methods_paper.md) | Preprint-ready methods paper |

---

## Examples

| Example | Description |
|---------|-------------|
| `examples/quickstart.py` | Minimal working example |
| `examples/demo_ep_vs_backprop.py` | EP vs backprop comparison |
| `examples/mnist_comparison.py` | MNIST classification demo |
| `examples/train_char_lm.py` | Character-level LM training |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run performance regression tests
pytest tests/regression/test_performance_baseline.py -v
```

---

## Contributing

Contributions welcome! See [docs/research/ROADMAP_RESEARCH.md](docs/research/ROADMAP_RESEARCH.md) for:
- Current research priorities
- Collaboration opportunities
- How to contribute

---

## Citation

```bibtex
@software{mep2026,
  title = {MEP: Muon Equilibrium Propagation},
  author = {MEP Contributors},
  year = {2026},
  url = {https://github.com/your-username/mep},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Last updated: 2026-02-18*
