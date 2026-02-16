# MEP Development Plan: From Prototype to Research Framework

**Version:** 0.2.0  
**Status:** Approved  
**Execution Mode:** Sequential (Phase 1 → 2 → 3 → 4)

---

## Executive Summary

Transform MEP from a research prototype into a robust, well-tested framework with rigorous academic/industrial-scale evaluation capabilities. This plan addresses all critical gaps identified in the codebase analysis.

**Core Goals:**
1. **Optimizer Robustness:** Ensure SDMEP/SMEP are functionally correct, numerically stable, and well-tested
2. **Research Infrastructure:** Build comprehensive evaluation framework with academic rigor
3. **Baseline Comparisons:** Include EqProp variants and Muon standalone optimizer
4. **Communication:** Enhance README.md to inspire further research and demonstrate applicability

**Timeline:** 11-15 days (Phases 1-4)

---

## Phase 1: Foundation & Packaging (1-2 days)

**Goal:** Make MEP installable, reproducible, and properly configured.

### Deliverables

#### [NEW] `pyproject.toml`

Complete package configuration:

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mep"
version = "0.2.0"
description = "Muon Equilibrium Propagation: Biologically plausible deep learning"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["deep-learning", "equilibrium-propagation", "biologically-plausible", "energy-based-models"]

dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "flake8>=6.0.0",
]
research = [
    "wandb>=0.15.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "pandas>=1.5.0",
    "scipy>=1.9.0",
    "tqdm>=4.65.0",
]
all = ["mep[dev,research]"]

[project.scripts]
mep-benchmark = "mep.cli:benchmark"

[tool.black]
line-length = 100
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --cov=mep --cov-report=html --cov-report=term"
```

#### [NEW] `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
```

#### [MODIFY] `.gitignore`

Expand to include testing, research artifacts, and build directories.

### Verification

```bash
cd /home/me/mep
pip install -e ".[dev,research]"
python -c "import mep; print(mep.__version__)"
```

---

## Phase 2: Testing Infrastructure (3-4 days)

**Goal:** Achieve >85% test coverage with comprehensive validation.

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_energy.py             # Energy function correctness
│   ├── test_settling.py           # Settling dynamics
│   ├── test_optimizers.py         # Optimizer mechanics
│   └── test_spectral.py           # Spectral constraint enforcement
├── integration/
│   ├── test_training_loop.py      # End-to-end training
│   ├── test_convergence.py        # Training convergence
│   └── test_numerical_gradients.py # EP vs finite differences (CRITICAL)
└── regression/
    └── test_benchmarks.py         # Reproduce README results
```

### Critical Tests

**1. Numerical Gradient Validation** (`test_numerical_gradients.py`)
- Compare EP gradients against finite differences
- Must pass with <10% relative error
- This validates the entire EP implementation

**2. Energy Monotonicity** (`test_energy.py`)
- Energy must decrease during free settling
- Validates settling dynamics correctness

**3. Spectral Constraint** (`test_spectral.py`)
- Spectral norm must stay ≤ γ throughout training
- Validates SDMEP's core stability guarantee

**4. Training Convergence** (`test_convergence.py`)
- Model must learn on synthetic data (e.g., XOR)
- End-to-end validation

**5. Benchmark Reproduction** (`test_benchmarks.py`)
- Reproduce README results (MNIST, 3 epochs, ~66% accuracy)
- Regression testing

### Verification

```bash
pytest tests/ -v --cov=mep --cov-report=html
# Expected: >85% coverage, all tests pass
```

---

## Phase 3: Research Evaluation Framework (4-5 days)

**Goal:** Build comprehensive benchmarking with academic rigor and baseline comparisons.

### Architecture

```
mep/
├── benchmarks/
│   ├── config/
│   │   ├── base.yaml
│   │   ├── mnist.yaml
│   │   ├── cifar10.yaml
│   │   └── fashion_mnist.yaml
│   ├── baselines.py              # Baseline implementations
│   ├── runner.py                 # Experiment orchestration
│   ├── metrics.py                # Tracking & logging
│   └── visualization.py          # Plotting utilities
└── experiments/
    └── [timestamp]/              # Auto-generated run directories
```

### Baseline Implementations

**Required Baselines:**

1. **Standard Backprop Optimizers:**
   - SGD with momentum
   - Adam
   - AdamW

2. **Pure EqProp:**
   - Vanilla EP without spectral constraints or Muon
   - Shows value of SDMEP safety harness

3. **Muon (Standalone):**
   - Newton-Schulz orthogonalization only
   - No spectral constraints (SMEP variant)

4. **SMEP (Muon + EP):**
   - Muon optimizer with EP gradients
   - Soft spectral constraints

5. **SDMEP (Full):**
   - Muon/Dion hybrid + spectral constraint + error feedback
   - Complete safety harness

### Metrics Tracking

Track per epoch:
- Train/validation loss and accuracy
- Spectral norm (for EP variants)
- Energy (free and nudged phases)
- Settling steps taken
- Gradient norms
- Epoch time
- NaN/Inf detection

### Visualization

Automated plots:
- Training curves (loss, accuracy)
- Spectral norm evolution
- Energy landscape
- Optimizer comparison (side-by-side)
- Convergence speed analysis

### Configuration Management

YAML-based configs for reproducibility:

```yaml
# mnist.yaml
dataset: MNIST
model: MLP
architecture:
  dims: [784, 1000, 10]
optimizer: SDMEP
learning_rate: 0.02
beta: 0.5
epochs: 10
seed: 42
```

### WandB Integration

Optional tracking:
```bash
mep-benchmark --config mnist.yaml --wandb --project mep-research
```

### Verification

```bash
# Run full benchmark suite
python -m mep.benchmarks.runner \
  --config mep/benchmarks/config/mnist.yaml \
  --baselines sgd adam adamw eqprop muon smep sdmep \
  --output experiments/mnist_comparison
```

**Expected output:**
- Comparison plots showing SDMEP stability vs baselines
- Metrics demonstrating spectral constraint enforcement
- Evidence of Muon's benefit for EP training

---

## Phase 4: Optimizer Robustness & API Improvements (3-4 days)

**Goal:** Production-ready optimizers with validation, error handling, and clean APIs.

### Code Enhancements

#### Type Annotations

Add type hints throughout:
```python
from typing import List, Optional, Tuple, Dict
import torch

def settle(
    self,
    x: torch.Tensor,
    y_target: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    steps: int = 15,
    early_stopping: bool = False,
    sparsity: float = 0.0
) -> List[torch.Tensor]:
    ...
```

#### Input Validation

```python
def settle(self, x, y_target=None, beta=0.0, ...):
    if x.ndim < 2:
        raise ValueError(f"Input must be at least 2D, got shape {x.shape}")
    if not 0 <= beta <= 1:
        raise ValueError(f"Beta must be in [0, 1], got {beta}")
    ...
```

#### Numerical Stability

```python
for t in range(steps):
    E = self.energy(x, self.states, y_target, beta)
    
    if torch.isnan(E) or torch.isinf(E):
        raise RuntimeError(
            f"Energy diverged at step {t}: E={E.item():.4f}. "
            f"Try reducing learning rate or beta."
        )
    ...
```

#### API Improvements

**Breaking changes (v0.2.0):**
- Rename `params_lazy` → `early_stopping`
- `compute_ep_gradients()` returns diagnostics dict
- `activation` parameter in `EPNetwork.__init__()`

**Backward compatibility:**
- Keep old parameter names with deprecation warnings for one version

### Verification

```bash
# Type checking
mypy mep/

# Linting
black mep/ tests/
flake8 mep/ tests/

# Tests still pass
pytest tests/ -v
```

---

## Phase 5: README Enhancement (2-3 days)

**Goal:** Transform README into an inspiring blog-post/pre-print that communicates MEP's potential.

### Enhanced Structure

```markdown
# MEP: Muon Equilibrium Propagation
### Biologically Plausible Deep Learning Without Backpropagation

## Abstract
[250-word summary suitable for a pre-print]

## Introduction: The Backpropagation Bottleneck
[Problem statement with neuromorphic computing motivation]

## The MEP Framework
### Theoretical Foundation
[Energy-based learning, contraction mappings, stability guarantees]

### The Safety Harness: Spectral-Dion-Muon
[S, D, M components with mathematical intuition]

## Implementation
[Architecture diagrams, code examples]

## Experimental Validation
### Benchmarks
[Tables comparing MEP vs baselines with plots]

### Ablation Studies
[EqProp → SMEP → SDMEP progression showing each component's value]

## Applications & Future Directions
### Neuromorphic Hardware
[Energy efficiency, O(1) memory, analog computing]

### Continual Learning
[Error feedback mechanism]

### Scaling to Transformers
[Preliminary results or research roadmap]

## How to Use MEP
[Installation, quickstart, examples]

## Contributing & Research Opportunities
[Open problems, collaboration invitation]

## Citation
[BibTeX entry]

## References
[Key papers: EP, Muon, Spectral Normalization]
```

### Visual Enhancements

**Add diagrams:**
- Architecture diagram (EP network structure)
- Energy landscape visualization
- Comparison plots from benchmarks
- Spectral norm evolution

**Add tables:**
- Benchmark results (MEP vs baselines)
- Computational complexity comparison
- Memory usage comparison

### Key Messages

1. **Innovation:** New stability guarantees for EP through spectral constraints
2. **Validation:** Rigorous testing and numerical verification
3. **Applicability:** Path to neuromorphic deployment and continual learning
4. **Transparency:** Honest performance trade-offs with future optimization path

### Verification

- README renders correctly on GitHub
- Links work
- Images/plots display properly
- Code examples run without errors

---

## Implementation Timeline

| Phase | Duration | Start After | Deliverables |
|-------|----------|-------------|--------------|
| **Phase 1** | 1-2 days | Immediate | Package installable, dependencies managed |
| **Phase 2** | 3-4 days | Phase 1 complete | 30+ tests, >85% coverage, numerical validation |
| **Phase 3** | 4-5 days | Phase 2 complete | Benchmark suite, baselines (EqProp, Muon, etc.) |
| **Phase 4** | 3-4 days | Phase 3 complete | Type hints, validation, API improvements |
| **Phase 5** | 2-3 days | Phase 4 complete | Enhanced README as blog-post/pre-print |

**Total Duration:** 13-18 days

---

## Success Criteria

### Phase 1 ✓
- [ ] Package installable via `pip install -e .`
- [ ] All dependencies specified
- [ ] Can import `mep` module

### Phase 2 ✓
- [ ] >85% test coverage
- [ ] Numerical gradient validation passes (<10% error)
- [ ] Energy monotonicity verified
- [ ] Spectral constraint enforcement verified
- [ ] README benchmark reproduced in regression tests

### Phase 3 ✓
- [ ] 5 baseline implementations (SGD, Adam, AdamW, EqProp, Muon)
- [ ] 3 EP variants (EqProp, SMEP, SDMEP)
- [ ] Automated metrics tracking
- [ ] Visualization suite functional
- [ ] WandB integration working
- [ ] Reproducible experiments (seeded, logged)

### Phase 4 ✓
- [ ] All public functions type-annotated
- [ ] Input validation on all user-facing APIs
- [ ] NaN/Inf detection in training loops
- [ ] Meaningful error messages
- [ ] API documentation (docstrings)

### Phase 5 ✓
- [ ] README includes abstract suitable for pre-print
- [ ] Benchmark plots embedded
- [ ] Architecture diagrams added
- [ ] Clear installation and usage instructions
- [ ] Research directions outlined
- [ ] Citation information included

---

## Future Work (Post-MVP)

### Performance Optimization
1. **Dion CUDA Kernel:** Replace Python SVD with iterative QR (5-10× speedup)
2. **JIT Compilation:** torch.jit.script for settling loops
3. **Mixed Precision:** FP16/BF16 training support

### Research Extensions
1. **Convolutional Support:** Native Conv2d layers, CIFAR-10 benchmarks
2. **Transformer Variant:** Causal attention with EP
3. **Continual Learning:** Leverage error feedback for task-incremental learning
4. **Neuromorphic Deployment:** SNN conversion, hardware profiling

### Infrastructure
1. **CLI Tool:** `mep train`, `mep eval` commands
2. **Model Zoo:** Pre-trained checkpoints
3. **Interactive Dashboard:** Real-time monitoring

---

## Getting Started (After Phase 1)

**Install:**
```bash
git clone https://github.com/yourusername/mep
cd mep
pip install -e ".[dev,research]"
```

**Run Tests:**
```bash
pytest tests/ -v --cov=mep
```

**Run Benchmark:**
```bash
python examples/benchmark_mnist.py --optimizer SDMEP --epochs 5
```

**Comprehensive Evaluation:**
```bash
mep-benchmark --config mep/benchmarks/config/mnist.yaml --wandb
```

---

## Notes

- **Sequential Execution:** Complete each phase before starting the next
- **Breaking Changes:** Version 0.2.0 signals API changes
- **Testing First:** Numerical validation is critical before research framework
- **Documentation:** README enhancement happens last, showcasing validated results

---

**Plan Status:** ✅ Approved  
**Next Step:** Begin Phase 1 implementation
