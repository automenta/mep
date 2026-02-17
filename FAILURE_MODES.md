# MEP: Failure Modes and Best Practices Guide

This document provides guidance on troubleshooting common issues, understanding limitations, and achieving optimal results with Equilibrium Propagation optimizers.

---

## Quick Reference: When to Use Each Optimizer

| Use Case | Recommended Optimizer | Expected Performance |
|----------|----------------------|---------------------|
| Standard classification | **Adam** or **SGD** | Best accuracy |
| Muon orthogonalization research | **Muon** (backprop mode) | Comparable to SGD |
| Regression with EP | **SMEP** (MSE loss) | Good convergence |
| Classification with EP | **SMEP** (CrossEntropy) | ~10-15% below Adam |
| Large models (>100K params) | **SDMEP** | Faster but watch stability |
| Biological plausibility research | **LocalEPMuon** | Layer-local updates |
| Geometry-aware optimization | **NaturalEPMuon** | Fisher whitening |

---

## Common Failure Modes

### 1. EP Classification Gap

**Symptom:** EP optimizers achieve significantly lower accuracy (~10-15%) than backpropagation on classification tasks.

**Cause:** 
- EP's energy-based formulation with contrastive gradients doesn't perfectly match CrossEntropy geometry
- The β approximation introduces gradient estimation error
- Settling dynamics may not fully converge for deep networks

**Mitigation:**
```python
# Use smaller beta for more accurate gradients
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,
    mode='ep',
    beta=0.1,           # Smaller = more accurate but slower
    settle_steps=30,    # More steps for better convergence
    settle_lr=0.01,     # Lower settling learning rate
    loss_type='cross_entropy',
    softmax_temperature=0.5,  # Sharper softmax
    max_grad_norm=5.0   # Gradient clipping
)
```

**When to accept:** For research on biologically plausible learning, the gap is expected and acceptable. For production tasks, use backpropagation.

---

### 2. SDMEP Numerical Instability (NaN/Inf)

**Symptom:** Training produces NaN or Inf gradients/losses, especially with SDMEPOptimizer.

**Cause:**
- Low-rank SVD can be numerically unstable with ill-conditioned gradients
- Error feedback accumulation can explode
- Large learning rates combined with Dion updates

**Mitigation:**
```python
optimizer = SDMEPOptimizer(
    model.parameters(),
    model=model,
    mode='ep',
    lr=0.01,            # Lower learning rate
    beta=0.2,           # Moderate beta
    rank_frac=0.3,      # Higher rank = more stable
    dion_thresh=200000, # Only use Dion for very large matrices
    max_grad_norm=5.0,  # Aggressive gradient clipping
    use_error_feedback=True,
    error_beta=0.8      # Lower error decay
)
```

**Fallback:** If instability persists, switch to SMEPOptimizer (pure Muon, no Dion).

---

### 3. Settling Divergence

**Symptom:** Energy increases during settling phase instead of decreasing. Error message: "Energy diverged at settling step X".

**Cause:**
- Settling learning rate too high
- Beta too large for the network architecture
- ReLU activations create non-smooth energy landscape

**Mitigation:**
```python
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,
    mode='ep',
    settle_lr=0.005,    # Reduce settling learning rate
    settle_steps=40,    # More steps with smaller steps
    beta=0.1,           # Smaller nudge
    spectral_timing='during_settling',  # Add spectral penalty
    spectral_lambda=0.5
)
```

**Architecture fix:** Use smoother activations like GELU or SiLU instead of ReLU:
```python
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.GELU(),  # Instead of ReLU
    nn.Linear(512, 10)
)
```

---

### 4. Slow Training (EP Overhead)

**Symptom:** EP training is 1.3-2x slower than backpropagation per epoch.

**Cause:**
- Settling iterations add computational overhead
- Multiple forward passes per batch (free + nudged phase)

**Mitigation:**
```python
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,
    mode='ep',
    settle_steps=10,    # Fewer settling steps
    beta=0.3,           # Larger beta compensates for fewer steps
    settle_lr=0.03      # Higher settling learning rate
)
```

**Trade-off:** Fewer settling steps = faster but less accurate gradients. Monitor validation accuracy to find the sweet spot.

---

### 5. Spectral Constraint Not Enforced

**Symptom:** Spectral norm exceeds gamma threshold during training.

**Cause:**
- Spectral timing set to 'post_update' but constraint checked at wrong time
- Power iteration not converging fast enough

**Mitigation:**
```python
optimizer = SMEPOptimizer(
    model.parameters(),
    model=model,
    use_spectral_constraint=True,
    gamma=0.95,
    spectral_timing='both',  # Enforce during settling AND post-update
    spectral_lambda=1.0      # Stronger penalty
)
```

---

### 6. No Activation Captured During Settling

**Symptom:** RuntimeError: "No activations captured during settling. Expected X layer(s) but got 0."

**Cause:**
- Model doesn't contain recognized layer types (nn.Linear, nn.Conv2d)
- Custom layers not in the structure inspection list
- Model uses functional API instead of nn.Module layers

**Mitigation:**
```python
# Ensure model uses standard PyTorch layers
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # Use nn.Module, not functional
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # Functional is OK for activations
        x = self.fc(x)
        return x

# Or wrap custom layers
optimizer = SMEPOptimizer(model.parameters(), model=model, mode='ep')
```

---

## Hyperparameter Tuning Guide

### Learning Rate (lr)

| Optimizer | Recommended Range | Notes |
|-----------|------------------|-------|
| SGD/Adam | 0.001 - 0.1 | Standard deep learning ranges |
| SMEP/SDMEP | 0.01 - 0.05 | Lower than backprop |
| LocalEPMuon | 0.005 - 0.02 | Even lower due to local updates |

**Tuning strategy:** Start with 0.02, reduce by 2x if unstable, increase by 1.5x if too slow.

---

### Beta (β) - Nudging Strength

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.05 - 0.1 | Very accurate gradients, slow | Numerical validation |
| 0.2 - 0.3 | Balanced | Default recommendation |
| 0.4 - 0.5 | Faster but approximate | Quick experiments |

**Rule of thumb:** β should be large enough to produce measurable contrast but small enough for linear approximation to hold.

---

### Settling Steps

| Steps | Accuracy | Speed | Recommendation |
|-------|----------|-------|----------------|
| 5-10 | Low | Fast | Quick prototyping |
| 15-25 | Medium | Moderate | Default |
| 30-50 | High | Slow | Final experiments, validation |

**Early stopping:** Monitor energy during settling. If energy plateaus before max steps, reduce steps.

---

### Settling Learning Rate (settle_lr)

| Value | Effect |
|-------|--------|
| 0.01 - 0.02 | Conservative, stable |
| 0.03 - 0.05 | Default |
| 0.05 - 0.1 | Aggressive, may diverge |

**Tuning:** Set settle_lr ≈ 2 × optimizer lr as starting point.

---

### Newton-Schulz Steps (ns_steps)

| Steps | Orthogonality | Speed |
|-------|---------------|-------|
| 3 | Approximate | Fast |
| 5 | Good | Default |
| 7-10 | Excellent | Slow |

**Note:** More than 5 steps usually provides diminishing returns.

---

### Rank Fraction (rank_frac, SDMEP only)

| Value | Compression | Stability | Use Case |
|-------|-------------|-----------|----------|
| 0.1 | High | Lower | Very large models |
| 0.2 | Medium | Good | Default |
| 0.3-0.5 | Low | High | Stability-critical |

---

## Architecture Recommendations

### Good Architectures for EP

```python
# Shallow MLP (recommended)
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.GELU(),
    nn.Linear(512, 10)
)

# CNN with moderate depth
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.GELU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.GELU(),
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(512, 10)
)
```

### Architectures to Avoid

```python
# Very deep networks (>10 layers) - settling may not converge
# Residual connections - EP doesn't handle skip connections well (yet)
# BatchNorm - Can interfere with energy-based settling
# Very wide layers (>4096 units) - Use SDMEP with appropriate dion_thresh
```

---

## Debugging Checklist

When EP training fails, check these in order:

1. **Gradient Check:** Run `test_numerical_gradients.py` to verify EP gradients match finite differences
2. **Energy Monitoring:** Check that energy decreases during settling
3. **Spectral Norm:** Verify σ(W) ≤ γ throughout training
4. **Learning Rate:** Try reducing by 2× or 5×
5. **Beta:** Try smaller β (0.1) for more accurate gradients
6. **Settling:** Increase settle_steps or reduce settle_lr
7. **Architecture:** Simplify model (fewer layers, smoother activations)
8. **Fallback:** Switch to backprop mode to isolate EP-specific issues

---

## Performance Expectations

### MNIST MLP (784→512→256→10)

| Optimizer | Epochs | Test Accuracy | Time/Epoch |
|-----------|--------|---------------|------------|
| Adam | 10 | 97-98% | ~3s |
| SGD | 10 | 95-97% | ~3s |
| SMEP | 10 | 85-90% | ~4-5s |
| SDMEP | 10 | 80-88% | ~4-5s |

### CIFAR-10 CNN

| Optimizer | Epochs | Test Accuracy | Time/Epoch |
|-----------|--------|---------------|------------|
| Adam | 20 | 75-80% | ~10s |
| SGD | 20 | 70-75% | ~10s |
| SMEP | 20 | 55-65% | ~15s |

**Note:** EP performance gap is an active research area. These numbers represent current implementation capabilities.

---

## Research Opportunities

### Open Problems

1. **Better Classification Energy:** Current KL divergence formulation is approximate. Research needed on energy functions that better match CrossEntropy geometry.

2. **Deep Network Scaling:** EP settling becomes unstable for networks >10 layers. Potential solutions:
   - Layer-wise settling
   - Hierarchical energy functions
   - Better initialization schemes

3. **Residual Connections:** EP doesn't naturally handle skip connections. Research needed on energy formulation for ResNets.

4. **BatchNorm Integration:** BatchNorm interferes with EP settling. Alternatives:
   - LayerNorm (works better with EP)
   - No normalization with careful initialization
   - EP-aware normalization

### Recommended Starting Points

For researchers new to EP:

1. Start with **shallow MLPs** on MNIST/Fashion-MNIST
2. Use **SMEPOptimizer** (not SDMEP) initially
3. Run **numerical gradient validation** to confirm correct implementation
4. Compare against **backprop baseline** with same architecture
5. Experiment with **beta and settle_steps** to understand trade-offs

---

## Getting Help

### Validation Tests

Run the test suite to verify your installation:
```bash
pytest tests/integration/test_numerical_gradients.py -v
pytest tests/unit/test_conv2d.py -v
```

### Benchmark Reproduction

Reproduce benchmark results:
```bash
python -m mep.benchmarks.runner \
  --config mep/benchmarks/config/mnist.yaml \
  --epochs 5 \
  --output benchmarks/results/my_run
```

### Reporting Issues

When reporting bugs, include:
1. Full error message and traceback
2. Model architecture
3. Optimizer configuration
4. Dataset and batch size
5. Steps to reproduce

---

## References

1. Scellier & Bengio (2017). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. *Frontiers in Computational Neuroscience*.

2. Jordan (2024). The Muon Optimizer. *GitHub Repository*.

3. Miyato et al. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.

4. Scellier et al. (2024). Energy-Based Learning in Continuous Time. *arXiv preprint*.
