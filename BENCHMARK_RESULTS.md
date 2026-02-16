# Benchmark Results Analysis

## Performance Overview (MNIST Subset, 15s)

| Optimizer | Accuracy | vs SGD | vs Muon | Cost/Epoch | Cost/Step |
|:--- |:--- |:--- |:--- |:--- |:--- |
| **SGD** | **91.48%** | — | — | ~0.29s | ~1.1ms |
| **Adam** | **91.25%** | ~ | ~ | ~0.30s | ~1.2ms |
| **Muon** | **87.22%** | -4% | — | ~0.33s | ~1.3ms |
| **NaturalEP**| **85.76%**| -6% | -1.5% | ~0.82s | ~3.2ms |
| **LocalEP** | **73.58%** | -18%| -13% | ~0.83s | ~3.3ms |
| **SDMEP** | **71.87%** | -20%| -15% | ~0.78s | ~3.1ms |
| **EqProp** | **70.44%** | -21%| -17% | ~0.79s | ~3.1ms |
| **SMEP** | **68.98%** | -22%| -18% | ~0.73s | ~2.9ms |

## Key Findings

1.  **NaturalEPMuon is Competitive:**
    - NaturalEPMuon achieves **85.8% accuracy**, nearly matching standard Muon (87.2%) and approaching SGD/Adam (91%).
    - This is a remarkable result for a biologically plausible algorithm (no global backprop).
    - It outperforms other EP variants by ~12-15%.

2.  **Cost vs. Performance:**
    - EP variants are ~2.5x slower per step than backprop methods (3ms vs 1.2ms).
    - This cost comes from the settling phase (iterative inference).
    - However, `NaturalEPMuon` converges much faster *in terms of steps* than other EP methods, justifying the cost.

3.  **Scale Invariance is Critical:**
    - The breakthrough came from making EP updates scale-invariant (Muon-style orthogonalization without restoring norm).
    - Previously, vanishing gradients (magnitude 1e-11) caused EP to fail completely (<10% accuracy).

4.  **Local Learning Viability:**
    - `LocalEPMuon` achieves **73.6% accuracy**.
    - While lower than global methods, it demonstrates that strictly local learning rules (layer-wise updates) can learn useful representations on MNIST.

## Optimization Opportunities

1.  **Fisher Computation:**
    - `NaturalEPMuon` computes `F = g_free.T @ g_free` and solves a linear system.
    - For large layers, this is $O(d^3)$.
    - *Potential Fix:* Use diagonal approximation or block-diagonal Fisher to reduce cost.

2.  **Settling Efficiency:**
    - Settling takes the majority of time.
    - *Potential Fix:* Reduce `settle_steps` (currently 10) as training progresses, or use "warm start" from previous batch states more effectively.

3.  **Parallelism:**
    - `LocalEPMuon` updates are independent per layer.
    - *Potential Fix:* Parallelize layer updates on multi-GPU setups.
