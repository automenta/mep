# Benchmark Results Analysis

## Performance Overview (MNIST Subset, 15s, CPU)

| Optimizer | Accuracy | vs SGD | vs Muon | Cost/Epoch | Cost/Step |
|:--- |:--- |:--- |:--- |:--- |:--- |
| **SGD** | **92.12%** | — | — | ~1.14s | ~57ms |
| **Adam** | **92.00%** | ~ | ~ | ~1.05s | ~52ms |
| **Muon** | **88.61%** | -4% | — | ~1.51s | ~75ms |
| **SMEP** | **75.02%** | -17%| -13% | ~2.25s | ~112ms |
| **SDMEP** | **73.76%** | -18%| -15% | ~2.23s | ~111ms |
| **EqProp** | **68.81%** | -23%| -20% | ~2.16s | ~108ms |
| **NaturalEP**| **85.17%**| -7% | -3.5% | ~2.70s | ~135ms |
| **LocalEP** | **74.37%** | -18%| -14% | ~2.83s | ~141ms |

## Key Findings

1.  **Reduced Overhead relative to Backprop:**
    - The optimizations (manual settle loop, single backward pass) reduced the EP overhead significantly.
    - **SMEP** is now only **~1.96x slower** than SGD (down from ~2.5x previously).
    - This makes biologically plausible learning much more practical for experimentation.

2.  **NaturalEPMuon Efficiency:**
    - `NaturalEPMuon` is now **faster** than `LocalEPMuon` (~2.70s vs ~2.83s), reversing previous trends where it was the slowest.
    - This is due to the single backward pass optimization and efficient Fisher computation.
    - It maintains high accuracy (~85%) while being only ~2.36x slower than SGD.
    - Using the new `use_diagonal_fisher=True` option provides an additional ~5% speedup (~3.93s vs ~4.11s in isolated tests).

3.  **Local Learning Costs:**
    - `LocalEPMuon` is now the slowest variant (~2.83s), likely due to the overhead of per-layer backward passes (N passes vs 1 or 2 global passes).
    - However, it still offers the unique benefit of strictly local updates without global error signals.

4.  **Scale Invariance is Critical:**
    - As before, scale-invariant updates (Muon/Dion) are essential for convergence.
    - `EqProp` (Vanilla EP without Muon) performs significantly worse (~69% vs ~75% for SMEP).

## Optimization Opportunities

1.  **Parallelism for LocalEP:**
    - `LocalEPMuon` updates are independent per layer. Parallelizing these on multi-GPU setups could eliminate the per-layer overhead and make it faster than global methods.

2.  **Settling Efficiency:**
    - Settling remains the dominant cost (10 forward passes per step).
    - Techniques like "warm start" or reducing steps could further close the gap with SGD.

3.  **GPU Acceleration:**
    - Current benchmarks were run on CPU. On GPU, the matrix multiplication speedup might shift the balance again, potentially making `NaturalEP`'s Fisher computation relatively more expensive, but the diagonal approximation would mitigate this.
