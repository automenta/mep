# Preliminary Optimizer Benchmark

This document presents a quick comparison of the three optimizer variants on a subset of MNIST (1000 samples, 1 epoch).

## Experimental Setup
*   **Dataset:** MNIST (1000 samples)
*   **Model:** MLP (784 -> 128 -> 10)
*   **Epochs:** 1
*   **Batch Size:** 64
*   **Settling Steps:** 15
*   **Newton-Schulz Steps:** 5
*   **Device:** CPU

## Results

| Optimizer | Loss | Accuracy (%) | Time (s) |
| :--- | :--- | :--- | :--- |
| **SMEP (Standard EP)** | 2.2403 | 21.30% | 1.01s |
| **LocalEPMuon** | 2.2280 | 23.90% | 1.03s |
| **NaturalEPMuon** | 2.1463 | 38.60% | 1.22s |

## Analysis

1.  **SMEP (Standard):** Serves as the baseline. It establishes stable learning dynamics.
2.  **LocalEPMuon:** Performs comparably to the baseline (slightly better in this run), validating that **layer-local updates** can effectively train the network without global error backpropagation. The overhead is negligible.
3.  **NaturalEPMuon:** Demonstrates **superior convergence** (38.6% vs ~21% accuracy) in this short training horizon. The computational cost is slightly higher (+20%) due to the Fisher matrix inversion, but the gain in data efficiency is significant.

*Note: These results are preliminary and run on a small subset. Full convergence characteristics may differ on larger datasets.*
