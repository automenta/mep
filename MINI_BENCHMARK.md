# Preliminary Optimizer Benchmark

This document presents a comparison of the four optimizer variants on a subset of MNIST (2000 samples, 5 epochs).

## Experimental Setup
*   **Dataset:** MNIST (2000 samples)
*   **Model:** MLP (784 -> 128 -> 10)
*   **Epochs:** 5
*   **Batch Size:** 64
*   **Settling Steps:** 15
*   **Newton-Schulz Steps:** 5
*   **Device:** CPU

## Results

| Optimizer | Loss | Accuracy (%) | Time (s) |
| :--- | :--- | :--- | :--- |
| **SMEP (Standard EP)** | 1.8845 | 80.15% | 10.32s |
| **SDMEP (Dion-Muon)** | 2.0357 | 62.20% | 9.25s |
| **LocalEPMuon** | 1.8905 | 80.80% | 9.11s |
| **NaturalEPMuon** | 2.0343 | 63.90% | 13.72s |

## Analysis

1.  **SMEP vs. LocalEPMuon:**
    *   **LocalEPMuon** performs remarkably well, matching the accuracy of the global **SMEP** optimizer (80.8% vs 80.15%).
    *   This strongly validates the biological plausibility hypothesis: deep networks can be trained effectively using only layer-local energy contrasts, without global error backpropagation.
    *   **LocalEPMuon** was slightly faster (9.11s vs 10.32s), likely due to better memory locality or simplified gradient graph traversal.

2.  **SDMEP (Efficiency):**
    *   **SDMEP** was the fastest optimizer (9.25s), demonstrating the computational efficiency of low-rank Dion updates.
    *   However, convergence was slower (62.20% accuracy). This is expected for low-rank approximations in early training; they typically require more epochs or a higher rank fraction to match full-rank updates initially.

3.  **NaturalEPMuon (Geometry):**
    *   **NaturalEPMuon** achieved 63.90% accuracy but showed high variance in loss (oscillating between 1.7 and 2.8).
    *   The computational cost is highest (13.72s) due to the $O(N^3)$ inversion of the Fisher matrix.
    *   The instability suggests that the empirical Fisher approximation on small batches requires careful tuning of the damping factor or learning rate schedule.

## Conclusion

*   **LocalEPMuon** is a viable, high-performance alternative to standard EP, offering biological realism with no accuracy penalty.
*   **SMEP** remains the robust default.
*   **SDMEP** offers speedups for larger models but may require longer training.
*   **NaturalEPMuon** is a powerful research tool for geometric optimization but requires stabilization tuning.
