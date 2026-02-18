# MEP Development Roadmap

## Strategic Priorities for Maximizing Impact

**Status:** TODO.md complete. Core functionality implemented and tested.

**Current Focus:** 
- Validate theoretical advantages (O(1) memory, continual learning) with proper experimental design
- Document EP's qualitative differences from backpropagation
- Identify research niches where EP's unique properties matter

**Note on Validation Studies:** Initial measurements had methodological limitations. Proper validation requires:
- Memory: Gradient checkpointing at extreme depths (1000+ layers), measuring activations only
- Continual learning: Sequential training on single model with proper forgetting metrics
- Both outcomes (confirmation or refutation) are valuable research contributions

---

## ✅ What's Done (TODO.md Complete)

| Item | Status | Notes |
|------|--------|-------|
| Comprehensive tests | ✅ | 142 unit tests, 85% coverage |
| Numerical gradient verification | ✅ | EP vs BP comparison tests |
| Type hints & mypy | ✅ | `mypy mep/optimizers/` passes |
| Input validation & NaN guards | ✅ | Robust error handling |
| Adaptive settling | ✅ | Early stopping implemented |
| Convolutional layers | ✅ | Conv2d, BatchNorm2d support |
| Mixed-precision (AMP) | ✅ | Implemented and tested |
| torch.compile | ✅ | Compatible and tested |
| NaturalGradient Fisher | ✅ | Working implementation |
| Transformer/attention support | ✅ | LayerNorm, residuals, MHA support |
| Dion CUDA kernels | ✅ | Fused settling kernel added |
| Benchmark suite | ✅ | `mep/benchmarks/` complete |
| Working examples | ✅ | `quickstart.py`, `mnist_comparison.py`, `train_char_lm.py` |
| Error feedback fix | ✅ | Works correctly with Dion, disabled for Muon |

**Functional Status:** EP trains classification models with proper gradient flow through all layers.

**Note:** Transformer architecture support exists, but full transformer-based LM training (e.g., nanoGPT-style) remains to be validated as part of the Character LM goal.

---

## 🎯 Immediate Priorities (Week 1-2)

### 1. Proper Memory Validation [HIGH PRIORITY]

**Why:** Initial measurement was methodologically flawed. Need to properly validate O(1) claim using bioplausible's approach.

**What:**
```python
# Proper validation:
# - Use gradient checkpointing to isolate activation memory
# - Test at extreme depths (1000, 2000, 5000+ layers)
# - Compare activation memory only, not total memory
# - Follow bioplausible Track 35 methodology
```

**Success criteria:**
- [ ] Replicate bioplausible's 19.4× savings at depth 100 (or document why different)
- [ ] Show activation memory scales O(1) for EP vs O(depth) for backprop
- [ ] Identify crossover point where EP's advantage becomes significant

**Effort:** 3-5 days

---

### 2. Proper Continual Learning Benchmark [HIGH PRIORITY]

**Why:** Preliminary test was invalid (reinitialized model per task = not continual learning).

**What:**
```python
# Correct CL design:
model = initialize_once()
for task in tasks:
    train_on_task(model, task)  # Same model, sequential
    for prev_task in tasks[:current]:
        measure_accuracy(model, prev_task)  # Measure forgetting
```

**Success criteria:**
- [ ] Single model trained sequentially on 5+ tasks
- [ ] Proper forgetting metric (accuracy drop on previous tasks)
- [ ] Compare EP+EF vs EP-without-EF vs backprop vs EWC

**Effort:** 3-5 days

---

### 3. Character LM Example [VALIDATION]

**Why:** Demonstrates EP works on sequential prediction tasks, not just classification.

**What:**
```bash
python examples/train_char_lm.py
# Compare EP vs backprop on Shakespeare character LM
# Document qualitative differences in learning dynamics
```

**Success criteria:**
- [ ] Script runs without errors
- [ ] Generated text shows coherent learning
- [ ] Document how EP's dynamics differ from BPTT

**Effort:** 1-2 days

---

### 4. Document EP's Qualitative Value [IMPORTANT]

**Why:** EP's value isn't just "beating backprop on benchmarks." It's about:
- Biological plausibility (no weight transport problem)
- Energy-based formulation (different theoretical framework)
- Neuromorphic compatibility (natural fit for analog hardware)
- Research tool (study alternative learning mechanisms)

**What:** Update README with honest, constructive positioning:
- When EP is useful (research, neuromorphic, memory-constrained)
- When backprop is better (production, speed-critical, standard tasks)
- EP's unique characteristics and research value

**Effort:** 1-2 days

---

## 🚀 Medium-Term Priorities (Month 1-2) - CONTINGENT

*These priorities depend on the outcome of Immediate Priorities 1-3.*

### 4. Deep Network Scaling Study [IF memory shows promise]

**Why:** Initial memory test showed 0% savings, but activation-only measurement may tell different story.

**What:**
- Measure activation memory only (exclude weights)
- Use gradient checkpointing baseline for fair comparison
- Test at extreme depths (1000+ layers)

**Success criteria:**
- [ ] EP shows >30% activation memory savings
- [ ] Savings increase with depth

**If negative:** Memory advantage claim should be abandoned.

**Effort:** 1 week

---

### 5. Add Proper Baselines to Benchmarks [ALWAYS NEEDED]

**Why:** Current benchmarks compare EP to... itself. Need external baselines.

**What:**
- SGD, Adam (standard optimization baselines)
- EWC, GEM (continual learning baselines)
- Muon-only (ablate the EP component)

**Success criteria:**
- [ ] All benchmarks include at least 2 external baselines
- [ ] Results show where EP wins and loses

**Effort:** 1 week

---

### 6. Publish Results (Honest Assessment) [ALWAYS NEEDED]

**Why:** The community needs honest information about EP's tradeoffs.

**What:**
- Technical report with all findings (positive and negative)
- Blog post explaining when EP is/isn't appropriate
- Open discussion with EP research community

**Success criteria:**
- [ ] Honest, reproducible results published
- [ ] Clear guidance for practitioners
- [ ] Community feedback incorporated

**Effort:** 2-4 weeks

---

## 🏗️ Long-Term Priorities (Month 3+) - CONTINGENT

*These priorities depend on validation study outcomes.*

### 7. Find Domains Where EP Excels [RESEARCH FOCUS]

**Goal:** Identify practical niches where EP's unique properties provide advantages.

**Candidate domains to investigate:**
| Domain | Hypothesis | Validation Status |
|--------|------------|-------------------|
| Continual learning | EP+EF reduces forgetting | Needs proper benchmark |
| Memory-constrained | O(1) activation storage enables deeper nets | Needs proper validation |
| Edge devices | Lower memory = fits on-device | Untested |
| Privacy-preserving | No stored activations | Theoretical only |
| Neuromorphic hardware | Local rules match analog substrates | Requires partnership |

**Success criteria:**
- [ ] One or more domains with clear, reproducible advantage
- [ ] Published results or demo

**If no quantitative advantage found:** EP still provides value as research/educational tool for studying alternative learning mechanisms.

**Effort:** 2-3 months

---

### 8. PyTorch Lightning Integration [LOW PRIORITY]

**Status:** Not implemented. Low priority until core value is proven.

**Rationale:** Only worth the effort if MEP gains adoption.

**Effort:** 2-3 days (if/when needed)

---

### 9. Advanced CUDA Optimization [ON HOLD]

**Status:** Fused settling kernel implemented. Further optimization deferred.

**Rationale:** Don't optimize until we know EP has a use case worth optimizing for.

**Effort:** 4-8 weeks (if/when needed)

---

## 📊 Effort vs. Impact Matrix

```
Impact
  ▲
  │    ┌─────────────────┐    ┌──────────────┐
  │    │ 1. Memory       │    │ 7. Domain    │
  │    │    Validation   │    │    Where EP  │
  │    │    (proper)     │    │    Wins      │
  │    │ 2. CL Benchmark │    │ 8. Lightning │
  │    │    (proper)     │    │    (low pri) │
  │    │ 5. Deep Scaling │    │ 9. CUDA Opt  │
  │    └─────────────────┘    │    (on hold) │
  │    ┌─────────────┐        └──────────────┘
  │    │ 3. Char LM  │
  │    │ 4. Qual     │
  │    │ 6. Baselines│
  │    │ 7. Publish  │
  │    └─────────────┘
  │
  └──────────────────────────────────────► Effort
       Low                    High

Note: Priorities are research-focused.
      Validation studies may confirm or refute claims.
      Both outcomes are valuable contributions.
```

---

## 📈 Success Metrics

| Metric | Current | Target (3mo) | Target (12mo) |
|--------|---------|--------------|---------------|
| Working examples | 5 | 8 | 15+ |
| Documented use cases | 1 (classification) | 3 (CL, LM, memory) | 5+ |
| External contributors | 0 | 1+ | 5+ |
| GitHub Stars | ~0 | 100+ | 500+ |
| Citations | 0 | 2+ (methods paper) | 20+ |
| EP Speed (vs backprop) | ~1.5× slower | ~1.5× slower | ~1.2× slower |
| Memory validation | Inconclusive | Proper validation study | Quantified results |
| CL validation | Invalid test | Proper benchmark | Competitive results |

---

## 🔍 Research Questions

1. **Does EP's O(1) activation memory enable training deeper networks?** ← Needs proper validation
2. **Does EP+EF reduce catastrophic forgetting?** ← Needs proper CL benchmark
3. **What qualitative differences does EP exhibit?** ← Document learning dynamics
4. **When is EP preferable to backprop?** ← Identify niches
5. **Does biological plausibility matter practically?** ← Long-term question

---

## 🤝 Collaboration Opportunities

| Domain | Potential Partners | Priority | Status |
|--------|-------------------|----------|--------|
| Continual Learning | CL research groups | High | Need proper benchmark |
| Memory-Efficient DL | Systems/ML groups | High | Compare with gradient checkpointing |
| Neuromorphic Hardware | Intel Labs, SpiNNaker | Medium | After software validation |
| Energy-Based Models | Yann LeCun's group | Medium | Theoretical alignment |
| ML Education | Universities | Medium | As teaching tool |

---

## 📝 Action Items

### This Week
- [ ] Design proper memory validation (gradient checkpointing, 1000+ layers)
- [ ] Implement proper CL benchmark (sequential, forgetting metric)
- [ ] Run `examples/train_char_lm.py`
- [ ] Update README with qualitative value documentation

### This Month
- [ ] Complete memory validation study
- [ ] Complete CL benchmark with baselines
- [ ] Document EP's qualitative differences
- [ ] Community feedback on findings

### This Quarter
- [ ] Publish validation results (positive or negative)
- [ ] Identify EP's niches (if any)
- [ ] Build community around research use cases

---

## 💡 Final Thought

**The goal is not to replace backpropagation.** Backprop works exceptionally well for standard deep learning.

**The goal IS to:**
1. Enable biologically plausible learning research
2. Provide tools for studying alternative learning mechanisms
3. Explore niches where EP's unique properties matter (neuromorphic, memory-constrained, continual learning)
4. Demonstrate that effective deep learning doesn't require backpropagation

**What We've Built:** A functional, well-tested EP implementation with modern features (adaptive settling, AMP, torch.compile) and comprehensive test coverage.

**What We're Studying:** Whether EP's theoretical advantages (O(1) memory, reduced forgetting) translate to practice, and what qualitative differences emerge from EP's contrastive learning mechanism.

**Value Regardless:** Even if EP doesn't "beat" backprop on standard benchmarks, it provides:
- A research tool for studying biologically plausible learning
- An educational tool demonstrating alternatives to backprop
- A foundation for neuromorphic hardware deployment
- A different perspective on learning (energy-based, local rules)

**Success =** MEP becomes the go-to framework for EP research, with clear documentation of when and why to use it.

---

## Appendix: Research Status

### What Works (Confirmed)
- ✅ EP trains classification models (~89% MNIST)
- ✅ Gradients flow through all layers (verified)
- ✅ Adaptive settling reduces overhead (~1.5× vs 3× slowdown)
- ✅ Error feedback works correctly with Dion updates
- ✅ 142 unit tests pass, 85% coverage
- ✅ Conv2d, LayerNorm, MultiheadAttention support
- ✅ torch.compile compatible
- ✅ AMP compatible

### What Needs Proper Validation (Research Questions)
- ➖ **O(1) memory** - Theoretically sound. Initial measurement flawed (measured total memory, not activations). Need gradient checkpointing validation at extreme depths (1000+ layers).
- ➖ **Continual learning** - Preliminary test invalid (reinitialized model per task). Need sequential training with proper forgetting metric.
- ➖ **Sequential prediction** - Character LM example pending validation.

### What's Different About EP (Qualitative Properties)
- 🔄 No backward pass through computation graph
- 🔄 Local learning rules (no weight transport problem)
- 🔄 Energy-based formulation
- 🔄 Biologically plausible (Hebbian-like updates)
- 🔄 O(1) activation storage (theoretical)
- 🔄 Natural fit for neuromorphic hardware

### Research Plan
1. Proper memory validation (gradient checkpointing, 1000+ layers)
2. Proper CL benchmark (sequential, forgetting metric)
3. Character LM validation
4. Document qualitative differences and research value
5. Identify niches where EP's properties matter

