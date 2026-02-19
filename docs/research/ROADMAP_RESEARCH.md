# MEP Development Roadmap

## Executive Summary

**Status:** EP achieves performance parity with backpropagation on classification tasks (~91-95% MNIST). Core functionality validated with 156 passing tests.

**Mission:** Enable biologically plausible learning research with a performant, well-tested EP implementation.

**Key Achievement:** After systematic bug fixes and parameter optimization, EP now matches Adam/SGD performance on standard classification benchmarks.

**Next Focus:** Technical excellence before outreach—achieve O(1) memory, demonstrate scaling advantages, build compelling results.

---

## ✅ What's Complete (Foundation Established)

| Component | Status | Notes |
|-----------|--------|-------|
| Core EP Implementation | ✅ | Fully functional, well-tested |
| Performance Parity | ✅ | EP ~91-95% MNIST (matches Adam/SGD) |
| Test Coverage | ✅ | 156 tests passing, 85% coverage |
| Performance Regression Tests | ✅ | Automated baseline monitoring |
| Dropout Compatibility | ✅ | Fixed - dropout skipped during settling |
| Documentation | ✅ | Comprehensive guides and baselines |
| Benchmark Suite | ✅ | MNIST, CIFAR, continual learning |
| CUDA Kernels | ✅ | Fused settling kernel |
| AMP Support | ✅ | Mixed precision compatible |
| torch.compile | ✅ | Compilation compatible |

### Validated Performance (2026-02-18)

| Benchmark | EP | SGD | Adam | Status |
|-----------|-----|-----|------|--------|
| MNIST (3 epoch) | 91.4% | 91.0% | 90.2% | ✅ EP WINS |
| MNIST (10 epoch) | 95.37% | 93.80% | 95.75% | ✅ EP TIES |
| XOR (100 step) | 100% | 100% | 100% | ✅ PARITY |

---

## 🎯 Strategic Research Trajectory

### Phase 1: Solidify Foundation (Q1 2026) - ✅ COMPLETE

**Goal:** Ensure EP performance is stable, documented, and reproducible.

#### Completed
- [x] Fix gradient accumulation bug
- [x] Fix baseline configuration bugs
- [x] Fix dropout incompatibility
- [x] Discover optimal settling parameters
- [x] Achieve performance parity with backprop
- [x] Create performance regression tests
- [x] Document performance baselines

**Success Criteria:** ✅ All met

---

### Phase 2: Technical Excellence (Q2-Q3 2026) - HIGH PRIORITY

**Goal:** Achieve compelling technical advantages before external outreach.

**Philosophy:** Build undeniable results first, then share. Partnerships are more productive when we have clear advantages to demonstrate.

**Detailed Plans:**
- [phase2_detailed_plan.md](phase2_detailed_plan.md) — Full 6-month plan
- [phase2_week1-2.md](phase2_week1-2.md) — Immediate action items

---

#### Priority 1: O(1) Memory Implementation 🔴 CRITICAL

**Hypothesis:** EP can achieve O(1) activation memory by avoiding unnecessary PyTorch functionality that triggers activation storage.

**Why this matters:**
- Theoretical advantage of EP over backprop
- Enables training deeper networks on memory-constrained hardware
- Key differentiator for neuromorphic deployment

**Technical Approach:**

1. **Avoid PyTorch Autograd Overhead**
   - Use `torch.no_grad()` aggressively during settling
   - Manual gradient computation without graph construction
   - Detach states early and often

2. **Minimize Intermediate Activations**
   - In-place operations where possible
   - Avoid storing computation history
   - Custom settling kernels that don't trigger autograd

3. **Gradient Checkpointing for EP**
   - Store only boundary states
   - Recompute intermediate states during contrast
   - Trade compute for memory (favorable for EP)

4. **Custom CUDA Kernels**
   - Fused settling kernel (already exists)
   - Avoid PyTorch dispatch overhead
   - Direct memory management

**Action Items:**
- [ ] Profile current memory usage by component
- [ ] Identify PyTorch operations triggering activation storage
- [ ] Implement manual settling without autograd
- [ ] Test at extreme depths (1000, 2000, 5000+ layers)
- [ ] Compare vs backprop+checkpointing at each depth

**Success Criteria:**
- EP activation memory flat vs depth (O(1))
- Backprop activation memory linear vs depth (O(depth))
- Crossover point where EP wins identified
- Results reproducible and documented

**Timeline:** 2-3 months
**Impact:** Very High - validates core EP advantage

---

#### Priority 2: Deep Network Scaling

**Hypothesis:** With O(1) memory, EP can train networks that are impractical for backprop.

**Action Items:**
- [ ] Test at 1000+ layer depth
- [ ] Test at 5000+ layer depth
- [ ] Test at 10000+ layer depth (if memory allows)
- [ ] Document training dynamics at extreme depth
- [ ] Identify any depth-related failure modes

**Success Criteria:**
- Successful training at 5000+ layers
- Convergence maintained at depth
- Clear scaling curves documented

**Timeline:** 1-2 months (after O(1) memory)
**Impact:** High - demonstrates unique EP capability

---

#### Priority 3: Continual Learning (Technical Foundation)

**Hypothesis:** EP + proper CL methods can reduce catastrophic forgetting.

**Current Status:** Error feedback reduces forgetting (32% vs 48%) but EWC is more effective (5-15%).

**Action Items:**
- [ ] Implement EWC integration for EP
- [ ] Test on standard CL benchmarks (Permuted MNIST, Split CIFAR)
- [ ] Compare EP+EWC vs backprop+EWC
- [ ] Analyze why EP+EF reduces forgetting
- [ ] Publish technical report with results

**Timeline:** 2-3 months
**Impact:** Medium-High - CL is important research area

---

#### Priority 4: Speed Optimization

**Hypothesis:** Settling overhead can be reduced without losing convergence.

**Current Status:** EP is 2-3× slower than backprop (fundamental settling cost).

**Action Items:**
- [ ] Profile settling time by component
- [ ] Test adaptive settling (early stopping)
- [ ] Optimize CUDA kernels
- [ ] Explore approximate settling methods
- [ ] Document speed/accuracy tradeoffs

**Timeline:** 1-2 months
**Impact:** Medium - speed is important for adoption

---

### Phase 3: Results & Outreach (Q4 2026+) - CONTINGENT

**Goal:** Share compelling results with research community.

**Prerequisites:**
- ✅ O(1) memory demonstrated
- ✅ Deep scaling results (5000+ layers)
- ✅ CL results (EP+EWC competitive)
- ✅ Speed optimizations complete

**Only after Phase 2 is complete:**

#### Neuromorphic Partnerships
- [ ] Reach out to Intel Labs (Loihi)
- [ ] Reach out to SpiNNaker group
- [ ] Benchmark on neuromorphic hardware
- [ ] Publish energy efficiency study

#### Biological Plausibility Research
- [ ] Partner with computational neuroscience labs
- [ ] Compare EP dynamics to neural data
- [ ] Publish biological plausibility study

#### Community Building
- [ ] Release comprehensive benchmark suite
- [ ] Write tutorial/guide papers
- [ ] Present at relevant venues (NeurIPS, ICLR, CNS)

---

## 📊 Success Metrics

| Metric | Current | Target (6mo) | Target (12mo) |
|--------|---------|-------------|---------------|
| MNIST Accuracy | 95.37% | 95%+ (maintain) | 95%+ (maintain) |
| Test Coverage | 85% | 85%+ (maintain) | 90% |
| Memory Scaling | O(depth) | **O(1)** | O(1) verified |
| Max Depth Tested | 2000 | **5000+** | 10000+ |
| CL Forgetting | 32% (EF) | **<15%** (EWC) | <10% |
| Speed vs BP | 2-3× slower | **1.5-2×** | 1.5× |
| External Contributors | 0 | 2+ | 10+ |
| GitHub Stars | ~0 | 50+ | 200+ |
| Citations | 0 | 0 (pre-results) | 10+ (post-results) |

---

## 🔬 Open Research Questions

### Technical Questions (Phase 2 Focus)
1. **Can EP achieve O(1) memory?** What PyTorch features trigger activation storage?
2. **What's the maximum trainable depth?** 5000? 10000? 100000+ layers?
3. **Can settling be accelerated?** Without losing convergence?
4. **Does EP+EWC outperform backprop+EWC?** On standard CL benchmarks?

### Scientific Questions (Phase 3 Focus)
1. **What is EP's true niche?** Where does it genuinely excel vs backprop?
2. **Can EP train transformers?** What architectural changes are needed?
3. **Is EP more energy-efficient?** Under what conditions?
4. **Does biological plausibility matter?** For what applications?

---

## 🚧 Known Limitations (Honest Assessment)

| Limitation | Status | Plan |
|------------|--------|------|
| Memory usage | 🔴 **Priority** | O(1) implementation in progress |
| Training speed | ⚠️ Acceptable | 2× slowdown for biological plausibility is reasonable |
| Dropout incompatibility | ✅ Fixed | Skip dropout during settling |
| Continual learning | 🔴 In progress | EWC integration planned |
| Very deep networks | 🔴 Untested | Waiting for O(1) memory |

---

## 📁 Key Documentation

| Document | Purpose |
|----------|---------|
| [README.md](../README.md) | Quick start, how MEP works |
| [docs/index.md](../docs/index.md) | Documentation hub |
| [docs/benchmarks/PERFORMANCE_BASELINES.md](../docs/benchmarks/PERFORMANCE_BASELINES.md) | Performance thresholds, optimal config |
| [docs/benchmarks/VALIDATION_RESULTS.md](../docs/benchmarks/VALIDATION_RESULTS.md) | Full validation study |
| [docs/methods_paper.md](../docs/methods_paper.md) | Preprint-ready methods paper |

---

## 💡 Why This Roadmap

**Phase 2 before Phase 3:** Outreach is more effective with compelling results. A partnership proposal that says "EP matches backprop accuracy" is good. One that says "EP matches backprop AND achieves O(1) memory AND trains 10000-layer networks" is irresistible.

**Technical excellence first:**
1. Solve the memory problem (O(1) implementation)
2. Demonstrate unique capabilities (deep scaling)
3. Show competitive CL performance (EP+EWC)
4. Then share with the world

**This approach:**
- Avoids premature hype
- Builds genuine advantages
- Makes partnerships more productive
- Establishes MEP as serious research tool

---

## 📅 Immediate Action Items (Next 2 Weeks)

- [ ] Profile memory usage by component
- [ ] Identify PyTorch operations triggering activation storage
- [ ] Design O(1) memory settling implementation
- [ ] Set up deep network scaling test infrastructure
- [ ] Review EWC implementation for EP integration

---

## 🎓 Final Thought

**We've proven EP can match backprop on accuracy.** That was step one.

**Step two is proving EP's theoretical advantages in practice:** O(1) memory, deep scaling, continual learning.

**Step three is sharing those results** with researchers who can benefit from them.

This roadmap prioritizes substance over hype, results over partnerships, and technical excellence over premature announcements.

**Success =** MEP becomes the standard tool for EP research, with demonstrated advantages that speak for themselves.

---

*Last updated: 2026-02-18*
*Status: Foundation complete, technical excellence phase beginning*
