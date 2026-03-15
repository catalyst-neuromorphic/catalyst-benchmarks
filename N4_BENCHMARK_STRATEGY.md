# N4 Benchmark Strategy: Path to #1 on Every Benchmark

**Date:** 6 March 2026
**Objective:** Achieve world #1 accuracy on every benchmark we choose to include, using N4 hardware-targetable architectures.

---

## CURRENT STATE vs WORLD SOTA

| Benchmark | Our Best | Model | World SOTA | Method | Gap |
|-----------|----------|-------|------------|--------|-----|
| **SHD** | 91.0% | 2-layer rec adLIF | 96.44% (MCRE, 2025) | Multi-scale temporal chunking | **-5.4%** |
| **SSC** | 76.4% | 2-layer rec adLIF | 85.98% (SpikCommander, 2025) | Spike-driven attention | **-9.6%** |
| **N-MNIST** | 99.2% | Conv+adLIF | 99.67% (STCA-SNN, 2023) | Temporal-channel attention | **-0.5%** |
| **DVS Gesture** | None yet | — | 99.3% (SG-SNN, 2025) | Conv SNN + gradient | **N/A** |
| **GSC-12** | 88.0% | 1-layer rec adLIF | 96.92% (SpikCommander, 2025) | Spike attention + delays | **-8.9%** |
| **GSC-35** | Training | — | 95.35% (DCLS-Delays, 2024) | Learnable delays, vanilla LIF | **TBD** |

**Loihi 2 hardware results for reference:**
- SHD: 90.9% (Khacef et al. 2025, EventProp pipeline)
- SSC: 69.8% (Khacef et al. 2025, feedforward+delay)
- GSC: 71.1% (Khacef et al. 2025)

---

## THE THREE TECHNIQUES THAT MATTER

Analysis of every paper in the top 5 for each benchmark reveals three techniques account for virtually all gains above 90%:

### 1. LEARNABLE SYNAPTIC DELAYS (Hammouamri et al., ICLR 2024)

**What it does:** Each synapse has a learnable delay d_i that shifts the input spike in time. Trained via differentiable linear interpolation (gradients flow through the delay).

**Impact (standalone, vanilla LIF, NO recurrence):**
- SHD: 95.07% (+4% over our 91%)
- SSC: 80.69% (+4.3% over our 76.4%)
- GSC-35: 95.35% (+7% over our 88%)

**Why it works:** Speech is fundamentally temporal — phonemes have precise onset timing. Delays let each neuron "listen" to different time offsets, effectively creating a learned temporal receptive field.

**N4 hardware mapping:** DIRECT. N4 has 8-bit per-synapse delay fields in all synapse formats (Full 78-bit, Inference 53-bit). Delay queue per core: 4,096 slots, configurable up to 255 timesteps. The trained floating-point delays quantize to integer delay register values.

**Our code status:** `DelayedLinear` class EXISTS in `common/neurons.py` but is NOT USED in any benchmark training script.

### 2. ADAPTIVE NEURONS (SE-adLIF / Bittar & Garner 2022)

**What it does:** Neuron threshold adapts based on recent spiking history. Specifically the Symplectic Euler discretization updates adaptation BEFORE threshold computation.

**Impact:**
- SHD: 95.81% (SE-adLIF alone, Bittar 2022)
- Combined with delays: likely 96%+

**Why it works:** Adaptive thresholds give neurons a form of short-term memory — recently active neurons become harder to fire, which decorrelates representations and improves temporal discrimination.

**N4 hardware mapping:** DIRECT. N4's ALIF neuron model has hardware adaptation (theta increments on spike, multiplicative decay). Liquid Time-Constants (LTC) add input-dependent decay: `decay_eff = decay_v + tau_input * |i_syn|`.

**Our code status:** Already implemented and used (`AdaptiveLIF` in `common/neurons.py`). But our SE implementation matches Bittar's formulation. The issue is we haven't COMBINED it with delays.

### 3. SPIKE-DRIVEN ATTENTION (SpikCommander, Chen et al. 2025)

**What it does:** Multi-view temporal attention over spike trains. Q/K projections from membrane voltages (continuous), V from spikes (binary). Self-attention over temporal windows.

**Impact:**
- SHD: 96.41% (SpikCommander)
- SSC: 85.98% (SpikCommander)
- GSC: 96.92% (SpikCommander)

**Why it works:** Attention dynamically weights which time steps matter for classification. For speech, this means the network learns which phoneme boundaries are informative.

**N4 hardware mapping:** PARTIAL. N4 has a native spiking transformer with 8 attention heads per core, CAM-based spike-coincidence detection, and Winner-Take-All output. However, SpikCommander's exact "multi-view" architecture may need adaptation to fit N4's attention hardware. The key question is whether N4's attention heads can implement the Q/K/V projections needed.

**Our code status:** One experimental script (`shd/train_attention.py`) with basic 8-head attention. Result: 86.8% on SHD (worse than adLIF). Not integrated into main benchmarks.

---

## N4 UNIQUE HARDWARE ADVANTAGES (Beyond Top-3 Techniques)

These are features NO other neuromorphic chip has:

| Feature | What It Enables | Benchmark Impact |
|---------|----------------|------------------|
| **Liquid Time-Constants** | Input-dependent temporal filtering | SHD/SSC/GSC: handle variable-speed speech |
| **Dendritic Compartments (16/neuron)** | Nonlinear feature extraction per neuron | All: richer computation per neuron |
| **12-bit Eligibility Traces** | Precise STDP/e-prop (vs 8-bit on N3) | All: reduced quantization noise in learning |
| **Spike Tensor Core (16x16)** | 256 ops/cycle for conv layers | DVS Gesture/N-MNIST: 64x throughput |
| **Dual Weights (fast/slow)** | Complementary Learning Systems | Continual learning benchmarks |
| **Structural Plasticity** | Grow/prune synapses during training | Online learning benchmarks |
| **32 Neuromodulation Channels** | Task-specific reward modulation | Reinforcement learning benchmarks |
| **Oscillator Bank (4/tile)** | Theta/gamma temporal coding | SHD/SSC: syllable rhythm encoding |
| **Predictive Coding Engine** | Only propagate prediction errors | Efficiency + potential accuracy boost |
| **Surrogate Gradient Engine** | Hardware σ'(v-θ) computation | On-chip training for all benchmarks |

---

## PER-BENCHMARK STRATEGY

### SHD: 91.0% -> 96.5%+ (Target: #1)

**Current architecture:** 700 -> 1024(rec, adLIF) -> 512(rec, adLIF) -> 20
**Current result:** 91.0%

**Evidence-based path:**

| Step | Technique | Expected Result | Evidence |
|------|-----------|----------------|----------|
| 1 | Add learnable delays to both layers | ~94-95% | DCLS-Delays alone: 95.07% |
| 2 | Combine delays + adLIF (SE formulation) | ~95.5-96% | SE-adLIF alone: 95.81%, delays complement |
| 3 | Add temporal attention (4-8 heads) | ~96-96.5% | SpikCommander: 96.41% |
| 4 | LTC (liquid time-constants) | ~96.5%+ | Novel — no prior work combines LTC+delays+adLIF |

**Architecture (proposed):**
```
Input (700) -> DelayedLinear(max_delay=60)
  -> AdaptiveLIF(1024, recurrent, LTC)
  -> Attention(4 heads)
  -> DelayedLinear(max_delay=30)
  -> AdaptiveLIF(512, recurrent, LTC)
  -> ReadoutLIF(20)
```

**Key insight:** DCLS-Delays uses vanilla LIF (no adaptation) and gets 95%. We already have adLIF which alone gets 95.8%. The combination should push beyond both. Nobody has published delays + adLIF + attention together.

**Hardware mapping:** All components map to N4:
- DelayedLinear -> per-synapse 8-bit delay registers
- AdaptiveLIF -> ALIF hardwired model (1 cycle)
- LTC -> hardware liquid time-constant circuit
- Attention -> 8-head spiking transformer unit
- Recurrence -> configurable recurrent routing

**Estimated training time:** ~4-6 hours on RTX 3080 Ti (200 epochs, 3.5M params)

### SSC: 76.4% -> 86%+ (Target: #1)

**Current architecture:** 700 -> 1024(rec, adLIF) -> 512(rec, adLIF) -> 35
**Current result:** 76.4%

This is our biggest gap. SpikCommander gets 85.98%. The key papers show:
- Without delays: ~72-77% ceiling for recurrent SNNs
- With delays: 80.69% (DCLS, no attention, no adaptation)
- With attention: 85.98% (SpikCommander)

**Path:**

| Step | Technique | Expected | Evidence |
|------|-----------|----------|----------|
| 1 | Add delays (max_delay=80, SSC is longer) | ~80-82% | DCLS-Delays: 80.69% |
| 2 | Combine delays + adLIF + 2 layers | ~82-84% | adLIF adds ~2-3% on SHD |
| 3 | Add temporal attention (8 heads) | ~85-86% | SpikCommander: 85.98% |
| 4 | Better augmentation (MixUp, CutMix) | ~86%+ | Standard technique, +0.5-1% |

**Architecture (proposed):**
```
Input (700) -> DelayedLinear(max_delay=80)
  -> AdaptiveLIF(1024, recurrent, LTC)
  -> SpikeAttention(8 heads)
  -> DelayedLinear(max_delay=40)
  -> AdaptiveLIF(512, recurrent, LTC)
  -> ReadoutLIF(35)
```

**Why SSC is harder than SHD:** 35 classes (vs 20), longer utterances, more acoustic variability. Delays are even MORE critical here because the temporal structure spans more timesteps.

**Estimated training time:** ~8-12 hours (300 epochs, 4M params, SSC dataset is larger)

### GSC-12: 88.0% -> 97%+ (Target: #1)

**Current architecture:** 40 -> 512(rec, adLIF) -> 12
**Current result:** 88.0%

**The evidence is overwhelming:**
- DCLS-Delays (vanilla LIF, no recurrence): 95.35% on GSC-35
- SpikCommander: 96.92% on GSC-12
- Spiking LMUFormer: 96.12% (Legendre Memory Units)

Our 88% uses a tiny model (290K params) with no delays. The ceiling is much higher.

**Path:**

| Step | Technique | Expected | Evidence |
|------|-----------|----------|----------|
| 1 | Increase model size (512->1024) | ~90-91% | SpiNNaker 2: 91.1% |
| 2 | Add delays (max_delay=40) | ~94-95% | DCLS: 95.35% on harder GSC-35 |
| 3 | Add adLIF adaptation | ~95-96% | +1-2% typical |
| 4 | Add temporal attention | ~96-97% | SpikCommander: 96.92% |
| 5 | Hybrid INT8 front-end (N4 ANN mode) | ~97%+ | Preserve mel info before spiking |

**Architecture (proposed):**
```
Mel spectrogram (40 bands, 100 steps)
  -> ANNINT8Linear(40, 256)  # N4 ANN dense mode, preserves continuous features
  -> DelayedLinear(max_delay=40)
  -> AdaptiveLIF(1024, recurrent, LTC)
  -> SpikeAttention(4 heads)
  -> ReadoutLIF(12)
```

**Estimated training time:** ~3-4 hours (200 epochs, 1.5M params)

### N-MNIST: 99.2% -> 99.7%+ (Target: #1)

**Current architecture:** Conv2D+LIF -> Conv2D+LIF -> 256(adLIF) -> 10
**Current result:** 99.2%

This benchmark is nearly saturated. STCA-SNN gets 99.67% using temporal-channel attention. The gap is only 0.47%.

**Path:**

| Step | Technique | Expected | Evidence |
|------|-----------|----------|----------|
| 1 | Add temporal-channel attention | ~99.4-99.5% | STCA-SNN approach |
| 2 | Knowledge distillation from ANN teacher | ~99.5-99.6% | +0.1-0.2% typical |
| 3 | Test augmentation (polarity flip, jitter) | ~99.6-99.7% | Standard DVS augmentation |

**Lower priority** — diminishing returns. 99.2% is already competitive and N-MNIST is not considered a serious differentiator.

**Estimated training time:** ~1-2 hours

### DVS128 Gesture: New -> 99%+ (Target: #1)

**Current result:** 80.3% (early experiment, not properly tuned)
**SOTA:** SG-SNN 99.3%, TENNs-PLEIADES 99.59% (non-SNN)

**Path:**

| Step | Technique | Expected | Evidence |
|------|-----------|----------|----------|
| 1 | Proper conv SNN (3-layer) + augmentation | ~95-96% | Standard approach |
| 2 | Add temporal attention | ~97-98% | Spiking transformer papers |
| 3 | Spike Tensor Core optimization (N4) | ~98-99% | 64x throughput for conv |
| 4 | DVS-specific augmentation (NDA) | ~99%+ | Neuromorphic Data Augmentation |

**Architecture (proposed):**
```
Input (2, 128, 128, T=300)
  -> Conv2D(2, 64, 3x3) + BN + LIF
  -> Conv2D(64, 128, 3x3) + BN + LIF
  -> Conv2D(128, 256, 3x3) + BN + LIF
  -> GlobalAvgPool -> AdaptiveLIF(512, recurrent)
  -> SpikeAttention(4 heads)
  -> ReadoutLIF(11)
```

**Estimated training time:** ~6-8 hours (DVS Gesture has longer sequences)

---

## WHAT WE ALREADY HAVE vs WHAT WE NEED TO BUILD

### Already Implemented (common/neurons.py):
- [x] AdaptiveLIF with SE discretization
- [x] DelayedLinear (exists but unused!)
- [x] ConvLIFLayer with batch norm
- [x] Activity regularization (proven +2-3%)
- [x] Event drop augmentation
- [x] Time stretch augmentation
- [x] Cosine annealing + warmup
- [x] ANNINT8 for hybrid layers (N3/N4)

### Need to Build:
- [ ] **Integrate DelayedLinear into all benchmarks** (code exists, just wire it up)
- [ ] **LiquidTimeConstant neuron** (input-dependent decay, new class)
- [ ] **SpikeAttention module** (proper implementation, not the experimental one)
- [ ] **MixUp/CutMix on spike trains** (new augmentation)
- [ ] **Multiple surrogate gradients** (alpha-sigmoid, triangle, test which is best)
- [ ] **Temporal-channel attention** (for DVS/N-MNIST)
- [ ] **Knowledge distillation loss** (ANN teacher -> SNN student)
- [ ] **Learnable per-neuron thresholds** (currently fixed on non-adLIF layers)

### Estimated Build Time:
- DelayedLinear integration: 30 minutes (wire existing code)
- LiquidTimeConstant: 1-2 hours (new neuron, ~50 lines)
- SpikeAttention: 2-3 hours (proper multi-head, ~100 lines)
- MixUp/CutMix: 30 minutes (~20 lines)
- Surrogate variants: 30 minutes (~15 lines each)
- Full benchmark re-training: ~24-36 hours total GPU time

---

## PRIORITY ORDER (Bang for Buck)

### Phase 1: DELAYS (Biggest single improvement, lowest effort)
**Action:** Integrate `DelayedLinear` into SHD, SSC, GSC training scripts.
**Expected total improvement:** +4-8% across all speech benchmarks.
**Effort:** 2-3 hours coding + 12 hours training.
**Justification:** DCLS-Delays proves that delays alone close most of the gap. Our `DelayedLinear` code already exists and is tested.

### Phase 2: DELAYS + adLIF COMBINATION
**Action:** Ensure both delays and adaptive neurons are active simultaneously.
**Expected total improvement:** +1-3% on top of Phase 1.
**Effort:** 1 hour coding + 12 hours training.
**Justification:** Nobody has published this exact combination. SE-adLIF and DCLS-Delays are complementary mechanisms (one adapts thresholds, other adapts timing).

### Phase 3: SPIKE ATTENTION
**Action:** Build proper multi-head spike attention module, integrate into all speech benchmarks.
**Expected total improvement:** +1-2% on top of Phase 2.
**Effort:** 3-4 hours coding + 12 hours training.
**Justification:** SpikCommander's edge over DCLS comes entirely from attention. This is the final ingredient.

### Phase 4: AUGMENTATION + TUNING
**Action:** MixUp, CutMix, better surrogate gradients, hyperparameter sweep.
**Expected total improvement:** +0.5-1%.
**Effort:** 2 hours coding + 24 hours training (sweep).
**Justification:** Polish. Gets us from "competitive" to "definitive #1".

### Phase 5: DVS + N-MNIST
**Action:** Build proper conv SNN for DVS Gesture, add attention to N-MNIST.
**Expected improvement:** 99%+ on both.
**Effort:** 4 hours coding + 8 hours training.
**Justification:** Complete the benchmark suite.

---

## CRITICAL QUESTION: CAN WE CLAIM "HARDWARE-DEPLOYED"?

The key differentiator is that SpikCommander, DCLS-Delays, SE-adLIF are all **GPU-trained, GPU-evaluated** models. Nobody has deployed them on neuromorphic hardware.

If we:
1. Train on GPU with delays + adLIF + attention
2. Deploy the trained model on N4 FPGA
3. Run inference on hardware and report accuracy

Then we can legitimately claim **"#1 accuracy among hardware-deployed neuromorphic systems"** even if a GPU-only model technically scores 0.1% higher. This is a much stronger claim than any software-only result because it proves the architecture actually works in silicon.

**N4 FPGA validation** (AFI v4, 126/126 PASS, 14,983 ts/sec) is already done. We just need to:
1. Train the improved models on GPU
2. Quantize weights for hardware (our `deploy.py` already does this)
3. Load onto N4 FPGA
4. Run the test set through hardware
5. Report hardware accuracy

Even a 0.5-1% accuracy loss from quantization would still put us well above Loihi 2 and potentially above all other hardware results.

---

## REALISTIC ACCURACY TARGETS (Conservative)

| Benchmark | Current | Phase 1 (delays) | Phase 2 (+adLIF) | Phase 3 (+attention) | Phase 4 (tuning) | SOTA |
|-----------|---------|-------------------|-------|----------|---------|------|
| **SHD** | 91.0% | 94-95% | 95-96% | 96-96.5% | 96.5%+ | 96.44% |
| **SSC** | 76.4% | 80-82% | 82-84% | 85-86% | 86%+ | 85.98% |
| **GSC-12** | 88.0% | 94-95% | 95-96% | 96-97% | 97%+ | 96.92% |
| **N-MNIST** | 99.2% | 99.2% | 99.2% | 99.5% | 99.6%+ | 99.67% |
| **DVS Gesture** | 80.3% | 95% | 97% | 98-99% | 99%+ | 99.3% |

These are conservative estimates based on published results with each technique. The combination of ALL techniques together (which no one has published) could exceed current SOTA on every benchmark.

---

## THE STORY WE TELL

Once we have these results:

> "Catalyst N4 achieves state-of-the-art accuracy across five major neuromorphic benchmarks, the first time a single hardware platform has held simultaneous #1 positions on SHD, SSC, GSC, DVS Gesture, and N-MNIST. Unlike prior SOTA results achieved on GPUs with architectures that cannot run on neuromorphic hardware, all Catalyst results are validated on FPGA hardware at 62.5 MHz, demonstrating that our hardware-software co-design approach eliminates the accuracy gap between GPU training and neuromorphic deployment."

This is defensible if:
1. We match or exceed SOTA accuracy
2. We run inference on actual N4 FPGA hardware
3. We report both GPU-trained and hardware-deployed numbers

---

## GPU RESOURCE PLAN

We have:
- **cuda:0** = RTX 3080 Ti (12GB) — currently running DVS Gesture v3 (PID 169020)
- **cuda:1** = RTX 3080 (20GB) — SSC fine-tune running (PID 202212, should be killed)

**Proposed schedule:**
1. Kill SSC FT (PID 202212) — frees cuda:1
2. Run SHD+delays on cuda:1 (Phase 1, ~4-6 hours)
3. When DVS Gesture v3 finishes on cuda:0, run SSC+delays
4. Iterate through phases

**Total estimated GPU time for all benchmarks:** ~72-96 hours (~3-4 days)
