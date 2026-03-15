# SHD — Spiking Heidelberg Digits

**Result: 90.7% test accuracy** (beats Intel Loihi 1 at 89.0%)

## Benchmark Details

| Property | Value |
|----------|-------|
| Dataset | Spiking Heidelberg Digits (Cramer et al. 2020) |
| Task | Spoken digit classification (0-9, German + English) |
| Classes | 20 |
| Input | 700-channel artificial cochlea spikes |
| Train/Test | 8,156 / 2,264 samples |
| Paper | Cramer et al., IEEE TNNLS, 2020 |

## Best Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.68%** (reported as 90.7%) |
| Best Epoch | 119 / 200 (early stopped at 169) |
| Training Time | 199.7 min (~3.3 hours) |
| Parameters | 1,789,972 (1.79M) |
| Neuron Model | Adaptive LIF (adLIF) |
| Checkpoint | `checkpoints/shd_adlif_v7.pt` |
| Log | `shd/logs/v7_best_90.7.log` |

## Architecture

```
Input:  700 channels (cochlea-encoded spikes)
   |
   v
fc1:    Linear(700, 1024) + recurrent adLIF (1024 → 1024)
   |            └── recurrent dropout (spikes dropped before feedback)
   v
fc_out: Linear(1024, 20) + non-spiking leaky readout (time-averaged)
```

## Reproduce

```bash
python shd/train.py --neuron adlif --hidden 1024 --epochs 200 \
    --weight-decay 5e-4 --device cuda:0 --amp
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Learning rate | 1e-3 → 1e-5 (cosine) |
| Weight decay | 5e-4 |
| Dropout | 0.2 |
| Alpha (membrane decay) | 0.95 (learnable) |
| Rho (threshold decay) | 0.85 (learnable) |
| Beta_a (adaptation init) | 0.05 |
| Label smoothing | 0.05 |
| Activity regularization | lambda=0.01, target_rate=0.05 |
| Event drop | on |
| Batch size | 128 |
| Time bins | 250 |
| Early stopping patience | 50 epochs |

## Training History

| Version | Architecture | Accuracy | Notes |
|---------|-------------|----------|-------|
| Baseline | 700→512(LIF)→20 | 85.9% | Original LIF result |
| v2 | 700→512(adLIF)→20 | 84.5% | beta_a=1.8 bug, plateaued |
| v3 | 700→512(adLIF)→20 | 87.6% | beta_a=0.05 fix |
| v5h | 700→1024→512(adLIF)→20 | 87.2% | Two-layer, no improvement |
| v6b | 700→1024(adLIF)→20 | 87.2% | Bigger single layer |
| **v7** | **700→1024(rec adLIF)→20** | **90.7%** | **Recurrent dropout + activity reg** |

## Key Insight

The breakthrough from 87% to 90.7% came from two changes:
1. **Recurrent dropout**: applying dropout to spikes *before* they feed back into the recurrent connection, not after
2. **Activity regularization**: L2 penalty pushing mean firing rate toward 5%, preventing dead/saturated neurons

## Competitive Context

| Platform | Accuracy | Source |
|----------|----------|--------|
| **Catalyst N2** | **90.7%** | This benchmark |
| Intel Loihi 2 | 90.9% | Bittar & Bhatt 2024 |
| Intel Loihi 1 | 89.0% | Cramer et al. 2022 |
| Software (SRNN) | 90.4% | Yin et al. 2021 |
| Software (SuperSpike) | 84.4% | Zenke & Vogels 2021 |
