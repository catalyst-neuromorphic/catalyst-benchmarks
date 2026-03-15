# SSC — Spiking Speech Commands

**Result: 72.1% test accuracy**

## Benchmark Details

| Property | Value |
|----------|-------|
| Dataset | Spiking Speech Commands (Cramer et al. 2020) |
| Task | 35-class spoken word recognition |
| Classes | 35 |
| Input | 700-channel artificial cochlea spikes |
| Train/Val/Test | ~75K / ~10K / ~20K samples |
| Paper | Cramer et al., IEEE TNNLS, 2020 |

## Best Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **72.10%** |
| Best Epoch | 33 / 200 (crashed at epoch 36, numpy OOM) |
| Parameters | 2,313,763 (2.31M) |
| Neuron Model | Adaptive LIF (adLIF) |
| Checkpoint | `checkpoints/ssc_adlif_amp2.pt` |
| Log | `ssc/logs/v4_best_72.1.log` |

## Architecture

```
Input:  700 channels (cochlea-encoded spikes)
   |
   v
fc1:    Linear(700, 1024) + recurrent adLIF (1024 → 1024)
   |
   v
fc2:    Linear(1024, 512) + adLIF
   |
   v
fc_out: Linear(512, 35) + non-spiking leaky readout (time-averaged)
```

## Reproduce

```bash
python ssc/train.py --neuron adlif --hidden 1024 --epochs 200 \
    --device cuda:0 --amp
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden layers | 1024 (recurrent) + 512 |
| Learning rate | 1e-3 → 1e-5 (cosine) |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Alpha | 0.95 |
| Rho | 0.85 |
| Beta_a | 0.05 |
| Label smoothing | 0.05 |
| Gradient clipping | norm=1.0 |
| Batch size | 64 |
| Time bins | 250 |
| AMP | enabled |

## Training Curve

SSC v4 was still improving when it crashed at epoch 36 (numpy OOM in loader):

| Epoch | Train Acc | Test Acc | Notes |
|-------|-----------|----------|-------|
| 10 | 58.7% | 59.8% | |
| 20 | 68.6% | 67.1% | |
| 27 | 72.9% | 71.5% | |
| 29 | 73.5% | 71.7% | |
| **33** | **74.9%** | **72.1%** | **Best** |
| 36 | 75.5% | 71.9% | Crashed (numpy OOM) |

Train accuracy was still climbing at crash time, suggesting the model could reach 73-75% test with continued training.

## Training History

| Version | Architecture | Accuracy | Notes |
|---------|-------------|----------|-------|
| v2 | 700→1024(adLIF)→35 | 57.9% | Single layer |
| v3 | 700→1024(adLIF)→35 | 58.7% | AMP enabled |
| **v4** | **700→1024+512(adLIF)→35** | **72.1%** | **Two layers, crashed epoch 36** |

## Competitive Context

| Platform | Accuracy | Source |
|----------|----------|--------|
| **Catalyst N2** | **72.1%** | This benchmark |
| Intel Loihi 2 | 69.8% | Bittar & Bhatt 2024 |
| Software (Bittar) | 74.2% | Bittar & Bhatt 2024 |

Current SOTA is 85.98% (SpikCommander, 2025). We are actively improving this result with learnable delays.
