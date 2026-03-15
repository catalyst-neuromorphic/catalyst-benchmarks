# N-MNIST — Neuromorphic MNIST

**Result: 99.2% test accuracy**

## Benchmark Details

| Property | Value |
|----------|-------|
| Dataset | Neuromorphic MNIST (Orchard et al. 2015) |
| Task | Handwritten digit recognition from DVS events |
| Classes | 10 |
| Input | 34x34 DVS events (2 polarities) |
| Train/Test | 60,000 / 10,000 samples |
| Paper | Orchard et al., Frontiers in Neuroscience, 2015 |

## Best Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.23%** (reported as 99.2%) |
| Best Epoch | 49 / 50 |
| Training Time | 165.7 min (~2.8 hours) |
| Parameters | 466,273 (466K) |
| Neuron Model | LIF (conv front-end + LIF readout) |
| Checkpoint | `checkpoints/nmnist_conv_v2b.pt` |
| Log | `nmnist/logs/v2b_best_99.2.log` |

## Architecture

```
Input:  [T=20, 2, 34, 34] DVS event frames
   |
   v
conv1:  Conv2D(2, 32, 3) + BatchNorm + LIF  → [T, 32, 32, 32]
pool1:  AvgPool2D(2)                          → [T, 32, 16, 16]
   |
   v
conv2:  Conv2D(32, 64, 3) + BatchNorm + LIF → [T, 64, 14, 14]
pool2:  AvgPool2D(2)                          → [T, 64, 7, 7]
   |
   v
flat:   Flatten                               → [T, 3136]
   |
   v
fc:     Linear(3136, 256) + LIF
   |
   v
fc_out: Linear(256, 10) + non-spiking readout (time-averaged)
```

## Reproduce

```bash
python nmnist/train.py --conv --epochs 50 --device cuda:0 --amp
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Conv channels | 32 → 64 |
| FC hidden | 256 |
| Learning rate | 1e-3 → 1e-5 (cosine) |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Time bins | 20 |
| Batch size | 128 |
| AMP | enabled |

## Training Curve

| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 10 | 98.2% | 98.5% |
| 20 | 98.7% | 98.7% |
| 30 | 98.9% | 98.9% |
| 40 | 99.0% | 99.1% |
| 45 | 99.1% | 99.2% |
| **49** | **99.2%** | **99.23%** |

## Training History

| Version | Architecture | Accuracy | Notes |
|---------|-------------|----------|-------|
| v1 | FC only (2312→512→256→10, adLIF) | 97.8% | Fully-connected |
| v2 | Conv2D + LIF | 99.0% | Conv front-end |
| **v2b** | **Conv2D + LIF (longer training)** | **99.2%** | **Best** |

## Competitive Context

| Platform | Accuracy | Source |
|----------|----------|--------|
| Intel Loihi 1 | 99.5% | Shrestha & Orchard 2018 |
| **Catalyst N2** | **99.2%** | This benchmark |
| Software (SLAYER) | 99.2% | Shrestha & Orchard 2018 |

N-MNIST at 99.2% is near the ceiling for this benchmark. Loihi 1 edges us by 0.3%.
