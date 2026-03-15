---
language: en
license: mit
library_name: pytorch
tags:
  - spiking-neural-network
  - neuromorphic
  - surrogate-gradient
  - benchmark
  - catalyst
  - shd
datasets:
  - shd
metrics:
  - accuracy
model-index:
  - name: Catalyst SHD SNN Benchmark
    results:
      - task:
          type: audio-classification
          name: Spoken Digit Classification
        dataset:
          name: Spiking Heidelberg Digits (SHD)
          type: shd
        metrics:
          - name: Float Accuracy (N3)
            type: accuracy
            value: 91.0
          - name: Float Accuracy (N2)
            type: accuracy
            value: 84.5
          - name: Float Accuracy (N1)
            type: accuracy
            value: 90.6
          - name: Quantised Accuracy (N3, int16)
            type: accuracy
            value: 90.8
---

# Catalyst SHD SNN Benchmark

Spiking Neural Network trained on the Spiking Heidelberg Digits (SHD) dataset using surrogate gradient BPTT. Achieves 91.0% on SHD with adaptive LIF neurons (90.8% quantised int16).

## Model Description

- **Architecture (N3)**: 700 → 1536 (recurrent adLIF) → 20
- **Architecture (N2)**: 700 → 512 (recurrent adLIF) → 20
- **Architecture (N1)**: 700 → 1024 (recurrent LIF) → 20
- **Neuron model**: Adaptive Leaky Integrate-and-Fire (adLIF) with learnable per-neuron thresholds
- **Training**: Surrogate gradient BPTT, fast-sigmoid surrogate (scale=25), cosine LR scheduling
- **Hardware target**: Catalyst N1/N2/N3 neuromorphic processors

## Results

| Generation | Architecture | Float Accuracy | Params | vs SOTA |
|------------|-------------|----------------|--------|---------|
| **N3** | 700→1536→20 (rec, adLIF) | **91.0%** | 3.47M | Matches Loihi 2 (90.9%) |
| N2 | 700→512→20 (rec, adLIF) | 84.5% | 759K | — |
| N1 | 700→1024→20 (rec, LIF) | 90.6% | 1.79M | Basic LIF baseline |

## Reproduce

```bash
git clone https://github.com/catalyst-neuromorphic/catalyst-benchmarks.git
cd catalyst-benchmarks
pip install -e .

# N3 (91.0%)
python shd/train.py --neuron adlif --hidden 1536 --epochs 200 --device cuda:0 --amp

# N2 (84.5%)
python shd/train.py --neuron adlif --hidden 512 --epochs 200 --device cuda:0

# N1 (90.6%)
python shd/train.py --neuron lif --hidden 1024 --epochs 200 --device cuda:0
```

## Deploy to Catalyst Hardware

```bash
python shd/deploy.py --checkpoint shd_model.pt --threshold-hw 1000
```

## Links

- **Benchmark repo**: [catalyst-neuromorphic/catalyst-benchmarks](https://github.com/catalyst-neuromorphic/catalyst-benchmarks)
- **Hardware**: [catalyst-neuromorphic.com](https://catalyst-neuromorphic.com)
- **N3 paper**: [Zenodo DOI 10.5281/zenodo.18881283](https://zenodo.org/records/18881283)
- **N2 paper**: [Zenodo DOI 10.5281/zenodo.18728256](https://zenodo.org/records/18728256)

## Citation

```bibtex
@misc{catalyst-benchmarks-2026,
  author = {Shulayev Barnes, Henry},
  title = {Catalyst Neuromorphic Benchmarks},
  year = {2026},
  url = {https://github.com/catalyst-neuromorphic/catalyst-benchmarks}
}
```
