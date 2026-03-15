# Catalyst Neuromorphic Benchmarks

[![CI](https://github.com/catalyst-neuromorphic/catalyst-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/catalyst-neuromorphic/catalyst-benchmarks/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official benchmark suite for [Catalyst neuromorphic processors](https://catalyst-neuromorphic.com). Trains spiking neural networks (SNNs) on standard neuromorphic benchmarks and deploys to Catalyst N1/N2/N3 FPGA hardware.

All results are reproducible. Clone, install, train, deploy.

## Results — Catalyst N3 (Latest)

| Benchmark | Classes | Architecture | Neuron | Float Acc | Params |
|-----------|---------|-------------|--------|-----------|--------|
| **SHD** | 20 | 700→1536→20 (rec) | adLIF | **91.0%** | 3.47M |
| **SSC** | 35 | 700→1024→512→35 (rec) | adLIF | **76.4%** | 2.31M |
| **N-MNIST** | 10 | Conv2D+LIF→10 | LIF | **99.1%** | 691K |
| **GSC-12** | 12 | 40→512→12 (rec, S2S) | adLIF | **88.0%** | 291K |
| **DVS Gesture** | 11 | Deep conv+rec | adLIF | **89.0%** | ~1.2M |

All N3 models use adaptive LIF neurons with surrogate gradient BPTT and cosine LR scheduling.

## Results — Catalyst N2

| Benchmark | Classes | Architecture | Neuron | Float Acc | Params |
|-----------|---------|-------------|--------|-----------|--------|
| **SHD** | 20 | 700→512→20 (rec) | adLIF | **84.5%** | 759K |
| **SSC** | 35 | 700→1024→512→35 (rec) | adLIF | **72.1%** | 2.31M |
| **N-MNIST** | 10 | Conv2D+LIF→10 | adLIF | **97.8%** | 466K |
| **GSC KWS** | 12 | 40→512→12 (rec, S2S) | adLIF | **88.0%** | 291K |
| **MIT-BIH ECG** | 5 | 187→128→5 (rec) | adLIF | **90.9%** | ~35K |

All N2 models deploy to Catalyst N2 FPGA hardware via int16 weight quantization.

## Results — Catalyst N1

| Benchmark | Classes | Architecture | Neuron | Float Acc | Params |
|-----------|---------|-------------|--------|-----------|--------|
| **SHD** | 20 | 700→1024→20 (rec) | LIF | **90.6%** | 1.79M |
| **N-MNIST** | 10 | Conv2D+LIF→10 | LIF | **99.2%** | 466K |
| **DVS Gesture** | 11 | Deep conv+rec | LIF | **69.7%** | ~1.2M |
| **GSC-12** | 12 | 40→512→12 (rec, S2S) | LIF | **86.4%** | 291K |

N1 uses basic LIF neurons only (no adaptation). Demonstrates competitive performance through model capacity alone — the N2's adaptive threshold provides a clear efficiency advantage at matched model sizes.

## Competitive Context

Loihi 2 results from [Mészáros et al. 2025](https://arxiv.org/abs/2510.13757) (Table I, best per-dataset).

| Benchmark | Catalyst N3 | Catalyst N2 | Catalyst N1 | Loihi 2 |
|-----------|-------------|-------------|-------------|---------|
| SHD | **91.0%** | 84.5% | 90.6% | 90.9% |
| SSC | **76.4%** | 72.1% | — | 69.8% |
| N-MNIST | **99.1%** | 97.8% | — | — |
| GSC-12 | **88.0%** | 88.0% | — | — |

## FPGA Hardware Characterisation

### Kria K26 Edge (xczu5ev, 2-core variants, 100 MHz target)

| Processor | LUTs | LUT% | FFs | FF% | BRAM | DSP | WNS | Fmax | Power |
|-----------|------|------|-----|-----|------|-----|-----|------|-------|
| **N1** | 20,109 | 17.2% | 30,847 | 13.2% | 52.5 (36.5%) | 14 (1.1%) | +0.008ns | 100 MHz | 0.642W |
| **N2** | 26,431 | 22.6% | 38,666 | 16.5% | 52.5 (36.5%) | 16 (1.3%) | -0.168ns | ~97 MHz | 0.688W |
| **N3** | 53,420 | 45.6% | 80,395 | 34.3% | 24 (16.7%) | 20 (1.6%) | -7.075ns | ~58.5 MHz | 0.867W |

### AWS F2 Cloud FPGA (Xilinx VU47P)

| Processor | Tests | Pass Rate | Throughput | Frequency |
|-----------|-------|-----------|------------|-----------|
| N1 | — | PASS | — | 62.5 MHz |
| N2 | 28/28 | 100% | 8,690 ts/sec | 62.5 MHz |
| N3 | 19/19 | 100% | 14,512 ts/sec | 62.5 MHz |

## Quick Start

```bash
git clone https://github.com/catalyst-neuromorphic/catalyst-benchmarks.git
cd catalyst-benchmarks
pip install -e .
```

### Train a benchmark

```bash
# SHD (Spiking Heidelberg Digits) — 91.0% with N3 adLIF
python shd/train.py --neuron adlif --hidden 1536 --epochs 200 --device cuda:0 --amp

# SSC (Spiking Speech Commands) — 76.4% with N3 adLIF
python ssc/train.py --hidden1 1024 --hidden2 768 --recurrent2 --epochs 70 --device cuda:0 --amp

# N-MNIST — 99.1% with Conv front-end
python nmnist/train.py --data-dir data/nmnist --epochs 80 --device cuda:0 --amp

# Google Speech Commands (KWS) — 88.0% with Speech2Spikes encoding
python gsc_kws/train.py --hidden 512 --dropout 0.3 --epochs 200 --device cuda:0 --amp
```

SHD and SSC datasets auto-download on first run. N-MNIST requires [tonic](https://tonic.readthedocs.io/). GSC requires manual download from [Google Speech Commands v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

### Deploy to Catalyst hardware

Trained models deploy to Catalyst N2 via the [Catalyst CLI](https://github.com/catalyst-neuromorphic/catalyst-neurocore), authenticate with a subscription or API key:

```bash
# Install the Catalyst CLI
npm install -g @catalyst-neuromorphic/cli

# Authenticate (opens browser, or use an API key)
catalyst login
# or: catalyst auth key cn_live_your_key_here

# Quantize and deploy a trained checkpoint
python shd/deploy.py --checkpoint shd_model.pt --threshold-hw 1000
```

Or use the Python client for programmatic access:

```bash
pip install catalyst-cloud
```

```python
import catalyst_cloud as cc
client = cc.Client("cn_live_your_key_here")
result = client.simulate(network_id="...", timesteps=1000)
```

### Parallel training (dual GPU)

```bash
# Terminal 1
python shd/train.py --neuron adlif --device cuda:0

# Terminal 2
python nmnist/train.py --device cuda:1
```

## Architecture

All models use surrogate gradient backpropagation through time (BPTT) with
a fast-sigmoid surrogate gradient. The key neuron models are:

- **LIF**: Leaky Integrate-and-Fire with multiplicative decay. Maps to CUBA hardware neuron via `decay_v = round(beta * 4096)`.
- **adLIF**: Adaptive LIF with Symplectic Euler discretization. Updates adaptation *before* threshold computation for richer temporal dynamics. Adaptation is training-only; only membrane decay deploys to hardware.

Weight quantization: `weight_hw = round(w_float * threshold_hw / threshold_float)`, clamped to int16 range.

## Project Structure

```
catalyst-benchmarks/
├── common/              Shared neuron models, training loop, deployment, augmentation
│   ├── neurons.py       LIF, AdaptiveLIF, ConvLIF, surrogate gradient
│   ├── training.py      train_epoch, evaluate, run_training
│   ├── deploy.py        quantize_weights, build_hardware_network
│   └── augmentation.py  event_drop, time_stretch, spatial_jitter
├── shd/                 Spiking Heidelberg Digits (20 classes, 700ch cochlea)
├── ssc/                 Spiking Speech Commands (35 classes, 700ch)
├── nmnist/              Neuromorphic MNIST (10 classes, 34x34 DVS)
├── gsc_kws/             Google Speech Commands keyword spotting (12 classes, S2S)
├── dvs_gesture/         DVS128 Gesture (11 classes, 128x128 DVS) — in progress
├── results.json         Machine-readable benchmark results
└── pyproject.toml       Dependencies and project metadata
```

## Catalyst Cloud

Run neuromorphic simulations without local hardware via [Catalyst Cloud](https://catalyst-neuromorphic.com/cloud):

```python
import catalyst_cloud as cc

client = cc.Client("cn_live_your_key_here")

net = client.create_network(
    populations=[
        {"label": "input", "size": 700, "params": {"threshold": 1000}},
        {"label": "hidden", "size": 512},
    ],
    connections=[
        {"source": "input", "target": "hidden", "topology": "random_sparse",
         "weight": 500, "p": 0.3},
    ],
)

result = client.simulate(network_id=net["network_id"], timesteps=250)
print(result["result"]["firing_rates"])
```

```bash
pip install catalyst-cloud
```

- **Free tier**: 10 jobs/day, 1,024 neurons, no credit card
- **Paid tiers**: Higher limits, priority compute

[Cloud Pricing](https://catalyst-neuromorphic.com/cloud/pricing) | [API Docs](https://catalyst-neuromorphic.com/cloud/docs)

## Hardware Deployment

Trained models deploy to Catalyst N2 FPGA hardware:

1. **Quantize**: Float32 weights → int16, membrane decay → 12-bit fixed-point
2. **Deploy**: Via Catalyst CLI (`catalyst deploy`) or USB/UART
3. **Evaluate**: Run test set through hardware, compare to float accuracy

Typical quantization loss: <1% accuracy.

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@misc{catalyst-benchmarks-2026,
  author = {Shulayev Barnes, Henry},
  title = {Catalyst Neuromorphic Benchmarks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/catalyst-neuromorphic/catalyst-benchmarks}
}

@misc{catalyst-n3-2026,
  author = {Shulayev Barnes, Henry},
  title = {Catalyst N3: A 128-Core Hybrid Neuromorphic Processor with Hardware Virtualisation, Per-Tile Learning, and Silicon Metaplasticity},
  year = {2026},
  url = {https://catalyst-neuromorphic.com/research}
}

@misc{catalyst-n2-2026,
  author = {Shulayev Barnes, Henry},
  title = {Catalyst N2: A 128-Core Configurable Neuromorphic Processor},
  year = {2026},
  doi = {10.5281/zenodo.18728256},
  url = {https://zenodo.org/records/18728256}
}

@misc{catalyst-n1-2026,
  author = {Shulayev Barnes, Henry},
  title = {Catalyst N1: A Neuromorphic Processing Architecture},
  year = {2026},
  doi = {10.5281/zenodo.18727094},
  url = {https://zenodo.org/records/18727094}
}
```

## License

MIT License. See [catalyst-neuromorphic.com](https://catalyst-neuromorphic.com) for hardware and SDK licensing.
