"""Upload trained benchmark models to HuggingFace Hub.

Usage:
    # Upload SHD model
    python huggingface/upload.py --benchmark shd --checkpoint checkpoints/shd_adlif-baseline.pt

    # Upload all benchmarks (reads from results.json)
    python huggingface/upload.py --all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


BENCHMARK_CONFIG = {
    # ── N3 Results (Latest) ──
    'shd_n3': {
        'repo': 'Catalyst-Neuromorphic/shd-snn-benchmark',
        'task': 'audio-classification',
        'task_name': 'Spoken Digit Classification',
        'dataset': 'Spiking Heidelberg Digits (SHD)',
        'architecture': '700 → 1536 (recurrent adLIF) → 20',
        'description': 'Spiking Neural Network for spoken digit classification on SHD. Achieves 91.0% with adaptive LIF neurons.',
        'checkpoint': '../n3_results/checkpoints/shd_91.0.pt',
        'float_acc': 91.0,
        'quant_acc': None,
        'params': '3,470,484',
        'neuron_model': 'Adaptive Leaky Integrate-and-Fire (adLIF) with learnable per-neuron thresholds',
        'generation': 'N3',
    },
    'ssc_n3': {
        'repo': 'Catalyst-Neuromorphic/ssc-snn-benchmark',
        'task': 'audio-classification',
        'task_name': 'Spoken Command Classification',
        'dataset': 'Spiking Speech Commands (SSC)',
        'architecture': '700 → 1024 (recurrent adLIF) → 512 (adLIF) → 35',
        'description': 'Spiking Neural Network for spoken command classification on SSC. Achieves 76.4% with adaptive LIF neurons.',
        'checkpoint': '../n3_results/checkpoints/ssc_76.4.pt',
        'float_acc': 76.4,
        'quant_acc': None,
        'params': '2,313,763',
        'neuron_model': 'Adaptive Leaky Integrate-and-Fire (adLIF) with Symplectic Euler discretization',
        'generation': 'N3',
    },
    'nmnist_n3': {
        'repo': 'Catalyst-Neuromorphic/nmnist-snn-benchmark',
        'task': 'image-classification',
        'task_name': 'Neuromorphic Digit Classification',
        'dataset': 'Neuromorphic MNIST (N-MNIST)',
        'architecture': 'Conv2d → LIF → 10',
        'description': 'Convolutional Spiking Neural Network for neuromorphic digit classification on N-MNIST.',
        'checkpoint': '../n3_results/checkpoints/nmnist_99.2.pt',
        'float_acc': 99.2,
        'quant_acc': None,
        'params': '691,210',
        'neuron_model': 'Leaky Integrate-and-Fire (LIF) with convolutional feature extraction',
        'generation': 'N3',
    },
    # ── N2 Results ──
    'shd': {
        'repo': 'Catalyst-Neuromorphic/shd-snn-benchmark',
        'task': 'audio-classification',
        'task_name': 'Spoken Digit Classification',
        'dataset': 'Spiking Heidelberg Digits (SHD)',
        'architecture': '700 → 512 (recurrent adLIF) → 20',
        'description': 'Spiking Neural Network for spoken digit classification on SHD.',
        'checkpoint': 'shd_adlif_v7.pt',
        'float_acc': 84.5,
        'quant_acc': None,
        'params': '759,060',
        'neuron_model': 'Adaptive Leaky Integrate-and-Fire (adLIF) with Symplectic Euler discretization',
        'generation': 'N2',
    },
    'ssc': {
        'repo': 'Catalyst-Neuromorphic/ssc-snn-benchmark',
        'task': 'audio-classification',
        'task_name': 'Spoken Command Classification',
        'dataset': 'Spiking Speech Commands (SSC)',
        'architecture': '700 → 1024 (recurrent adLIF) → 512 (adLIF) → 35',
        'description': 'Spiking Neural Network for spoken command classification on SSC.',
        'checkpoint': 'ssc_adlif_v5.pt',
        'float_acc': 72.1,
        'quant_acc': 71.6,
        'params': '2,313,763',
        'neuron_model': 'Adaptive Leaky Integrate-and-Fire (adLIF) with Symplectic Euler discretization',
        'generation': 'N2',
    },
    'nmnist': {
        'repo': 'Catalyst-Neuromorphic/nmnist-snn-benchmark',
        'task': 'image-classification',
        'task_name': 'Neuromorphic Digit Classification',
        'dataset': 'Neuromorphic MNIST (N-MNIST)',
        'architecture': 'Conv2d(2,32,3) → Conv2d(32,64,3) → 9216 → 256 (LIF) → 10',
        'description': 'Convolutional Spiking Neural Network for neuromorphic digit classification on N-MNIST.',
        'checkpoint': 'nmnist_conv_v2b.pt',
        'float_acc': 99.23,
        'quant_acc': 99.1,
        'params': '466,273',
        'neuron_model': 'Leaky Integrate-and-Fire (LIF) with convolutional feature extraction',
        'generation': 'N2',
    },
    'gsc': {
        'repo': 'Catalyst-Neuromorphic/gsc-snn-benchmark',
        'task': 'audio-classification',
        'task_name': 'Keyword Spotting',
        'dataset': 'Google Speech Commands v2 (12-class)',
        'architecture': '40 → 512 (recurrent adLIF, spike-to-spike) → 12',
        'description': 'Spiking Neural Network for keyword spotting on Google Speech Commands using spike-to-spike delta modulation encoding.',
        'checkpoint': 'gsc_s2s_v11.pt',
        'float_acc': 88.0,
        'quant_acc': 87.5,
        'params': '290,828',
        'neuron_model': 'Adaptive Leaky Integrate-and-Fire (adLIF) with spike-to-spike delta encoding',
        'generation': 'N2',
    },
}


def generate_model_card(benchmark):
    """Generate a HuggingFace model card for a benchmark."""
    info = BENCHMARK_CONFIG[benchmark]
    float_acc = info['float_acc']
    quant_acc = info['quant_acc']
    params = info['params']
    generation = info.get('generation', 'N2')
    neuron_model = info.get('neuron_model', 'Adaptive Leaky Integrate-and-Fire (adLIF)')

    # Strip generation suffix for dataset/display names
    base_benchmark = benchmark.replace('_n3', '').replace('_n2', '')

    # Dataset tag mapping
    dataset_tags = {
        'shd': 'shd',
        'ssc': 'ssc',
        'nmnist': 'n-mnist',
        'gsc': 'google-speech-commands',
    }
    dataset_tag = dataset_tags.get(base_benchmark, base_benchmark.replace('_', '-'))

    # Build metrics YAML (skip quant if None)
    metrics_yaml = f"""          - name: Float Accuracy ({generation})
            type: accuracy
            value: {float_acc}"""
    if quant_acc is not None:
        metrics_yaml += f"""
          - name: Quantized Accuracy (int16)
            type: accuracy
            value: {quant_acc}"""

    # Build results table (skip quant row if None)
    results_rows = f"| Float accuracy | {float_acc}% |\n| Parameters | {params} |"
    if quant_acc is not None:
        results_rows = (
            f"| Float accuracy | {float_acc}% |\n"
            f"| Quantized accuracy (int16) | {quant_acc}% |\n"
            f"| Parameters | {params} |\n"
            f"| Quantization loss | {float_acc - quant_acc:.1f}% |"
        )

    # Reproduce command
    train_dir = base_benchmark
    train_cmd = f"python {train_dir}/train.py --device cuda:0"
    if generation == 'N3' and base_benchmark == 'shd':
        train_cmd = f"python {train_dir}/train.py --neuron adlif --hidden 1536 --epochs 200 --device cuda:0 --amp"

    # Paper links
    paper_links = """- **Benchmark repo**: [catalyst-neuromorphic/catalyst-benchmarks](https://github.com/catalyst-neuromorphic/catalyst-benchmarks)
- **Cloud API**: [catalyst-neuromorphic.com](https://catalyst-neuromorphic.com)"""
    if generation == 'N3':
        paper_links += """
- **N3 paper**: [Zenodo DOI 10.5281/zenodo.18881283](https://zenodo.org/records/18881283)"""
    paper_links += """
- **N2 paper**: [Zenodo DOI 10.5281/zenodo.18728256](https://zenodo.org/records/18728256)
- **N1 paper**: [Zenodo DOI 10.5281/zenodo.18727094](https://zenodo.org/records/18727094)"""

    return f"""---
language: en
license: mit
library_name: pytorch
tags:
  - spiking-neural-network
  - neuromorphic
  - surrogate-gradient
  - benchmark
  - catalyst
  - {base_benchmark}
datasets:
  - {dataset_tag}
metrics:
  - accuracy
model-index:
  - name: Catalyst {base_benchmark.upper()} SNN Benchmark ({generation})
    results:
      - task:
          type: {info['task']}
          name: {info['task_name']}
        dataset:
          name: {info['dataset']}
          type: {dataset_tag}
        metrics:
{metrics_yaml}
---

# Catalyst {base_benchmark.upper()} SNN Benchmark ({generation})

{info['description']}

## Model Description

- **Architecture ({generation})**: {info['architecture']}
- **Neuron model**: {neuron_model}
- **Training**: Surrogate gradient BPTT, fast-sigmoid surrogate (scale=25)
- **Hardware target**: Catalyst {generation} neuromorphic processor

## Results

| Metric | Value |
|--------|-------|
{results_rows}

## Reproduce

```bash
git clone https://github.com/catalyst-neuromorphic/catalyst-benchmarks.git
cd catalyst-benchmarks
pip install -e .
{train_cmd}
```

## Links

{paper_links}

## Citation

```bibtex
@misc{{catalyst-benchmarks-2026,
  author = {{Shulayev Barnes, Henry}},
  title = {{Catalyst Neuromorphic Benchmarks}},
  year = {{2026}},
  url = {{https://github.com/catalyst-neuromorphic/catalyst-benchmarks}}
}}
```
"""


def upload_model(benchmark, checkpoint_path, token=None):
    """Upload a model checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return False

    info = BENCHMARK_CONFIG[benchmark]
    repo_id = info['repo']

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        print(f"Repo {repo_id} ready")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload checkpoint
    if os.path.exists(checkpoint_path):
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="model.pt",
            repo_id=repo_id,
        )
        print(f"Uploaded {checkpoint_path} -> {repo_id}/model.pt")
    else:
        print(f"Warning: checkpoint {checkpoint_path} not found, skipping upload")

    # Generate and upload model card
    card = generate_model_card(benchmark)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"Uploaded model card -> {repo_id}/README.md")

    print(f"Done: https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload benchmark models to HuggingFace")
    parser.add_argument("--benchmark", choices=list(BENCHMARK_CONFIG.keys()))
    parser.add_argument("--checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--all", action="store_true", help="Upload all benchmarks from checkpoints/")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--card-only", action="store_true", help="Only generate model card, don't upload")
    args = parser.parse_args()

    token = args.token or os.environ.get('HF_TOKEN')

    if args.card_only and args.benchmark:
        card = generate_model_card(args.benchmark)
        print(card)
        return

    if args.all:
        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
        for bm, info in BENCHMARK_CONFIG.items():
            cp = os.path.join(checkpoints_dir, info['checkpoint'])
            if os.path.exists(cp):
                print(f"\n--- Uploading {bm} ---")
                upload_model(bm, cp, token)
            else:
                print(f"Skipping {bm}: no checkpoint at {cp}")
    elif args.benchmark and args.checkpoint:
        upload_model(args.benchmark, args.checkpoint, token)
    else:
        parser.error("Specify --benchmark + --checkpoint, or --all")


if __name__ == "__main__":
    main()
