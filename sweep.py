"""Hyperparameter sweep runner for all benchmarks.

Runs a grid of configurations for each benchmark, logging results.
Designed for dual-GPU parallel execution.

Usage:
    # Run all SHD sweeps on GPU 0
    python sweep.py --benchmark shd --device cuda:0

    # Run all benchmarks sequentially
    python sweep.py --all --device cuda:0

    # Run specific sweep by index
    python sweep.py --benchmark shd --index 3 --device cuda:0

    # Parallel: run SHD on GPU 0, N-MNIST on GPU 1
    # Terminal 1: CUDA_VISIBLE_DEVICES=0 python sweep.py --benchmark shd --device cuda:0
    # Terminal 2: CUDA_VISIBLE_DEVICES=1 python sweep.py --benchmark nmnist --device cuda:0
"""

import subprocess
import sys
import json
from datetime import datetime


# ---------------------------------------------------------------------------
# SHD sweep configs — Phase 2: push from 85.9% to 91%+
# ---------------------------------------------------------------------------
SHD_SWEEPS = [
    # Baseline adLIF (should already beat 85.9% LIF)
    {"neuron": "adlif", "hidden": 512, "epochs": 200, "dropout": 0.3,
     "label_smoothing": 0.0, "alpha_init": 0.90, "rho_init": 0.85,
     "beta_a_init": 0.05, "dt": 4e-3, "tag": "adlif-baseline"},

    # Higher alpha (slower membrane decay = longer memory)
    {"neuron": "adlif", "hidden": 512, "epochs": 200, "dropout": 0.3,
     "label_smoothing": 0.0, "alpha_init": 0.95, "rho_init": 0.85,
     "beta_a_init": 0.05, "dt": 4e-3, "tag": "adlif-alpha95"},

    # Alpha sweep with label smoothing
    {"neuron": "adlif", "hidden": 512, "epochs": 200, "dropout": 0.3,
     "label_smoothing": 0.05, "alpha_init": 0.93, "rho_init": 0.85,
     "beta_a_init": 0.05, "dt": 4e-3, "tag": "adlif-alpha93-ls05"},

    # Larger hidden with dropout
    {"neuron": "adlif", "hidden": 1024, "epochs": 200, "dropout": 0.4,
     "label_smoothing": 0.05, "alpha_init": 0.93, "rho_init": 0.85,
     "beta_a_init": 0.05, "dt": 4e-3, "tag": "adlif-h1024"},

    # Lower adaptation decay (faster adaptation)
    {"neuron": "adlif", "hidden": 512, "epochs": 200, "dropout": 0.3,
     "label_smoothing": 0.05, "alpha_init": 0.93, "rho_init": 0.80,
     "beta_a_init": 0.05, "dt": 4e-3, "tag": "adlif-rho80-ba20"},

    # Finer time bins (300 bins = dt=3.33ms)
    {"neuron": "adlif", "hidden": 512, "epochs": 200, "dropout": 0.3,
     "label_smoothing": 0.05, "alpha_init": 0.93, "rho_init": 0.85,
     "beta_a_init": 0.05, "dt": 3.33e-3, "tag": "adlif-dt333"},

    # Low dropout, high label smoothing
    {"neuron": "adlif", "hidden": 512, "epochs": 200, "dropout": 0.15,
     "label_smoothing": 0.1, "alpha_init": 0.93, "rho_init": 0.85,
     "beta_a_init": 0.05, "dt": 4e-3, "tag": "adlif-do15-ls10"},
]


# ---------------------------------------------------------------------------
# SHD N3 sweep configs — v8 two-layer recurrent, target 92%+
# ---------------------------------------------------------------------------
SHD_N3_SWEEPS = [
    # Two-layer baseline: 1024 rec -> 512 rec
    {"neuron": "adlif", "hidden": 1024, "hidden2": 512, "layers": 2,
     "epochs": 200, "dropout": 0.3, "label_smoothing": 0.05,
     "alpha_init": 0.95, "rho_init": 0.85, "beta_a_init": 0.05,
     "dt": 4e-3, "tag": "n3-2layer-baseline"},

    # Two-layer wider: 1280 rec -> 512 rec
    {"neuron": "adlif", "hidden": 1280, "hidden2": 512, "layers": 2,
     "epochs": 200, "dropout": 0.3, "label_smoothing": 0.05,
     "alpha_init": 0.95, "rho_init": 0.85, "beta_a_init": 0.05,
     "dt": 4e-3, "tag": "n3-2layer-h1280"},

    # Two-layer higher alpha (longer memory)
    {"neuron": "adlif", "hidden": 1024, "hidden2": 512, "layers": 2,
     "epochs": 200, "dropout": 0.3, "label_smoothing": 0.05,
     "alpha_init": 0.97, "rho_init": 0.85, "beta_a_init": 0.05,
     "dt": 4e-3, "tag": "n3-2layer-alpha97"},

    # Two-layer higher dropout
    {"neuron": "adlif", "hidden": 1024, "hidden2": 512, "layers": 2,
     "epochs": 200, "dropout": 0.4, "label_smoothing": 0.05,
     "alpha_init": 0.95, "rho_init": 0.85, "beta_a_init": 0.05,
     "dt": 4e-3, "tag": "n3-2layer-do40"},

    # Two-layer with AMP
    {"neuron": "adlif", "hidden": 1024, "hidden2": 256, "layers": 2,
     "epochs": 200, "dropout": 0.3, "label_smoothing": 0.05,
     "alpha_init": 0.95, "rho_init": 0.90, "beta_a_init": 0.05,
     "dt": 4e-3, "tag": "n3-2layer-h256-rho90"},

    # Two-layer low rho (faster adaptation)
    {"neuron": "adlif", "hidden": 1024, "hidden2": 512, "layers": 2,
     "epochs": 200, "dropout": 0.3, "label_smoothing": 0.05,
     "alpha_init": 0.93, "rho_init": 0.80, "beta_a_init": 0.05,
     "dt": 4e-3, "tag": "n3-2layer-alpha93-rho80"},
]


# ---------------------------------------------------------------------------
# N-MNIST sweep configs — Phase 3
# ---------------------------------------------------------------------------
NMNIST_SWEEPS = [
    # Default adLIF
    {"neuron": "adlif", "hidden1": 512, "hidden2": 256, "epochs": 50,
     "dropout": 0.2, "time_bins": 20, "tag": "adlif-default"},

    # More time bins
    {"neuron": "adlif", "hidden1": 512, "hidden2": 256, "epochs": 50,
     "dropout": 0.2, "time_bins": 30, "tag": "adlif-t30"},

    # LIF baseline comparison
    {"neuron": "lif", "hidden1": 512, "hidden2": 256, "epochs": 50,
     "dropout": 0.2, "time_bins": 20, "tag": "lif-default"},
]


# ---------------------------------------------------------------------------
# SSC sweep configs — Phase 4
# ---------------------------------------------------------------------------
SSC_SWEEPS = [
    # Default: large model, recurrent
    {"neuron": "adlif", "hidden1": 1024, "hidden2": 512, "epochs": 200,
     "dropout": 0.3, "label_smoothing": 0.05, "tag": "adlif-default"},

    # Higher dropout for regularization (larger dataset)
    {"neuron": "adlif", "hidden1": 1024, "hidden2": 512, "epochs": 200,
     "dropout": 0.4, "label_smoothing": 0.1, "tag": "adlif-do40-ls10"},

    # Smaller model (faster training, see if still beats Loihi)
    {"neuron": "adlif", "hidden1": 512, "hidden2": 256, "epochs": 200,
     "dropout": 0.3, "label_smoothing": 0.05, "tag": "adlif-small"},
]


# ---------------------------------------------------------------------------
# SSC N3 sweep configs — v5 two-layer recurrent, target 77%+
# ---------------------------------------------------------------------------
SSC_N3_SWEEPS = [
    # Two-layer recurrent: 1024 rec -> 768 rec
    {"neuron": "adlif", "hidden1": 1024, "hidden2": 768, "recurrent2": True,
     "epochs": 300, "dropout": 0.3, "label_smoothing": 0.05,
     "tag": "n3-rec2-h768"},

    # Two-layer recurrent: 1024 rec -> 512 rec, higher dropout
    {"neuron": "adlif", "hidden1": 1024, "hidden2": 512, "recurrent2": True,
     "epochs": 300, "dropout": 0.4, "label_smoothing": 0.05,
     "tag": "n3-rec2-h512-do40"},

    # Larger first layer with recurrent layer 2
    {"neuron": "adlif", "hidden1": 1280, "hidden2": 512, "recurrent2": True,
     "epochs": 300, "dropout": 0.3, "label_smoothing": 0.05,
     "tag": "n3-rec2-h1280"},
]


# ---------------------------------------------------------------------------
# DVS Gesture sweep configs — Phase 5
# ---------------------------------------------------------------------------
DVS_SWEEPS = [
    # Default
    {"neuron": "adlif", "hidden1": 512, "hidden2": 256, "epochs": 200,
     "dropout": 0.3, "time_bins": 20, "downsample": 32, "tag": "adlif-default"},

    # More time bins for temporal features
    {"neuron": "adlif", "hidden1": 512, "hidden2": 256, "epochs": 200,
     "dropout": 0.3, "time_bins": 30, "downsample": 32, "tag": "adlif-t30"},

    # Higher resolution input
    {"neuron": "adlif", "hidden1": 512, "hidden2": 256, "epochs": 200,
     "dropout": 0.3, "time_bins": 20, "downsample": 64, "tag": "adlif-ds64"},
]


# ---------------------------------------------------------------------------
# GSC KWS sweep configs — Phase 6
# ---------------------------------------------------------------------------
GSC_SWEEPS = [
    # Default
    {"neuron": "adlif", "hidden": 512, "epochs": 100, "dropout": 0.2,
     "time_bins": 100, "tag": "adlif-default"},

    # Larger model
    {"neuron": "adlif", "hidden": 1024, "epochs": 100, "dropout": 0.3,
     "time_bins": 100, "tag": "adlif-h1024"},

    # LIF baseline
    {"neuron": "lif", "hidden": 512, "epochs": 100, "dropout": 0.2,
     "time_bins": 100, "tag": "lif-default"},
]


ALL_SWEEPS = {
    'shd': SHD_SWEEPS,
    'shd_n3': SHD_N3_SWEEPS,
    'nmnist': NMNIST_SWEEPS,
    'ssc': SSC_SWEEPS,
    'ssc_n3': SSC_N3_SWEEPS,
    'dvs_gesture': DVS_SWEEPS,
    'gsc_kws': GSC_SWEEPS,
}


def build_command(benchmark, config, device):
    """Build the training command for a sweep config."""
    tag = config.get('tag', 'run')
    save_path = f"checkpoints/{benchmark}_{tag}.pt"

    if benchmark in ('shd', 'shd_n3'):
        cmd = [
            sys.executable, "shd/train.py",
            "--neuron", config['neuron'],
            "--hidden", str(config['hidden']),
            "--epochs", str(config['epochs']),
            "--dropout", str(config['dropout']),
            "--label-smoothing", str(config.get('label_smoothing', 0.0)),
            "--alpha-init", str(config.get('alpha_init', 0.90)),
            "--rho-init", str(config.get('rho_init', 0.85)),
            "--beta-a-init", str(config.get('beta_a_init', 0.05)),
            "--dt", str(config.get('dt', 4e-3)),
            "--save", save_path,
            "--device", device,
        ]
        if config.get('layers', 1) == 2:
            cmd.extend(["--layers", "2", "--hidden2", str(config.get('hidden2', 512))])
        if config['neuron'] == 'adlif':
            cmd.append("--event-drop")

    elif benchmark == 'nmnist':
        cmd = [
            sys.executable, "nmnist/train.py",
            "--neuron", config['neuron'],
            "--hidden1", str(config['hidden1']),
            "--hidden2", str(config['hidden2']),
            "--epochs", str(config['epochs']),
            "--dropout", str(config['dropout']),
            "--time-bins", str(config.get('time_bins', 20)),
            "--save", save_path,
            "--device", device,
            "--event-drop",
        ]

    elif benchmark in ('ssc', 'ssc_n3'):
        cmd = [
            sys.executable, "ssc/train.py",
            "--neuron", config['neuron'],
            "--hidden1", str(config['hidden1']),
            "--hidden2", str(config['hidden2']),
            "--epochs", str(config['epochs']),
            "--dropout", str(config['dropout']),
            "--label-smoothing", str(config.get('label_smoothing', 0.05)),
            "--save", save_path,
            "--device", device,
        ]
        if config.get('recurrent2', False):
            cmd.append("--recurrent2")
        cmd.append("--amp")

    elif benchmark == 'dvs_gesture':
        cmd = [
            sys.executable, "dvs_gesture/train.py",
            "--neuron", config['neuron'],
            "--hidden1", str(config['hidden1']),
            "--hidden2", str(config['hidden2']),
            "--epochs", str(config['epochs']),
            "--dropout", str(config['dropout']),
            "--time-bins", str(config.get('time_bins', 20)),
            "--downsample", str(config.get('downsample', 32)),
            "--save", save_path,
            "--device", device,
        ]

    elif benchmark == 'gsc_kws':
        cmd = [
            sys.executable, "gsc_kws/train.py",
            "--neuron", config['neuron'],
            "--hidden", str(config['hidden']),
            "--epochs", str(config['epochs']),
            "--dropout", str(config['dropout']),
            "--time-bins", str(config.get('time_bins', 100)),
            "--save", save_path,
            "--device", device,
        ]

    return cmd, save_path


def run_sweep(benchmark, configs, device, start_index=0):
    """Run a sequence of sweep configurations."""
    import os
    os.makedirs("checkpoints", exist_ok=True)

    results = []
    for i, config in enumerate(configs):
        if i < start_index:
            continue

        tag = config.get('tag', f'run{i}')
        print(f"\n{'='*60}")
        print(f"[{benchmark}] Sweep {i+1}/{len(configs)}: {tag}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        cmd, save_path = build_command(benchmark, config, device)
        print(f"Command: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd, check=True)
            results.append({
                'benchmark': benchmark,
                'tag': tag,
                'config': config,
                'save_path': save_path,
                'status': 'success',
            })
        except subprocess.CalledProcessError as e:
            print(f"\nFAILED: {tag} (exit code {e.returncode})")
            results.append({
                'benchmark': benchmark,
                'tag': tag,
                'config': config,
                'status': 'failed',
                'error': str(e),
            })

    # Save sweep results
    sweep_file = f"checkpoints/{benchmark}_sweep_results.json"
    with open(sweep_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {sweep_file}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    parser.add_argument("--benchmark", choices=list(ALL_SWEEPS.keys()))
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--index", type=int, default=0,
                        help="Start from this sweep index")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list", action="store_true",
                        help="List sweep configs without running")
    args = parser.parse_args()

    if args.list:
        for name, sweeps in ALL_SWEEPS.items():
            print(f"\n{name.upper()} ({len(sweeps)} configs):")
            for i, s in enumerate(sweeps):
                print(f"  [{i}] {s.get('tag', '?')}: {s}")
        return

    if args.all:
        benchmarks = list(ALL_SWEEPS.keys())
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        parser.error("Specify --benchmark or --all")

    for benchmark in benchmarks:
        configs = ALL_SWEEPS[benchmark]
        print(f"\n{'#'*60}")
        print(f"# Starting {benchmark.upper()} sweep ({len(configs)} configs)")
        print(f"# Device: {args.device}")
        print(f"{'#'*60}")
        run_sweep(benchmark, configs, args.device, args.index)


if __name__ == "__main__":
    main()
