"""NeuroBench benchmark harness integration.

Wraps trained Catalyst SNN models in the NeuroBench API for official
leaderboard submission. Supports algorithm and system track metrics.

Requires: pip install neurobench

Usage:
    python neurobench/submit.py --benchmark shd --checkpoint shd_model.pt
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def wrap_model_for_neurobench(model, benchmark_name):
    """Wrap a PyTorch SNN model for NeuroBench evaluation.

    Args:
        model: trained PyTorch model
        benchmark_name: benchmark identifier

    Returns:
        NeuroBenchModel wrapper
    """
    try:
        from neurobench.models import NeuroBenchModel
    except ImportError:
        raise ImportError(
            "neurobench required: pip install neurobench\n"
            "See https://neurobench.readthedocs.io/")

    return NeuroBenchModel(model)


def run_neurobench_evaluation(model, test_loader, benchmark_name,
                               preprocessors=None, postprocessors=None):
    """Run official NeuroBench benchmark evaluation.

    Args:
        model: NeuroBenchModel-wrapped model
        test_loader: test DataLoader
        benchmark_name: benchmark identifier
        preprocessors: optional list of preprocessors
        postprocessors: optional list of postprocessors

    Returns:
        dict with NeuroBench metrics
    """
    try:
        from neurobench.benchmarks import Benchmark
    except ImportError:
        raise ImportError("neurobench required: pip install neurobench")

    metrics = [
        "classification_accuracy",
        "activation_sparsity",
        "synaptic_operations",
        "number_of_parameters",
    ]

    benchmark = Benchmark(
        model,
        test_loader,
        preprocessors=preprocessors or [],
        postprocessors=postprocessors or [],
        metrics=metrics,
    )

    results = benchmark.run()
    return results


def main():
    parser = argparse.ArgumentParser(description="NeuroBench submission")
    parser.add_argument("--benchmark", required=True,
                        choices=["shd", "ssc", "nmnist", "dvs_gesture", "gsc_kws"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="neurobench_results.json")
    args = parser.parse_args()

    import torch

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', ckpt.get('args', {}))

    # Load benchmark-specific model and data
    if args.benchmark == 'shd':
        from shd.train import SHDSNN
        from shd.loader import SHDDataset, collate_fn
        data_dir = args.data_dir or "data/shd"
        model = SHDSNN(n_hidden=config.get('hidden', 512),
                       neuron_type=config.get('neuron_type', 'lif'),
                       dropout=0.0)
        test_ds = SHDDataset(data_dir, "test")
    elif args.benchmark == 'ssc':
        from ssc.train import SSCSNN
        from ssc.loader import SSCDataset, collate_fn
        data_dir = args.data_dir or "data/ssc"
        model = SSCSNN(n_hidden1=config.get('hidden', 1024), dropout=0.0,
                       neuron_type=config.get('neuron_type', 'adlif'))
        test_ds = SSCDataset(data_dir, "test")
    elif args.benchmark == 'nmnist':
        from nmnist.train import NMNISTSNN
        from nmnist.loader import NMNISTDataset, collate_fn
        data_dir = args.data_dir or "data/nmnist"
        model = NMNISTSNN(dropout=0.0,
                          neuron_type=config.get('neuron_type', 'adlif'))
        test_ds = NMNISTDataset(data_dir, train=False)
    elif args.benchmark == 'dvs_gesture':
        from dvs_gesture.train import DVSGestureSNN
        from dvs_gesture.loader import DVSGestureDataset, collate_fn
        data_dir = args.data_dir or "data/dvs_gesture"
        model = DVSGestureSNN(dropout=0.0,
                              neuron_type=config.get('neuron_type', 'adlif'))
        test_ds = DVSGestureDataset(data_dir, train=False)
    elif args.benchmark == 'gsc_kws':
        from gsc_kws.train import GSCSNN
        from gsc_kws.loader import GSCDataset, collate_fn
        data_dir = args.data_dir or "data/gsc"
        model = GSCSNN(n_hidden=config.get('hidden', 512), dropout=0.0,
                       neuron_type=config.get('neuron_type', 'adlif'))
        test_ds = GSCDataset(data_dir, split="testing")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # Wrap and evaluate
    nb_model = wrap_model_for_neurobench(model, args.benchmark)
    results = run_neurobench_evaluation(nb_model, test_loader, args.benchmark)

    # Save results
    print(f"\nNeuroBench Results for {args.benchmark}:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")

    with open(args.output, 'w') as f:
        json.dump({
            'benchmark': args.benchmark,
            'checkpoint': args.checkpoint,
            'results': {k: float(v) if hasattr(v, 'item') else v
                       for k, v in results.items()},
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
