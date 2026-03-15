"""Evaluate SHD model with N3 hardware features (post-training).

Runs precision sweep, FACTOR compression, and approximate computing
evaluation on a trained SHD checkpoint. No training needed.

Usage:
    python shd/eval_n3.py --checkpoint checkpoints/shd_n3_tier1.pt --device cpu
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shd.train import SHDSNNv8, SHDSNN
from shd.loader import SHDDataset, collate_fn
from common.deploy_n3 import (run_precision_sweep, run_factor_compressed_inference,
                               run_approximate_sweep)


def main():
    parser = argparse.ArgumentParser(description="N3 evaluation for SHD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint and determine model type
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    layers = config.get('layers', 1)

    test_ds = SHDDataset(args.data_dir, "test")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    if layers == 2:
        model_class = SHDSNNv8
        model_kwargs = {
            'n_hidden1': config.get('hidden', 1024),
            'n_hidden2': config.get('hidden2', 512),
        }
    else:
        model_class = SHDSNN
        model_kwargs = {'n_hidden': config.get('hidden', 1024)}

    print(f"=== N3 Evaluation: SHD ({layers}-layer) ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Float accuracy: {ckpt.get('test_acc', 0)*100:.2f}%")

    # 1. Precision sweep
    print("\n--- Precision Sweep ---")
    results = run_precision_sweep(
        lambda: _make_model(model_class, model_kwargs),
        args.checkpoint, test_loader, device=str(device))
    for prec, acc in results.items():
        print(f"  {prec:2d}-bit: {acc*100:.2f}%")

    # 2. FACTOR compression
    print("\n--- FACTOR Compression ---")
    for rank in [64, 32, 16]:
        acc, ratio = run_factor_compressed_inference(
            lambda: _make_model(model_class, model_kwargs),
            args.checkpoint, test_loader, rank=rank, device=str(device))
        print(f"  rank={rank:3d}: {acc*100:.2f}% (compression={ratio:.1%})")

    # 3. Approximate computing
    print("\n--- Approximate Computing ---")
    approx = run_approximate_sweep(
        lambda: _make_model(model_class, model_kwargs),
        args.checkpoint, test_loader, device=str(device))
    for quality, acc in approx.items():
        print(f"  quality={quality:.2f}: {acc*100:.2f}%")


def _make_model(model_class, kwargs):
    return model_class(**kwargs)


if __name__ == "__main__":
    main()
