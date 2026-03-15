"""Deploy a trained SHD model to Catalyst hardware.

Loads a PyTorch checkpoint, quantizes weights to int16, evaluates
quantized accuracy, and builds a Neurocore SDK Network for FPGA.

Usage:
    python shd/deploy.py --checkpoint shd_model.pt --data-dir data/shd
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.deploy import (
    compute_hardware_params,
    build_hardware_network,
    run_quantized_inference,
)
from shd.loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES
from shd.train import SHDSNN


def main():
    parser = argparse.ArgumentParser(description="Deploy trained SHD model")
    parser.add_argument("--checkpoint", default="shd_model.pt")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--threshold-hw", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=4e-3)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', ckpt.get('args', {}))
    print(f"  Training accuracy: {ckpt['test_acc']*100:.1f}%")
    print(f"  Architecture: {N_CHANNELS}->{config.get('hidden', '?')}->{N_CLASSES}")

    print("\nLoading test dataset...")
    test_ds = SHDDataset(args.data_dir, "test", dt=args.dt)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    print(f"  {len(test_ds)} samples, {test_ds.n_bins} time bins")

    # Hardware parameter mapping
    print("\n--- Hardware parameter mapping ---")
    hw_params = compute_hardware_params(ckpt, args.threshold_hw)
    for k, v in sorted(hw_params.items()):
        print(f"  {k}: {v}")

    # Quantized inference
    print("\n--- Quantized inference ---")
    quant_acc = run_quantized_inference(SHDSNN, ckpt, test_loader,
                                        threshold_hw=args.threshold_hw)

    # Build SDK network
    print("\n--- SDK network ---")
    try:
        net, _ = build_hardware_network(ckpt, threshold_hw=args.threshold_hw)
        print("  Network built successfully")
    except ImportError:
        print("  neurocore not installed — skipping SDK network build")

    # Summary
    print("\n=== Results ===")
    print(f"  Float accuracy:     {ckpt['test_acc']*100:.1f}%")
    print(f"  Quantized accuracy: {quant_acc*100:.1f}%")
    gap = abs(ckpt['test_acc'] - quant_acc) * 100
    print(f"  Quantization loss:  {gap:.1f}%")


if __name__ == "__main__":
    main()
