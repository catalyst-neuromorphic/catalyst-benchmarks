"""Deploy a trained N-MNIST model to Catalyst hardware.

Usage:
    python nmnist/deploy.py --checkpoint nmnist_model.pt --data-dir data/nmnist
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.deploy import compute_hardware_params, build_hardware_network, run_quantized_inference
from nmnist.loader import NMNISTDataset, collate_fn
from nmnist.train import NMNISTSNN


def main():
    parser = argparse.ArgumentParser(description="Deploy trained N-MNIST model")
    parser.add_argument("--checkpoint", default="nmnist_model.pt")
    parser.add_argument("--data-dir", default="data/nmnist")
    parser.add_argument("--threshold-hw", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"  Training accuracy: {ckpt['test_acc']*100:.1f}%")

    print("\nLoading test dataset...")
    test_ds = NMNISTDataset(args.data_dir, train=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print("\n--- Hardware parameters ---")
    hw_params = compute_hardware_params(ckpt, args.threshold_hw)
    for k, v in sorted(hw_params.items()):
        print(f"  {k}: {v}")

    print("\n--- Quantized inference ---")
    quant_acc = run_quantized_inference(NMNISTSNN, ckpt, test_loader,
                                        threshold_hw=args.threshold_hw)

    print("\n=== Results ===")
    print(f"  Float accuracy:     {ckpt['test_acc']*100:.1f}%")
    print(f"  Quantized accuracy: {quant_acc*100:.1f}%")
    print(f"  Quantization loss:  {abs(ckpt['test_acc'] - quant_acc)*100:.1f}%")


if __name__ == "__main__":
    main()
