"""Deploy a trained SSC model to Catalyst hardware.

Usage:
    python ssc/deploy.py --checkpoint ssc_model.pt --data-dir data/ssc
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.deploy import compute_hardware_params, run_quantized_inference
from ssc.loader import SSCDataset, collate_fn
from ssc.train import SSCSNN


def main():
    parser = argparse.ArgumentParser(description="Deploy trained SSC model")
    parser.add_argument("--checkpoint", default="ssc_model.pt")
    parser.add_argument("--data-dir", default="data/ssc")
    parser.add_argument("--threshold-hw", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=4e-3)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"  Training accuracy: {ckpt['test_acc']*100:.1f}%")

    print("\nLoading test dataset...")
    test_ds = SSCDataset(args.data_dir, "test", dt=args.dt)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print("\n--- Hardware parameters ---")
    hw_params = compute_hardware_params(ckpt, args.threshold_hw)
    for k, v in sorted(hw_params.items()):
        print(f"  {k}: {v}")

    print("\n--- Quantized inference ---")
    quant_acc = run_quantized_inference(SSCSNN, ckpt, test_loader,
                                        threshold_hw=args.threshold_hw)

    print("\n=== Results ===")
    print(f"  Float accuracy:     {ckpt['test_acc']*100:.1f}%")
    print(f"  Quantized accuracy: {quant_acc*100:.1f}%")
    print(f"  Quantization loss:  {abs(ckpt['test_acc'] - quant_acc)*100:.1f}%")


if __name__ == "__main__":
    main()
