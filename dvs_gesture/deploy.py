"""Deploy a trained DVS Gesture model to Catalyst hardware.

Usage:
    python dvs_gesture/deploy.py --checkpoint dvs_gesture_model.pt
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.deploy import compute_hardware_params, run_quantized_inference
from dvs_gesture.loader import DVSGestureDataset, collate_fn
from dvs_gesture.train import DVSGestureSNN


def main():
    parser = argparse.ArgumentParser(description="Deploy trained DVS Gesture model")
    parser.add_argument("--checkpoint", default="dvs_gesture_model.pt")
    parser.add_argument("--data-dir", default="data/dvs_gesture")
    parser.add_argument("--threshold-hw", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"  Training accuracy: {ckpt['test_acc']*100:.1f}%")

    print("\nLoading test dataset...")
    test_ds = DVSGestureDataset(args.data_dir, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print("\n--- Hardware parameters ---")
    hw_params = compute_hardware_params(ckpt, args.threshold_hw)
    for k, v in sorted(hw_params.items()):
        print(f"  {k}: {v}")

    print("\n--- Quantized inference ---")
    quant_acc = run_quantized_inference(DVSGestureSNN, ckpt, test_loader,
                                        threshold_hw=args.threshold_hw)

    print("\n=== Results ===")
    print(f"  Float accuracy:     {ckpt['test_acc']*100:.1f}%")
    print(f"  Quantized accuracy: {quant_acc*100:.1f}%")
    print(f"  Quantization loss:  {abs(ckpt['test_acc'] - quant_acc)*100:.1f}%")


if __name__ == "__main__":
    main()
