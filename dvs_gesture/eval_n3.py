"""Evaluate DVS Gesture model with N3 hardware features (post-training).

Usage:
    python dvs_gesture/eval_n3.py --checkpoint checkpoints/dvs_n3.pt --device cpu
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dvs_gesture.train import ConvDVSGestureSNN, DVSGestureSNN, WTAConvDVSGestureSNN
from dvs_gesture.loader import DVSGestureDataset, collate_fn
from common.deploy_n3 import (run_precision_sweep, run_factor_compressed_inference,
                               run_approximate_sweep)


def main():
    parser = argparse.ArgumentParser(description="N3 evaluation for DVS Gesture")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/dvs_gesture")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    wta = config.get('wta', False)
    conv = config.get('conv', False)

    test_ds = DVSGestureDataset(args.data_dir, train=False, flatten=not conv)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    def make_model():
        if wta:
            return WTAConvDVSGestureSNN(
                n_fc=config.get('fc_hidden', 256),
                n_wta_groups=config.get('wta_groups', 8),
                wta_k=config.get('wta_k', 2),
            )
        elif conv:
            return ConvDVSGestureSNN(n_fc=config.get('fc_hidden', 256))
        else:
            return DVSGestureSNN(
                n_input=config.get('n_input', 2048),
                n_hidden1=config.get('hidden1', 512),
                n_hidden2=config.get('hidden2', 256),
            )

    arch = 'N3-WTA' if wta else ('Conv' if conv else 'FC')
    print(f"=== N3 Evaluation: DVS Gesture ({arch}) ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Float accuracy: {ckpt.get('test_acc', 0)*100:.2f}%")

    print("\n--- Precision Sweep ---")
    for prec, acc in run_precision_sweep(make_model, args.checkpoint, test_loader, device=str(device)).items():
        print(f"  {prec:2d}-bit: {acc*100:.2f}%")

    print("\n--- FACTOR Compression ---")
    for rank in [64, 32, 16]:
        acc, ratio = run_factor_compressed_inference(make_model, args.checkpoint, test_loader, rank=rank, device=str(device))
        print(f"  rank={rank:3d}: {acc*100:.2f}% (compression={ratio:.1%})")

    print("\n--- Approximate Computing ---")
    for quality, acc in run_approximate_sweep(make_model, args.checkpoint, test_loader, device=str(device)).items():
        print(f"  quality={quality:.2f}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
