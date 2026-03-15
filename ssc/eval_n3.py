"""Evaluate SSC model with N3 hardware features (post-training).

Usage:
    python ssc/eval_n3.py --checkpoint checkpoints/ssc_n3_tier1.pt --device cpu
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ssc.train import SSCSNN
from ssc.loader import SSCDataset, collate_fn
from common.deploy_n3 import (run_precision_sweep, run_factor_compressed_inference,
                               run_approximate_sweep)


def main():
    parser = argparse.ArgumentParser(description="N3 evaluation for SSC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/ssc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    test_ds = SSCDataset(args.data_dir, "test")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    def make_model():
        return SSCSNN(
            n_hidden1=config.get('hidden', 1024),
            n_hidden2=config.get('n_hidden2', 768),
            recurrent2=config.get('recurrent2', True),
        )

    print(f"=== N3 Evaluation: SSC ===")
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
