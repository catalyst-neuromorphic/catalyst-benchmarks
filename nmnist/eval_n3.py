"""Evaluate N-MNIST model with N3 hardware features (post-training).

Usage:
    python nmnist/eval_n3.py --checkpoint checkpoints/nmnist_n3.pt --device cpu
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nmnist.train import ConvNMNISTSNN, DeepConvNMNISTSNN, NMNISTSNN
from nmnist.loader import NMNISTDataset, collate_fn
from common.deploy_n3 import (run_precision_sweep, run_factor_compressed_inference,
                               run_approximate_sweep)


def main():
    parser = argparse.ArgumentParser(description="N3 evaluation for N-MNIST")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/nmnist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    n3_deep = config.get('n3_deep', False)
    conv = config.get('conv', False)

    test_ds = NMNISTDataset(args.data_dir, train=False, flatten=not conv)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    def make_model():
        if n3_deep:
            return DeepConvNMNISTSNN(n_fc=config.get('fc_hidden', 256))
        elif conv:
            return ConvNMNISTSNN(n_fc=config.get('fc_hidden', 256))
        else:
            return NMNISTSNN(
                n_hidden1=config.get('hidden1', 512),
                n_hidden2=config.get('hidden2', 256),
            )

    arch = 'N3-DeepConv' if n3_deep else ('Conv' if conv else 'FC')
    print(f"=== N3 Evaluation: N-MNIST ({arch}) ===")
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
