"""Evaluate GSC KWS model with N3 hardware features (post-training).

Usage:
    python gsc_kws/eval_n3.py --checkpoint checkpoints/gsc_n3.pt --device cpu
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gsc_kws.train import GSCSNN, ConvGSCSNN, TwoLayerGSCSNN, HybridGSCSNN
from gsc_kws.loader import GSCDataset, collate_fn
from common.deploy_n3 import (run_precision_sweep, run_factor_compressed_inference,
                               run_approximate_sweep)


def main():
    parser = argparse.ArgumentParser(description="N3 evaluation for GSC KWS")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/gsc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    # Determine model type from config
    n3_hybrid = config.get('n3_hybrid', False)
    two_layer = config.get('two_layer', False)
    conv = config.get('conv', False)

    # Use matching encoding for data loading
    encoding = 'n3' if n3_hybrid else 's2s'
    test_ds = GSCDataset(args.data_dir, "test", encoding=encoding)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    def make_model():
        if n3_hybrid:
            return HybridGSCSNN(
                n_mels=config.get('n_input', 40),
                n_proj=config.get('n_proj', 256),
                n_hidden=config.get('hidden', 512),
            )
        elif two_layer:
            return TwoLayerGSCSNN(
                n_input=config.get('n_input', 40),
                n_hidden1=config.get('hidden', 512),
                n_hidden2=config.get('hidden2', 256),
            )
        elif conv:
            return ConvGSCSNN(n_fc=config.get('fc_hidden', 256))
        else:
            return GSCSNN(
                n_input=config.get('n_input', 40),
                n_hidden=config.get('hidden', 512),
            )

    arch = 'N3-Hybrid' if n3_hybrid else ('TwoLayer' if two_layer else ('Conv' if conv else 'FC'))
    print(f"=== N3 Evaluation: GSC KWS ({arch}) ===")
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
