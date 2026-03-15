"""Train SSC with N4 KAN B-spline synapses.

Demonstrates N4 hardware synapse_mode=7 on the SSC benchmark.
Replaces the input linear layer with BSplineSynapseEfficient.

Architecture: 700 -> 1024 (KAN + rec adLIF) -> 512 (adLIF) -> 35

Usage:
    python ssc/train_kan.py --epochs 150 --device cuda:0 --amp
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.neurons import LIFNeuron, AdaptiveLIFNeuron
from common.training import run_training
from common.augmentation import event_drop, time_stretch
from kan_synapse.train import BSplineSynapseEfficient
from ssc.loader import SSCDataset, collate_fn, N_CHANNELS, N_CLASSES


class SSC_KAN_SNN(nn.Module):
    """SSC SNN with KAN B-spline synapses on input layer.

    700 -> 1024 (KAN + recurrent adLIF) -> 512 (adLIF) -> 35

    The KAN layer adds nonlinear weight functions (4 control points per
    connection) matching N4 hardware synapse_mode=7.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=512,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 beta_out=0.9, target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # KAN B-spline input (4x params vs linear, matches N4 hardware)
        self.kan_in = BSplineSynapseEfficient(n_input, n_hidden1)
        self.fc_rec = nn.Linear(n_hidden1, n_hidden1, bias=False)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.fc3 = nn.Linear(n_hidden2, n_output, bias=False)

        self.lif1 = AdaptiveLIFNeuron(n_hidden1, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)
        self.lif2 = AdaptiveLIFNeuron(n_hidden2, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden1, device=device)
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)

        spike_count1 = torch.zeros(batch, self.n_hidden1, device=device)

        a1 = torch.zeros(batch, self.n_hidden1, device=device)
        a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            # KAN synapse (nonlinear B-spline weight function)
            I1 = self.kan_in(x[:, t]) + self.fc_rec(spk1_d)

            v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count1 = spike_count1 + spk1

            I2 = self.fc2(spk1_d)
            v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            spk2_d = self.dropout(spk2) if self.training else spk2

            I3 = self.fc3(spk2_d)
            beta_out = self.lif_out.beta
            v_out = beta_out * v_out + (1.0 - beta_out) * I3
            out_sum = out_sum + v_out

        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count1 / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SSC with KAN synapses (N4)")
    parser.add_argument("--data-dir", default="data/ssc")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="checkpoints/ssc_n4_kan.pt")
    parser.add_argument("--activity-lambda", type=float, default=0.01)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--resume-weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("CATALYST N4 — SSC with KAN B-SPLINE SYNAPSES")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"N4 Feature: synapse_mode=7, De Casteljau B-spline evaluation")
    print(f"Each synapse has 4 learnable control points")
    print()

    print("Loading SSC dataset...")
    train_ds = SSCDataset(args.data_dir, "train", dt=args.dt)
    test_ds = SSCDataset(args.data_dir, "test", dt=args.dt)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {train_ds.n_bins} (dt={args.dt*1000:.1f}ms)")

    model = SSC_KAN_SNN(
        n_hidden1=args.hidden1,
        n_hidden2=args.hidden2,
        threshold=args.threshold,
        dropout=args.dropout,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
    ).to(device)

    if args.resume_weights:
        ckpt_data = torch.load(args.resume_weights, map_location=device, weights_only=False)
        # Load matching layers (fc_rec, fc2, fc3, lif1, lif2 transfer;
        # kan_in is new and won't match fc1)
        missing, unexpected = model.load_state_dict(
            ckpt_data['model_state_dict'], strict=False)
        print(f"Loaded weights from {args.resume_weights}")
        print(f"  Missing (new KAN layer): {len(missing)} keys")
        print(f"  Unexpected (old linear): {len(unexpected)} keys")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {N_CHANNELS}->{args.hidden1}(KAN+rec)->{args.hidden2}->{N_CLASSES}")
    print(f"Parameters: {n_params:,} (KAN adds 4x params on input layer)")

    def augment_fn(x):
        x = event_drop(x)
        x = time_stretch(x, factor_range=(0.9, 1.1))
        return x

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_path': args.save,
        'benchmark': 'ssc-kan',
        'augment_fn': augment_fn,
        'label_smoothing': 0.05,
        'use_amp': args.amp,
        'warmup_epochs': args.warmup_epochs,
        'gc_every': 5,
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden': args.hidden1,
            'n_hidden2': args.hidden2,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': 'adlif',
            'synapse_type': 'kan_bspline',
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
