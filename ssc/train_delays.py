"""Train SNN on SSC with learnable synaptic delays (N4 benchmark).

v6: Integrates DelayedLinear into the proven two-layer recurrent adLIF
architecture (v5, 76.4%). SSC has longer utterances than SHD so we use
larger max_delay (80 timesteps = 320ms at 4ms bins).

Key change: Input delays applied to full 700-channel cochlear sequence.
DCLS-Delays alone gets 80.69% on SSC with vanilla LIF, no recurrence.
Combined with adLIF + recurrence + activity reg, target: 82-84%.

Usage:
    python ssc/train_delays.py --device cuda:1 --epochs 300 --amp
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.neurons import LIFNeuron, AdaptiveLIFNeuron, DelayedLinear
from common.training import run_training
from common.augmentation import event_drop, time_stretch

from ssc.loader import SSCDataset, collate_fn, N_CHANNELS, N_CLASSES


class SSCDelaysSNN(nn.Module):
    """Two-layer recurrent SNN with learnable input delays for SSC.

    700 -> DelayedLinear(max_delay=80) -> 1024 (rec adLIF) ->
        768 (rec adLIF) -> 35 (readout)
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=768,
                 n_output=N_CLASSES, beta_out=0.9, threshold=1.0, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 target_rate=0.05, activity_lambda=0.01,
                 max_delay=80, recurrent2=True):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.recurrent2 = recurrent2
        self.aux_loss = None

        # Input with delays
        self.delay1 = DelayedLinear(n_input, n_hidden1, max_delay=max_delay,
                                     bias=False)

        # Recurrent connections
        self.fc_rec1 = nn.Linear(n_hidden1, n_hidden1, bias=False)

        # Layer 2
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        if recurrent2:
            self.fc_rec2 = nn.Linear(n_hidden2, n_hidden2, bias=False)

        # Readout
        self.fc_out = nn.Linear(n_hidden2, n_output, bias=False)

        # Neurons
        self.lif1 = AdaptiveLIFNeuron(
            n_hidden1, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)
        self.lif2 = AdaptiveLIFNeuron(
            n_hidden2, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        # Init
        nn.init.orthogonal_(self.fc_rec1.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        if recurrent2:
            nn.init.orthogonal_(self.fc_rec2.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # Apply input delays to full sequence
        x_delayed = self.delay1.apply_delays(x)

        # Init states
        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden1, device=device)
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)
        spk2_d = torch.zeros(batch, self.n_hidden2, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count1 = torch.zeros(batch, self.n_hidden1, device=device)
        a1 = torch.zeros(batch, self.n_hidden1, device=device)
        a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            # Layer 1: delayed input + recurrence
            I1 = self.delay1(x_delayed[:, t]) + self.fc_rec1(spk1_d)
            v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count1 = spike_count1 + spk1

            # Layer 2
            I2 = self.fc2(spk1_d)
            if self.recurrent2:
                I2 = I2 + self.fc_rec2(spk2_d)
            v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            spk2_d = self.dropout(spk2) if self.training else spk2

            # Readout
            I_out = self.fc_out(spk2_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count1 / T
            self.aux_loss = self.activity_lambda * (
                (mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(
        description="Train SSC with learnable delays (N4)")
    parser.add_argument("--data-dir", default="data/ssc")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=768)
    parser.add_argument("--max-delay", type=int, default=80,
                        help="Max input delay in timesteps (80 = 320ms)")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="ssc_delays_v6.pt")
    parser.add_argument("--alpha-init", type=float, default=0.95)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--event-drop", action="store_true", default=True)
    parser.add_argument("--time-stretch", action="store_true", default=True)
    parser.add_argument("--recurrent2", action="store_true", default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--activity-lambda", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

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

    model = SSCDelaysSNN(
        n_hidden1=args.hidden1,
        n_hidden2=args.hidden2,
        threshold=args.threshold,
        dropout=args.dropout,
        alpha_init=args.alpha_init,
        rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
        max_delay=args.max_delay,
        recurrent2=args.recurrent2,
    ).to(device)

    rec2_str = "+rec2" if args.recurrent2 else ""
    print(f"Model: {N_CHANNELS}->{args.hidden1}(rec,adLIF)->"
          f"{args.hidden2}(rec,adLIF){rec2_str}->{N_CLASSES}")
    print(f"Delays: input_delay={args.max_delay}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    def augment_fn(x):
        if args.event_drop:
            x = event_drop(x)
        if args.time_stretch:
            x = time_stretch(x, factor_range=(0.9, 1.1))
        return x

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_path': args.save,
        'benchmark': 'ssc',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': args.warmup_epochs,
        'gc_every': 5,
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'n_output': N_CLASSES,
            'max_delay': args.max_delay,
            'threshold': args.threshold,
            'neuron_type': 'adlif',
            'dropout': args.dropout,
            'recurrent2': args.recurrent2,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
