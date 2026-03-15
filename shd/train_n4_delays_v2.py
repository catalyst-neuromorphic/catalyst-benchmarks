"""Train N4 SHD with learnable delays + adLIF — FIXED version.

Key fixes over v1:
1. Separate learning rate for delay parameters (0.1x main LR)
2. Uniform delay initialization (not randn near zero)
3. Matches N3 v3 hyperparameters (the 91.0% config) as baseline
4. Optional multi-tap delays for richer temporal mixing

Usage:
    # Match N3 capacity + add delays (best shot at beating 91%)
    python shd/train_n4_delays_v2.py --hidden 1536 --device cuda:1

    # Multi-tap mode (K delay taps per channel, learned mixing)
    python shd/train_n4_delays_v2.py --hidden 1024 --multi-tap 4 --device cuda:1
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.neurons import LIFNeuron, AdaptiveLIFNeuron, DelayedLinear, surrogate_spike
from common.training import run_training
from common.augmentation import event_drop

from shd.loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES


class MultiTapDelayLayer(nn.Module):
    """Multi-tap delay: K fixed delay taps per input, with learned mixing.

    Creates K copies of each input channel at evenly-spaced delays.
    A Linear layer then selects which delay offsets matter for each output.
    This is effectively per-synapse delay selection via weight learning.

    Input: (batch, T, C) -> (batch, T, C*K) -> Linear(C*K, out)
    """

    def __init__(self, in_features, out_features, n_taps=4, max_delay=30, bias=False):
        super().__init__()
        self.in_features = in_features
        self.n_taps = n_taps
        self.max_delay = max_delay

        # Fixed delay values: evenly spaced from 0 to max_delay
        delays = torch.linspace(0, max_delay, n_taps)
        self.register_buffer('delays', delays)

        # Linear mixing over tapped inputs
        self.linear = nn.Linear(in_features * n_taps, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def apply_taps(self, x_seq):
        """Create K delayed copies of input sequence.

        Args:
            x_seq: (batch, T, C)

        Returns:
            (batch, T, C*K) — K delayed versions concatenated
        """
        batch, T, C = x_seq.shape
        max_d = self.max_delay

        # Zero-pad time dimension
        x_pad = F.pad(x_seq, (0, 0, max_d, 0))  # (batch, T+max_d, C)

        tapped = []
        for k in range(self.n_taps):
            d = self.delays[k]
            d_floor = int(d)
            d_frac = d - d_floor

            t_idx = torch.arange(T, device=x_seq.device)
            idx_f = (t_idx + max_d - d_floor).clamp(0, T + max_d - 1)
            idx_c = (idx_f - 1).clamp(0, T + max_d - 1)

            x_f = x_pad[:, idx_f, :]  # (batch, T, C)
            x_c = x_pad[:, idx_c, :]
            tapped.append(x_f + d_frac * (x_c - x_f))

        return torch.cat(tapped, dim=-1)  # (batch, T, C*K)

    def forward(self, x):
        """Standard linear on pre-tapped input. x: (batch, C*K)."""
        return self.linear(x)


class SHDDelaysV2(nn.Module):
    """N4 SHD model with proper delay training.

    Uses either per-channel delays (DelayedLinear) or multi-tap delays.
    Matched to N3 v3 config (hidden=1536, single-layer recurrent adLIF).
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=1536,
                 n_output=N_CLASSES, max_delay=30, beta_out=0.9,
                 threshold=1.0, dropout=0.3, neuron_type='adlif',
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 target_rate=0.05, activity_lambda=0.01,
                 multi_tap=0):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None
        self.multi_tap = multi_tap

        if multi_tap > 0:
            self.delay_layer = MultiTapDelayLayer(
                n_input, n_hidden, n_taps=multi_tap,
                max_delay=max_delay, bias=False)
        else:
            self.delay_layer = DelayedLinear(
                n_input, n_hidden, max_delay=max_delay, bias=False)

        # Recurrent
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)

        # Readout
        self.fc_out = nn.Linear(n_hidden, n_output, bias=False)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

        if neuron_type == 'adlif':
            self.lif1 = AdaptiveLIFNeuron(
                n_hidden, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden, beta_init=0.95,
                                   threshold=threshold, learn_beta=True)

        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # Apply delays to input sequence
        if self.multi_tap > 0:
            x_delayed = self.delay_layer.apply_taps(x)
        else:
            x_delayed = self.delay_layer.apply_delays(x)

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            if self.multi_tap > 0:
                I1 = self.delay_layer(x_delayed[:, t]) + self.fc_rec(spk1_d)
            else:
                I1 = self.delay_layer(x_delayed[:, t]) + self.fc_rec(spk1_d)

            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count = spike_count + spk1

            I_out = self.fc_out(spk1_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train N4 SHD v2 — fixed delays")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--delay-lr-scale", type=float, default=0.1,
                        help="Delay param LR = lr * delay_lr_scale (default 0.1)")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--max-delay", type=int, default=20,
                        help="Max delay in timesteps (default 20, ~80ms)")
    parser.add_argument("--multi-tap", type=int, default=0,
                        help="Multi-tap mode: K taps per input (0=use DelayedLinear)")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--beta-out", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default=None)
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--alpha-init", type=float, default=0.95)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--event-drop", action="store_true", default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--activity-lambda", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    args = parser.parse_args()

    if args.save is None:
        tag = f"h{args.hidden}_d{args.max_delay}"
        if args.multi_tap > 0:
            tag += f"_mt{args.multi_tap}"
        args.save = f"checkpoints/shd_n4v2_{tag}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading SHD dataset...")
    train_ds = SHDDataset(args.data_dir, "train", dt=args.dt)
    test_ds = SHDDataset(args.data_dir, "test", dt=args.dt)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {train_ds.n_bins} (dt={args.dt*1000:.1f}ms)")

    model = SHDDelaysV2(
        n_hidden=args.hidden,
        max_delay=args.max_delay,
        threshold=args.threshold,
        beta_out=args.beta_out,
        dropout=args.dropout,
        neuron_type=args.neuron,
        alpha_init=args.alpha_init,
        rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
        multi_tap=args.multi_tap,
    ).to(device)

    # Separate delay_raw params from weight params for different LR
    delay_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'delay_raw' in name:
            delay_params.append(param)
        else:
            other_params.append(param)

    delay_lr = args.lr * args.delay_lr_scale
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': delay_params, 'lr': delay_lr, 'weight_decay': 0.0},
    ])

    n_params = sum(p.numel() for p in model.parameters())
    n_delay = sum(p.numel() for p in delay_params)
    mode_str = f"multi-tap K={args.multi_tap}" if args.multi_tap > 0 else "per-channel"
    print(f"Model: {N_CHANNELS}->{args.hidden}(delay+rec)->{N_CLASSES}")
    print(f"  Neuron: {args.neuron.upper()}, delay_mode={mode_str}, max_delay={args.max_delay}")
    print(f"  Parameters: {n_params:,} ({n_delay} delay params)")
    print(f"  LR: weights={args.lr}, delays={delay_lr}")
    print(f"  Dropout={args.dropout}, label_smoothing={args.label_smoothing}")

    augment_fn = (lambda x: event_drop(x)) if args.event_drop else None

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': optimizer,
        'save_path': args.save,
        'benchmark': 'shd',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': 0,
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden': args.hidden,
            'max_delay': args.max_delay,
            'multi_tap': args.multi_tap,
            'delay_lr_scale': args.delay_lr_scale,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': args.neuron,
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
