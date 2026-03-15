"""Train SNN on SHD with learnable synaptic delays (N4 benchmark).

v9: Integrates DelayedLinear from common/neurons.py into the proven
two-layer recurrent adLIF architecture (v8, 91.0%).

Key change: Input delays applied to full sequence BEFORE the time loop.
Each of the 700 input channels gets a learnable delay d_i in [0, max_delay].
This lets neurons attend to different temporal offsets — proven to give
+4% on SHD (DCLS-Delays: 95.07% with vanilla LIF, no recurrence).

Combined with adLIF + recurrence (already at 91%), target: 95-96%.

Hardware mapping: N4 per-synapse 8-bit delay registers (up to 255 timesteps).

Usage:
    python shd/train_delays.py --device cuda:0 --epochs 200 --amp
    python shd/train_delays.py --max-delay 60 --hidden 1024 --hidden2 512
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
from common.augmentation import event_drop

from shd.loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES


class SHDDelaysSNN(nn.Module):
    """Two-layer recurrent SNN with learnable input delays.

    700 -> DelayedLinear(max_delay) -> hidden1 (rec adLIF) ->
        hidden2 (rec adLIF) -> 20 (readout)

    The DelayedLinear applies per-channel delays to the full input sequence
    before the time loop. This gives each input channel a learnable temporal
    offset, creating a learned temporal receptive field.

    Inter-layer delays can be added in Phase 2 via delay buffer approach.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=512,
                 n_output=N_CLASSES, beta_out=0.9, threshold=1.0, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 target_rate=0.05, activity_lambda=0.01,
                 max_delay=60, inter_delay=0):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None
        self.inter_delay = inter_delay

        # Input projection WITH delays
        self.delay1 = DelayedLinear(n_input, n_hidden1, max_delay=max_delay,
                                     bias=False)

        # Recurrent connections (no delay — recurrence is instantaneous)
        self.fc_rec1 = nn.Linear(n_hidden1, n_hidden1, bias=False)

        # Layer 2: inter-layer + recurrence
        if inter_delay > 0:
            self.delay2 = DelayedLinear(n_hidden1, n_hidden2,
                                         max_delay=inter_delay, bias=False)
        else:
            self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
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
        nn.init.orthogonal_(self.fc_rec2.weight, gain=0.2)
        if inter_delay == 0:
            nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # --- DELAYS: apply to full input sequence BEFORE time loop ---
        x_delayed = self.delay1.apply_delays(x)  # (batch, T, n_input)

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

        if self.inter_delay > 0:
            # Two-pass: run layer 1 fully, apply inter-layer delays, then layer 2
            spk1_seq = []
            for t in range(T):
                I1 = self.delay1(x_delayed[:, t]) + self.fc_rec1(spk1_d)
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
                spk1_d = self.dropout(spk1) if self.training else spk1
                spike_count1 = spike_count1 + spk1
                spk1_seq.append(spk1_d)

            spk1_full = torch.stack(spk1_seq, dim=1)  # (batch, T, hidden1)
            spk1_delayed = self.delay2.apply_delays(spk1_full)

            for t in range(T):
                I2 = self.delay2(spk1_delayed[:, t]) + self.fc_rec2(spk2_d)
                v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
                spk2_d = self.dropout(spk2) if self.training else spk2
                I_out = self.fc_out(spk2_d)
                beta_o = self.lif_out.beta
                v_out = beta_o * v_out + (1.0 - beta_o) * I_out
                out_sum = out_sum + v_out
        else:
            # Single-pass: both layers in one loop (no inter-layer delays)
            for t in range(T):
                I1 = self.delay1(x_delayed[:, t]) + self.fc_rec1(spk1_d)
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
                spk1_d = self.dropout(spk1) if self.training else spk1
                spike_count1 = spike_count1 + spk1

                I2 = self.fc2(spk1_d) + self.fc_rec2(spk2_d)
                v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
                spk2_d = self.dropout(spk2) if self.training else spk2

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
        description="Train SHD with learnable delays (N4)")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--max-delay", type=int, default=60,
                        help="Max input delay in timesteps (60 = 240ms at 4ms bins)")
    parser.add_argument("--inter-delay", type=int, default=0,
                        help="Max inter-layer delay (0=disabled, >0=two-pass)")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--beta-out", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="shd_delays_model.pt")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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

    model = SHDDelaysSNN(
        n_hidden1=args.hidden,
        n_hidden2=args.hidden2,
        threshold=args.threshold,
        beta_out=args.beta_out,
        dropout=args.dropout,
        alpha_init=args.alpha_init,
        rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
        max_delay=args.max_delay,
        inter_delay=args.inter_delay,
    ).to(device)

    delay_str = f"input_delay={args.max_delay}"
    if args.inter_delay > 0:
        delay_str += f", inter_delay={args.inter_delay}"
    print(f"Model: {N_CHANNELS}->{args.hidden}(rec,adLIF)->"
          f"{args.hidden2}(rec,adLIF)->{N_CLASSES}")
    print(f"Delays: {delay_str}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    augment_fn = (lambda x: event_drop(x)) if args.event_drop else None

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_path': args.save,
        'benchmark': 'shd',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': 5,
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden': args.hidden,
            'hidden2': args.hidden2,
            'n_output': N_CLASSES,
            'max_delay': args.max_delay,
            'inter_delay': args.inter_delay,
            'threshold': args.threshold,
            'neuron_type': 'adlif',
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
