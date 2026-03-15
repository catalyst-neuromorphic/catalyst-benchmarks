"""Train SNN on SHD with N4 learnable per-synapse delays + adLIF.

The key N4 advantage: per-synapse delays let each neuron attend to different
temporal offsets. Combined with adLIF adaptive thresholds, this provides
both temporal selectivity and short-term memory.

Reference: Hammouamri et al. (ICLR 2024) achieved 95.07% on SHD with
delays + vanilla LIF. We add adLIF on top for expected 95%+.

Hardware mapping: N4 has 8-bit per-synapse delay fields in all synapse
formats (Full 78-bit, Inference 53-bit). Delay queue per core: 4,096 slots,
configurable up to 255 timesteps.

Usage:
    python shd/train_n4_delays.py --epochs 200 --device cuda:0
    python shd/train_n4_delays.py --neuron lif --epochs 200  # delays + LIF only
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.neurons import LIFNeuron, AdaptiveLIFNeuron, DelayedLinear, surrogate_spike
from common.training import run_training
from common.augmentation import event_drop

from shd.loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES


class SHDDelaysSNN(nn.Module):
    """N4-class SHD model with learnable per-synapse delays.

    Architecture:
      x -> DelayedLinear(700, hidden) -> rec adLIF -> fc_out -> readout

    Delays are applied to the INPUT SEQUENCE per-channel (Hammouamri et al.
    2024). The delayed inputs feed into a recurrent adLIF layer. This
    combines N4's delay hardware with N2's adaptive thresholds.

    Key: Delays SHIFT the temporal alignment of each input channel.
    For speech, this means each neuron learns to listen to different
    phoneme timing offsets — a form of learned temporal receptive field.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=1024,
                 n_output=N_CLASSES, max_delay=30, beta_out=0.9,
                 threshold=1.0, dropout=0.3, neuron_type='adlif',
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # Delayed input projection (N4 hardware feature)
        self.delayed_fc1 = DelayedLinear(n_input, n_hidden, max_delay=max_delay, bias=False)

        # Recurrent connections (standard, no delays on recurrence)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)

        # Readout
        self.fc_out = nn.Linear(n_hidden, n_output, bias=False)

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

        # Initialization (match DCLS paper: Xavier for delayed, orthogonal for recurrence)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # Apply per-channel delays to ENTIRE input sequence at once
        # This is the key N4 operation — each input channel is shifted by
        # its learned delay value before feeding into the network
        x_delayed = self.delayed_fc1.apply_delays(x)

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            # Input uses delayed sequence + standard recurrence
            I1 = self.delayed_fc1(x_delayed[:, t]) + self.fc_rec(spk1_d)

            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count = spike_count + spk1

            # Non-spiking readout
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


class SHDDelaysSNNv2(nn.Module):
    """N4-class 2-layer SHD model with delays on BOTH layers.

    Architecture:
      x -> DelayedLinear(700, h1) -> rec adLIF
        -> DelayedLinear(h1, h2) -> rec adLIF -> fc_out -> readout

    Delays on both input and inter-layer connections. This maps to N4's
    per-synapse delay hardware on every core interconnect.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=512,
                 n_output=N_CLASSES, max_delay=30, beta_out=0.9,
                 threshold=1.0, dropout=0.3, neuron_type='adlif',
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # Layer 1: delayed input + recurrence
        self.delayed_fc1 = DelayedLinear(n_input, n_hidden1, max_delay=max_delay, bias=False)
        self.fc_rec1 = nn.Linear(n_hidden1, n_hidden1, bias=False)

        # Layer 2: delayed inter-layer (spikes accumulated then delayed)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.fc_rec2 = nn.Linear(n_hidden2, n_hidden2, bias=False)

        # Readout
        self.fc_out = nn.Linear(n_hidden2, n_output, bias=False)

        if neuron_type == 'adlif':
            self.lif1 = AdaptiveLIFNeuron(
                n_hidden1, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
            self.lif2 = AdaptiveLIFNeuron(
                n_hidden2, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden1, beta_init=0.95,
                                   threshold=threshold, learn_beta=True)
            self.lif2 = LIFNeuron(n_hidden2, beta_init=0.95,
                                   threshold=threshold, learn_beta=True)

        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        nn.init.orthogonal_(self.fc_rec1.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec2.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # Apply delays to input sequence
        x_delayed = self.delayed_fc1.apply_delays(x)

        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden1, device=device)
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)
        spk2_d = torch.zeros(batch, self.n_hidden2, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count = torch.zeros(batch, self.n_hidden1, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden1, device=device)
            a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            # Layer 1: delayed input + recurrence
            I1 = self.delayed_fc1(x_delayed[:, t]) + self.fc_rec1(spk1_d)
            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count = spike_count + spk1

            # Layer 2: standard inter-layer + recurrence
            I2 = self.fc2(spk1_d) + self.fc_rec2(spk2_d)
            if self.neuron_type == 'adlif':
                v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            else:
                v2, spk2 = self.lif2(I2, v2)
            spk2_d = self.dropout(spk2) if self.training else spk2

            # Non-spiking readout
            I_out = self.fc_out(spk2_d)
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
    parser = argparse.ArgumentParser(description="Train N4 SHD with learnable delays")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512,
                        help="Second layer hidden size (only used with --layers 2)")
    parser.add_argument("--layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--max-delay", type=int, default=30,
                        help="Maximum learnable delay in timesteps")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--beta-out", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="checkpoints/shd_n4_delays.pt")
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

    if args.layers == 2:
        model = SHDDelaysSNNv2(
            n_hidden1=args.hidden,
            n_hidden2=args.hidden2,
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
        ).to(device)
        arch_str = (f"{N_CHANNELS}->{args.hidden}(delay+rec)->"
                    f"{args.hidden2}(rec)->{N_CLASSES}")
    else:
        model = SHDDelaysSNN(
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
        ).to(device)
        arch_str = f"{N_CHANNELS}->{args.hidden}(delay+rec)->{N_CLASSES}"

    n_params = sum(p.numel() for p in model.parameters())
    delay_params = model.delayed_fc1.delay_raw.numel()
    print(f"Model: {arch_str}")
    print(f"  Neuron: {args.neuron.upper()}, max_delay={args.max_delay}")
    print(f"  Parameters: {n_params:,} ({delay_params} delay params)")
    print(f"  Dropout={args.dropout}, event_drop={args.event_drop}")

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
        'warmup_epochs': 0,
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden': args.hidden,
            'hidden2': args.hidden2 if args.layers == 2 else None,
            'layers': args.layers,
            'max_delay': args.max_delay,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': args.neuron,
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
