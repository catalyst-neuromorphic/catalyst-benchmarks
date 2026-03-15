"""Train SNN on Spiking Heidelberg Digits (SHD) benchmark.

v8: Two-layer recurrent architecture for N3 benchmarks.
  - v7 features: recurrent dropout, activity reg, weight decay
  - NEW: Optional second recurrent adLIF layer (--layers 2)
  - Single-layer: 700 -> 1024 (rec adLIF) -> 20
  - Two-layer:    700 -> 1024 (rec adLIF) -> 512 (rec adLIF) -> 20

Usage:
    python shd/train.py --neuron adlif --epochs 200 --weight-decay 5e-4
    python shd/train.py --layers 2 --hidden 1024 --hidden2 512 --epochs 200
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.neurons import LIFNeuron, AdaptiveLIFNeuron, surrogate_spike
from common.training import run_training
from common.augmentation import event_drop

from shd.loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES


class SHDSNN(nn.Module):
    """Single-layer recurrent SNN for SHD classification.

    700 -> 1024 (recurrent adLIF) -> 20 (non-spiking readout)

    Regularization:
      - Dropout on both output AND recurrent connections
      - Activity regularization penalizing deviation from target firing rate
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=1024,
                 n_output=N_CLASSES, beta_out=0.9, threshold=1.0, dropout=0.3,
                 neuron_type='adlif', alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None  # Set during forward pass

        # Input + recurrent connections
        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
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

        # Per-output learnable readout decay
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        # Initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        # For activity regularization
        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            # Recurrent connection uses DROPPED spikes (regularizes recurrence)
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1_d)

            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

            # Apply dropout — used for BOTH recurrence and readout
            spk1_d = self.dropout(spk1) if self.training else spk1

            # Accumulate spike counts for activity reg
            spike_count = spike_count + spk1

            # Non-spiking readout
            I_out = self.fc_out(spk1_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        # Activity regularization: penalize deviation from target firing rate
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count / T  # (batch, n_hidden), values in [0, 1]
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


class SHDSNNv8(nn.Module):
    """Two-layer recurrent SNN for SHD classification (N3 benchmark).

    700 -> hidden1 (rec adLIF) -> hidden2 (rec adLIF) -> 20

    Both layers have recurrence + dropout regularization. The second
    recurrent layer adds hierarchical temporal processing — proven to give
    +14% on SSC, expected similar boost on SHD.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=512,
                 n_output=N_CLASSES, beta_out=0.9, threshold=1.0, dropout=0.3,
                 neuron_type='adlif', alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # Layer 1: input + recurrence
        self.fc1 = nn.Linear(n_input, n_hidden1, bias=False)
        self.fc_rec1 = nn.Linear(n_hidden1, n_hidden1, bias=False)

        # Layer 2: inter-layer + recurrence
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

        # Initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec1.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec2.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden1, device=device)
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)
        spk2_d = torch.zeros(batch, self.n_hidden2, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        spike_count1 = torch.zeros(batch, self.n_hidden1, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden1, device=device)
            a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            # Layer 1: recurrent dropout
            I1 = self.fc1(x[:, t]) + self.fc_rec1(spk1_d)
            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count1 = spike_count1 + spk1

            # Layer 2: recurrent dropout
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
            mean_rate = spike_count1 / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SNN on SHD benchmark")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512,
                        help="Second layer hidden size (only used with --layers 2)")
    parser.add_argument("--layers", type=int, choices=[1, 2], default=1,
                        help="Number of recurrent layers (1=v7, 2=v8)")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--beta-out", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="shd_model.pt")
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--alpha-init", type=float, default=0.95)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--event-drop", action="store_true", default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05,
                        help="Target firing rate for activity regularization")
    parser.add_argument("--activity-lambda", type=float, default=0.01,
                        help="Activity regularization strength")
    parser.add_argument("--resume-weights", default=None,
                        help="Load model weights from a checkpoint (warm-start, no optimizer state)")
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1, or cpu")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed precision training (2-3x speedup)")
    args = parser.parse_args()

    if args.event_drop is None:
        args.event_drop = (args.neuron == 'adlif')

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
        model = SHDSNNv8(
            n_hidden1=args.hidden,
            n_hidden2=args.hidden2,
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
        arch_str = f"{N_CHANNELS}->{args.hidden}(rec)->{args.hidden2}(rec)->{N_CLASSES}"
    else:
        model = SHDSNN(
            n_hidden=args.hidden,
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
        arch_str = f"{N_CHANNELS}->{args.hidden}(rec)->{N_CLASSES}"

    if args.resume_weights:
        ckpt_data = torch.load(args.resume_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
        print(f"Loaded weights from {args.resume_weights} "
              f"(acc={ckpt_data.get('test_acc', 0)*100:.2f}%)")

    neuron_info = args.neuron.upper()
    if args.neuron == 'adlif':
        neuron_info += f" (alpha={args.alpha_init}, rho={args.rho_init}, beta_a={args.beta_a_init})"
    print(f"Model: {arch_str} ({neuron_info}, "
          f"recurrent=on, dropout={args.dropout}, event_drop={args.event_drop})")
    print(f"Regularization: activity_reg(target={args.target_rate}, lambda={args.activity_lambda}), "
          f"weight_decay={args.weight_decay}, recurrent_dropout=on")

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
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': args.neuron,
            'beta_out': args.beta_out,
            'dropout': args.dropout,
            'alpha_init': args.alpha_init,
            'rho_init': args.rho_init,
            'beta_a_init': args.beta_a_init,
            'target_rate': args.target_rate,
            'activity_lambda': args.activity_lambda,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
