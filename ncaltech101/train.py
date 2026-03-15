"""Train SNN on N-Caltech101 benchmark.

Architecture: 2048 -> hidden1 (recurrent adLIF) -> hidden2 (adLIF) -> 101
101-class event-camera object classification.

Usage:
    python ncaltech101/train.py --epochs 200 --device cuda:0
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

from ncaltech101.loader import NCaltech101Dataset, collate_fn, N_CHANNELS, N_CLASSES


class NCaltech101SNN(nn.Module):
    """Two-layer SNN for N-Caltech101 classification."""

    def __init__(self, n_input=N_CHANNELS, n_hidden1=512, n_hidden2=256,
                 n_output=N_CLASSES, beta_hidden=0.95, beta_out=0.9,
                 threshold=1.0, dropout=0.3, neuron_type='adlif',
                 alpha_init=0.93, rho_init=0.85, beta_a_init=0.05):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.neuron_type = neuron_type

        self.fc1 = nn.Linear(n_input, n_hidden1, bias=False)
        self.fc_rec = nn.Linear(n_hidden1, n_hidden1, bias=False)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.fc3 = nn.Linear(n_hidden2, n_output, bias=False)

        if neuron_type == 'adlif':
            self.lif1 = AdaptiveLIFNeuron(
                n_hidden1, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
            self.lif2 = AdaptiveLIFNeuron(
                n_hidden2, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden1, beta_init=beta_hidden,
                                   threshold=threshold, learn_beta=True)
            self.lif2 = LIFNeuron(n_hidden2, beta_init=beta_hidden,
                                   threshold=threshold, learn_beta=True)

        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden1, device=device)
            a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)
            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)
            spk1_d = self.dropout1(spk1) if self.training else spk1

            I2 = self.fc2(spk1_d)
            if self.neuron_type == 'adlif':
                v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            else:
                v2, spk2 = self.lif2(I2, v2)
            spk2_d = self.dropout2(spk2) if self.training else spk2

            I_out = self.fc3(spk2_d)
            beta_out = self.lif_out.beta
            v_out = beta_out * v_out + (1.0 - beta_out) * I_out
            out_sum = out_sum + v_out

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SNN on N-Caltech101")
    parser.add_argument("--data-dir", default="data/ncaltech101")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="ncaltech101_model.pt")
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--alpha-init", type=float, default=0.93)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--event-drop", action="store_true", default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading N-Caltech101 dataset...")
    train_ds = NCaltech101Dataset(args.data_dir, train=True, n_time_bins=args.time_bins)
    test_ds = NCaltech101Dataset(args.data_dir, train=False, n_time_bins=args.time_bins)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Time bins: {args.time_bins}")

    model = NCaltech101SNN(
        n_hidden1=args.hidden1, n_hidden2=args.hidden2,
        dropout=args.dropout, neuron_type=args.neuron,
        alpha_init=args.alpha_init, rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
    ).to(device)

    print(f"Model: {N_CHANNELS}->{args.hidden1}->{args.hidden2}->{N_CLASSES} "
          f"({args.neuron.upper()}, recurrent=on, dropout={args.dropout})")

    augment_fn = (lambda x: event_drop(x)) if args.event_drop else None

    config = {
        'device': device, 'epochs': args.epochs, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'save_path': args.save,
        'benchmark': 'ncaltech101', 'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'model_config': {
            'n_input': N_CHANNELS, 'hidden1': args.hidden1,
            'hidden2': args.hidden2, 'n_output': N_CLASSES,
            'neuron_type': args.neuron, 'dropout': args.dropout,
        },
    }
    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
