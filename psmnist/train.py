"""Train SNN on Permuted Sequential MNIST (PS-MNIST).

Architecture: 1 -> hidden (recurrent adLIF) -> 10
Tests long-range temporal dependency with destroyed spatial structure.

Usage:
    python psmnist/train.py --epochs 50 --device cuda:0
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
from common.neurons_n3 import TDMLIFNeuron
from common.training import run_training

from psmnist.loader import PSMNISTDataset, collate_fn, N_CHANNELS, N_CLASSES


class PSMNISTSNN(nn.Module):
    """Recurrent SNN for Permuted Sequential MNIST."""

    def __init__(self, n_input=N_CHANNELS, n_hidden=256, n_output=N_CLASSES,
                 beta_hidden=0.95, beta_out=0.9, threshold=1.0, dropout=0.2,
                 neuron_type='adlif', alpha_init=0.93, rho_init=0.85,
                 beta_a_init=0.05):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.neuron_type = neuron_type

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_output, bias=False)

        if neuron_type == 'adlif':
            self.lif1 = AdaptiveLIFNeuron(
                n_hidden, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden, beta_init=beta_hidden,
                                   threshold=threshold, learn_beta=True)

        self.lif2 = LIFNeuron(n_output, beta_init=beta_out,
                               threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        v2 = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)
            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)
            spk1_d = self.dropout(spk1) if self.training else spk1

            I2 = self.fc2(spk1_d)
            beta_out = self.lif2.beta
            v2 = beta_out * v2 + (1.0 - beta_out) * I2
            out_sum = out_sum + v2

        return out_sum / T


class TDMPSMNISTSNN(nn.Module):
    """Recurrent SNN with TDM shadow banks for PS-MNIST.

    Uses TDMLIFNeuron to simulate N3's Time-Division Multiplexing:
    one physical neuron handles n_banks virtual time slots via
    independent membrane potential banks. For 784 timesteps with
    4 banks, each physical neuron processes 196 effective timesteps.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=256, n_output=N_CLASSES,
                 beta_out=0.9, threshold=1.0, dropout=0.2, n_banks=4):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_banks = n_banks

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_output, bias=False)

        self.lif1 = TDMLIFNeuron(n_hidden, n_banks=n_banks, threshold=threshold)
        self.lif2 = LIFNeuron(n_output, beta_init=beta_out,
                               threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v_banks = torch.zeros(batch, self.n_banks, self.n_hidden, device=device)
        v2 = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)
            v_banks, spk1 = self.lif1(I1, v_banks, t)
            spk1_d = self.dropout(spk1) if self.training else spk1

            I2 = self.fc2(spk1_d)
            beta_out = self.lif2.beta
            v2 = beta_out * v2 + (1.0 - beta_out) * I2
            out_sum = out_sum + v2

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SNN on PS-MNIST")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="psmnist_model.pt")
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--alpha-init", type=float, default=0.93)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--tdm", action="store_true", default=False,
                        help="Use TDM shadow banks (N3 feature)")
    parser.add_argument("--tdm-banks", type=int, default=4,
                        help="Number of TDM banks")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading PS-MNIST dataset...")
    train_ds = PSMNISTDataset(train=True)
    test_ds = PSMNISTDataset(train=False)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    if args.tdm:
        model = TDMPSMNISTSNN(
            n_hidden=args.hidden, dropout=args.dropout,
            n_banks=args.tdm_banks,
        ).to(device)
        print(f"Model: {N_CHANNELS}->{args.hidden}(TDM {args.tdm_banks} banks)->{N_CLASSES} "
              f"(N3 TDM, recurrent=on, dropout={args.dropout})")
    else:
        model = PSMNISTSNN(
            n_hidden=args.hidden, dropout=args.dropout, neuron_type=args.neuron,
            alpha_init=args.alpha_init, rho_init=args.rho_init,
            beta_a_init=args.beta_a_init,
        ).to(device)
        print(f"Model: {N_CHANNELS}->{args.hidden}->{N_CLASSES} "
              f"({args.neuron.upper()}, recurrent=on, dropout={args.dropout})")

    config = {
        'device': device, 'epochs': args.epochs, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'save_path': args.save,
        'benchmark': 'psmnist', 'augment_fn': None,
        'label_smoothing': 0.0,
        'model_config': {
            'n_input': N_CHANNELS, 'hidden': args.hidden,
            'n_output': N_CLASSES, 'neuron_type': 'tdm' if args.tdm else args.neuron,
            'dropout': args.dropout,
            'tdm': args.tdm,
            'tdm_banks': args.tdm_banks if args.tdm else None,
        },
    }
    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
