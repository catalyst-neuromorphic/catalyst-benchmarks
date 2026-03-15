"""Train recurrent adLIF SNN on Google Speech Commands (35-class).

v4: Recurrent architecture (PROVEN to work on SHD 91.0%, SSC 76.4%).
Abandons dual-delay feedforward (v3 peaked at 47%, massive overfitting).

Key insight: delays are counterproductive for audio classification.
- SHD: 91.0% without delays vs 86.1% with delays
- SSC delays v6: 0.5% at epoch 6 (worse than random)
- GSC v3 delays: 47% best, test collapsed to 10% by epoch 115

Architecture: 40 -> 512 (rec adLIF) -> 256 (adLIF) -> 35 (readout)
Uses N3 encoding (mel+delta+delta2 = 120 channels) for richer input.

Usage:
    python gsc_kws/train_v4.py --device cuda:0 --amp --epochs 200
    python gsc_kws/train_v4.py --encoding n3 --hidden1 512 --hidden2 256 --amp
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

from gsc_kws.loader import (GSCDataset, collate_fn, N_CHANNELS,
                              N_CLASSES, N_CLASSES_35)


class GSC_RecurrentSNN(nn.Module):
    """Recurrent adLIF SNN for GSC-35 classification.

    Same architecture that achieved 91.0% SHD and 76.4% SSC.
    Recurrence provides temporal memory — no delays needed.
    """

    def __init__(self, n_input=N_CHANNELS * 3, n_hidden1=512, n_hidden2=256,
                 n_output=N_CLASSES_35, threshold=1.0, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 beta_out=0.9, target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # Layer 1: input + recurrent
        self.fc1 = nn.Linear(n_input, n_hidden1, bias=False)
        self.fc_rec = nn.Linear(n_hidden1, n_hidden1, bias=False)
        # Layer 2: feedforward
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        # Readout
        self.fc3 = nn.Linear(n_hidden2, n_output, bias=False)

        # Neurons
        self.lif1 = AdaptiveLIFNeuron(n_hidden1, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)
        self.lif2 = AdaptiveLIFNeuron(n_hidden2, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

        # Init
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
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

        # Adaptation state for adLIF
        a1 = torch.zeros(batch, self.n_hidden1, device=device)
        a2 = torch.zeros(batch, self.n_hidden2, device=device)

        spike_count1 = torch.zeros(batch, self.n_hidden1, device=device)

        for t in range(T):
            # Layer 1: input + recurrent feedback (use dropped spikes for recurrence)
            cur1 = self.fc1(x[:, t]) + self.fc_rec(spk1_d)
            v1, spk1, a1 = self.lif1(cur1, v1, a1, spk1)
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count1 += spk1.detach()

            # Layer 2: feedforward from layer 1
            cur2 = self.fc2(spk1_d)
            v2, spk2, a2 = self.lif2(cur2, v2, a2, spk2)
            spk2_d = self.dropout(spk2) if self.training else spk2

            # Readout: leaky accumulate (no spike)
            beta_out = self.lif_out.beta
            v_out = beta_out * v_out + (1.0 - beta_out) * self.fc3(spk2_d)
            out_sum += v_out

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count1 / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description='GSC v4: Recurrent adLIF SNN')
    parser.add_argument('--data-dir', default='data/gsc', help='GSC data directory')
    parser.add_argument('--encoding', default='n3', choices=['s2s', 'n3'],
                        help='Encoding: s2s (40ch binary) or n3 (120ch continuous)')
    parser.add_argument('--hidden1', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true', help='Mixed precision')
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save', default='checkpoints/gsc_v4.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Determine input channels
    n_channels = 40
    if args.encoding == 'n3':
        n_input = n_channels * 3  # mel + delta + delta2 = 120
    else:
        n_input = n_channels  # S2S binary = 40

    # Load data
    train_ds = GSCDataset(data_dir=args.data_dir, split='training',
                          n_channels=n_channels, full_35=True,
                          encoding=args.encoding, cache=True)
    test_ds = GSCDataset(data_dir=args.data_dir, split='testing',
                         n_channels=n_channels, full_35=True,
                         encoding=args.encoding, cache=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    model = GSC_RecurrentSNN(
        n_input=n_input,
        n_hidden1=args.hidden1,
        n_hidden2=args.hidden2,
        threshold=args.threshold,
        dropout=args.dropout,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_input}->{args.hidden1}(rec adLIF)->{args.hidden2}(adLIF)->35")
    print(f"Parameters: {params:,}")
    print(f"Encoding: {args.encoding} ({n_input} channels)")
    print(f"Mixed precision (AMP): {'enabled' if args.amp else 'disabled'}")

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'save_path': args.save,
        'benchmark': 'gsc',
        'label_smoothing': 0.05,
        'clip_norm': 1.0,
        'use_amp': args.amp,
    }
    run_training(model, train_loader, test_loader, config)


if __name__ == '__main__':
    main()
