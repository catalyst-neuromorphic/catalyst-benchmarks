"""Train attention-enhanced Conv SNN on N-MNIST.

Adds temporal attention readout and channel squeeze-excitation (SE) blocks
to the conv SNN architecture, targeting 99.5%+ accuracy.

Inspired by STCA-SNN (99.67%) and PLIF (99.61%). Our architecture already
has learnable per-channel membrane time constants (PLIF-style). The key
additions are:
  1. Channel SE blocks after each conv stage (channel attention)
  2. Temporal attention readout (replaces naive time-average)
  3. Learnable thresholds for conv layers
  4. Stronger augmentation

Architecture:
  Conv(2,64,5)+BN+SE+LIF -> Pool ->
  Conv(64,128,3)+BN+SE+LIF -> Pool ->
  Flatten(128*6*6=4608) -> 512(adLIF) -> temporal_attn -> 10

Usage:
    python nmnist/train_v2.py --epochs 80 --device cuda:0 --amp
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

from common.neurons import AdaptiveLIFNeuron, surrogate_spike
from common.training import run_training
from common.augmentation import event_drop

from nmnist.loader import NMNISTDataset, collate_fn, N_CHANNELS, N_CLASSES


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation block for channel attention on spike maps.

    Global avg pool -> FC -> ReLU -> FC -> Sigmoid -> channel reweight.
    Adapted for spiking feature maps (binary {0,1} inputs).
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        """x: (batch, channels, H, W) -> reweighted (batch, channels, H, W)."""
        # Squeeze: global average pool
        se = x.mean(dim=(-2, -1))  # (batch, channels)
        # Excitation
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        # Scale
        return x * se.unsqueeze(-1).unsqueeze(-1)


class AttnConvNMNISTSNN(nn.Module):
    """Attention-enhanced Conv SNN for N-MNIST.

    Key differences from ConvNMNISTSNN:
    - Wider conv channels (64, 128) for richer features
    - Channel SE blocks after each conv stage
    - Learnable per-channel thresholds for conv layers
    - Temporal attention readout (learned weighted sum over time)
    - Larger FC layer (512 adLIF)
    """

    def __init__(self, n_fc=512, n_output=N_CLASSES, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 beta_conv=0.9):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output

        # Conv layers (wider than baseline)
        self.conv1 = nn.Conv2d(2, 64, 5, bias=False)   # 34->30
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = ChannelSE(64, reduction=4)

        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)  # 15->13
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = ChannelSE(128, reduction=4)

        # Per-channel learnable membrane decay AND threshold
        init_bc = np.log(beta_conv / (1.0 - beta_conv))
        self.beta_conv1_raw = nn.Parameter(torch.full((64,), init_bc))
        self.beta_conv2_raw = nn.Parameter(torch.full((128,), init_bc))
        self.thresh_conv1 = nn.Parameter(torch.ones(64))
        self.thresh_conv2 = nn.Parameter(torch.ones(128))

        # After conv2 + pool(2): 128 * 6 * 6 = 4608
        # Conv1: 34->30, pool->15. Conv2: 15->13, pool->6
        self.fc_flat = 128 * 6 * 6

        # FC adLIF layer (larger)
        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        self.lif_fc = AdaptiveLIFNeuron(
            n_fc, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=1.0)

        # Readout
        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        init_beta_out = np.log(0.9 / 0.1)
        self.beta_out = nn.Parameter(torch.tensor(init_beta_out))

        # Temporal attention: learns which timesteps matter most
        self.temporal_attn = nn.Sequential(
            nn.Linear(n_output, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=4.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=2.0)
        # Init SE blocks to identity (sigmoid(0)=0.5, so init bias to ~2 for ~0.88)
        nn.init.zeros_(self.se1.fc1.weight)
        nn.init.zeros_(self.se1.fc2.weight)
        self.se1.fc2.bias.data.fill_(2.0)  # sigmoid(2) ≈ 0.88, near-identity
        nn.init.zeros_(self.se2.fc1.weight)
        nn.init.zeros_(self.se2.fc2.weight)
        self.se2.fc2.bias.data.fill_(2.0)

    def forward(self, x):
        # x: (batch, T, 2, 34, 34)
        batch, T = x.shape[:2]
        device = x.device

        v_c1 = torch.zeros(batch, 64, 15, 15, device=device)
        v_c2 = torch.zeros(batch, 128, 6, 6, device=device)
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc = torch.zeros(batch, self.n_fc, device=device)
        a_fc = torch.zeros(batch, self.n_fc, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)

        beta_c1 = torch.sigmoid(self.beta_conv1_raw).view(1, 64, 1, 1)
        beta_c2 = torch.sigmoid(self.beta_conv2_raw).view(1, 128, 1, 1)
        thresh_c1 = self.thresh_conv1.view(1, 64, 1, 1)
        thresh_c2 = self.thresh_conv2.view(1, 128, 1, 1)

        temporal_outputs = []  # collect readout at each timestep

        for t in range(T):
            # Conv1 + BN + Pool + SE + LIF (learnable threshold)
            c1 = self.bn1(self.conv1(x[:, t]))       # (B, 64, 30, 30)
            c1 = F.avg_pool2d(c1, 2)                   # (B, 64, 15, 15)
            c1 = self.se1(c1)                           # channel attention
            v_c1 = beta_c1 * v_c1 + (1.0 - beta_c1) * c1
            spk_c1 = surrogate_spike(v_c1 - thresh_c1)
            v_c1 = v_c1 * (1.0 - spk_c1)

            # Conv2 + BN + Pool + SE + LIF (learnable threshold)
            c2 = self.bn2(self.conv2(spk_c1))         # (B, 128, 13, 13)
            c2 = F.avg_pool2d(c2, 2)                    # (B, 128, 6, 6)
            c2 = self.se2(c2)
            v_c2 = beta_c2 * v_c2 + (1.0 - beta_c2) * c2
            spk_c2 = surrogate_spike(v_c2 - thresh_c2)
            v_c2 = v_c2 * (1.0 - spk_c2)

            # Flatten + FC adLIF
            flat = spk_c2.view(batch, -1)
            flat = self.dropout(flat) if self.training else flat
            I_fc = self.fc1(flat)
            v_fc, spk_fc, a_fc = self.lif_fc(I_fc, v_fc, a_fc, spk_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc

            # Non-spiking readout (collect for temporal attention)
            I_out = self.fc_out(spk_fc_d)
            beta_o = torch.sigmoid(self.beta_out)
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            temporal_outputs.append(v_out)

        # Residual temporal attention: standard mean + learned attention offset
        # This ensures gradient flow even when early readout voltages are ~0
        temporal_stack = torch.stack(temporal_outputs, dim=1)  # (B, T, n_output)
        mean_out = temporal_stack.mean(dim=1)  # (B, n_output) — standard baseline

        attn_logits = self.temporal_attn(temporal_stack)  # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        attn_out = (temporal_stack * attn_weights).sum(dim=1)  # (B, n_output)

        return mean_out + attn_out  # residual: baseline + learned correction


def main():
    parser = argparse.ArgumentParser(
        description="Train attention-enhanced Conv SNN on N-MNIST")
    parser.add_argument("--data-dir", default="data/nmnist")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fc-hidden", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--time-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="checkpoints/nmnist_n3_attn.pt")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading N-MNIST dataset...")
    train_ds = NMNISTDataset(args.data_dir, train=True,
                              n_time_bins=args.time_bins, flatten=False)
    test_ds = NMNISTDataset(args.data_dir, train=False,
                             n_time_bins=args.time_bins, flatten=False)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {args.time_bins}")

    model = AttnConvNMNISTSNN(
        n_fc=args.fc_hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Conv(2,64,5)+SE+LIF -> Pool -> Conv(64,128,3)+SE+LIF -> Pool -> "
          f"{args.fc_hidden}(adLIF) -> temporal_attn -> {N_CLASSES}")
    print(f"Key features: channel SE attention, learnable thresholds, "
          f"temporal attention readout")

    def augment_fn(x):
        # x: (B, T, 2, 34, 34) -- event_drop supports this shape
        return event_drop(x, drop_time_prob=0.15, drop_neuron_prob=0.08)

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_path': args.save,
        'benchmark': 'nmnist',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': args.warmup_epochs,
        'model_config': {
            'conv': True,
            'attention': True,
            'n_output': N_CLASSES,
            'fc_hidden': args.fc_hidden,
            'threshold': args.threshold,
            'dropout': args.dropout,
            'neuron_type': 'attn-se-conv-adlif',
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
