"""Train SHD with N4 spiking attention mechanism.

Demonstrates N4's 8-head spiking attention (spec feature 4M).
Adds a multi-head attention layer that operates on spike rates
to weight temporal features, matching the hardware attention unit.

Architecture: 700 -> 512 (rec adLIF) -> 512 (attention) -> 20

Usage:
    python shd/train_attention.py --epochs 100 --device cuda:0 --amp
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

from common.neurons import LIFNeuron, AdaptiveLIFNeuron
from common.training import run_training
from shd.loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES


class SpikingAttention(nn.Module):
    """Multi-head spiking attention matching N4 hardware.

    N4 implements 8-head spiking attention per tile (spec 4M):
    - Q, K, V derived from spike counts/rates
    - Attention scores via spike-rate dot products
    - Output is attention-weighted spike features

    This software implementation uses spike rates as Q/K/V inputs
    with scaled dot-product attention and learnable projections.
    """

    def __init__(self, embed_dim, n_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small values for stability
        for m in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn.init.xavier_uniform_(m.weight, gain=0.3)

    def forward(self, spikes, v_mem):
        """Compute spiking attention.

        Args:
            spikes: (batch, embed_dim) current spike output
            v_mem: (batch, embed_dim) membrane voltages (richer signal)

        Returns:
            (batch, embed_dim) attention-weighted output
        """
        batch = spikes.shape[0]
        # Use membrane voltage as continuous-valued input for Q/K/V
        # (spikes are binary, voltages carry more information)
        q = self.W_q(v_mem).view(batch, self.n_heads, self.head_dim)
        k = self.W_k(v_mem).view(batch, self.n_heads, self.head_dim)
        v = self.W_v(spikes).view(batch, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.einsum('bnh,bnh->bn', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)  # (batch, n_heads)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn.unsqueeze(-1) * v).reshape(batch, self.embed_dim)
        return self.W_out(out)


class AttentionSHDSNN(nn.Module):
    """SHD SNN with N4 spiking attention.

    700 -> 512 (rec adLIF) -> attention(8-head) -> 20

    The attention mechanism weights temporal features based on
    spike-rate patterns, enabling the network to focus on
    informative time segments.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=512, n_output=N_CLASSES,
                 n_heads=8, dropout=0.3, alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, threshold=1.0, beta_out=0.9,
                 target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)

        self.lif1 = AdaptiveLIFNeuron(n_hidden, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)

        # N4 spiking attention (8-head)
        self.attention = SpikingAttention(n_hidden, n_heads=n_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(n_hidden)

        self.fc_out = nn.Linear(n_hidden, n_output, bias=False)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        a1 = torch.zeros(batch, self.n_hidden, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        s1_prev = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(self.dropout(s1_prev))
            v1, s1, a1 = self.lif1(I1, v1, a1, s1_prev)

            spike_count += s1.detach()

            # Apply spiking attention
            attn_out = self.attention(s1, v1)
            s1_attn = s1 + attn_out  # residual connection
            s1_attn = self.attn_norm(s1_attn)

            s1_d = self.dropout(s1_attn) if self.training else s1_attn

            i_out = self.fc_out(s1_d)
            v_out, s_out = self.lif_out(i_out, v_out)
            out_sum += v_out
            s1_prev = s1

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description='SHD with N4 Spiking Attention')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save', default='checkpoints/shd_n4_attention.pt')
    parser.add_argument('--activity-lambda', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("CATALYST N4 — SHD with SPIKING ATTENTION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"N4 Feature: 8-head spiking attention (spec 4M)")
    print(f"Attention operates on spike rates + membrane voltages")
    print()

    print("Loading SHD dataset...")
    train_ds = SHDDataset(split="train")
    test_ds = SHDDataset(split="test")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = AttentionSHDSNN(
        n_hidden=args.hidden,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {N_CHANNELS}->{args.hidden}(rec+attn{args.n_heads}h)->{N_CLASSES}")
    print(f"Parameters: {n_params:,}")

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_path': args.save,
        'benchmark': 'shd-attention',
        'use_amp': args.amp,
        'warmup_epochs': args.warmup_epochs,
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden': args.hidden,
            'n_output': N_CLASSES,
            'n_heads': args.n_heads,
            'neuron_type': 'adlif',
            'synapse_type': 'linear',
            'attention': True,
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == '__main__':
    main()
