"""Train SNN on Google Speech Commands (GSC) for Keyword Spotting.

Uses Speech2Spikes-style delta modulation encoding: raw audio → log mel
spectrogram → delta modulation → binary spike trains {-1, 0, 1}.

Architecture: 40 (S2S spikes) -> 512 (rec adLIF) -> 12 (non-spiking readout)
Matches SHD v7 architecture that achieved 90.7% with recurrent dropout +
activity regularization.

Usage:
    python gsc_kws/train.py --epochs 200 --device cuda:0 --amp
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

from common.neurons import LIFNeuron, AdaptiveLIFNeuron, surrogate_spike
from common.neurons_n3 import ANNINT8Linear
from common.training import run_training
from common.augmentation import event_drop, time_stretch

from gsc_kws.loader import GSCDataset, collate_fn, N_CHANNELS, N_CLASSES


class ConvGSCSNN(nn.Module):
    """Hybrid Conv1D + SNN for Google Speech Commands.

    Input: (batch, T, 120) mel + delta + delta-delta features.
    Reshaped to (batch, T, 3, 40) — 3 feature types, 40 mel bands each.

    Conv layers are NON-SPIKING feature extractors (ReLU, not LIF).
    Only the recurrent FC layer uses spiking neurons.
    This preserves continuous mel information through the conv layers.

    Conv1d(3, 32, 5) + BN + ReLU -> Pool(2) ->
    Conv1d(32, 64, 3) + BN + ReLU -> Pool(2) ->
    Flatten(64*8=512) -> 256 (rec adLIF) -> 12 (readout)
    """

    def __init__(self, n_fc=256, n_output=N_CLASSES, threshold=1.0,
                 dropout=0.2, alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, n_mels=40,
                 target_rate=0.05, activity_lambda=0.005):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output
        self.threshold = threshold
        self.n_mels = n_mels
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # Conv1D layers over frequency axis (NON-SPIKING, ReLU activation)
        # Input: (batch, 3, 40) — 3 channels (mel, delta, delta-delta), 40 freq bins
        self.conv1 = nn.Conv1d(3, 32, 5, bias=False)    # 40->36
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, bias=False)   # 18->16
        self.bn2 = nn.BatchNorm1d(64)

        # After conv2 + pool(2): 64 * 8 = 512
        # Conv1: 40 -> 36 -> Pool(2) -> 18
        # Conv2: 18 -> 16 -> Pool(2) -> 8
        self.fc_flat = 64 * 8

        # FC recurrent adLIF layer (SPIKING)
        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        self.fc_rec = nn.Linear(n_fc, n_fc, bias=False)
        self.lif_fc = AdaptiveLIFNeuron(
            n_fc, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)

        # Readout (non-spiking)
        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        self.lif_out = LIFNeuron(n_output, beta_init=0.9,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        # x: (batch, T, 120) — reshape to (batch, T, 3, 40)
        batch, T, C = x.shape
        x = x.view(batch, T, 3, self.n_mels)  # (batch, T, 3, 40)

        device = x.device

        # FC states
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc_d = torch.zeros(batch, self.n_fc, device=device)
        a_fc = torch.zeros(batch, self.n_fc, device=device)

        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count = torch.zeros(batch, self.n_fc, device=device)

        for t in range(T):
            # Reshape: (batch, 3, 40) for Conv1d
            x_t = x[:, t]  # (batch, 3, 40)

            # Conv layer 1: Conv1d -> BN -> ReLU -> Pool (NON-SPIKING)
            c1 = F.relu(self.bn1(self.conv1(x_t)))  # (batch, 32, 36)
            c1 = F.avg_pool1d(c1, 2)                  # (batch, 32, 18)

            # Conv layer 2: Conv1d -> BN -> ReLU -> Pool (NON-SPIKING)
            c2 = F.relu(self.bn2(self.conv2(c1)))    # (batch, 64, 16)
            c2 = F.avg_pool1d(c2, 2)                   # (batch, 64, 8)

            # Flatten + dropout
            flat = c2.view(batch, -1)                   # (batch, 512)
            flat = self.dropout(flat) if self.training else flat

            # Recurrent FC adLIF (SPIKING)
            I_fc = self.fc1(flat) + self.fc_rec(spk_fc_d)
            v_fc, spk_fc, a_fc = self.lif_fc(I_fc, v_fc, a_fc, spk_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc
            spike_count = spike_count + spk_fc

            # Non-spiking readout
            I_out = self.fc_out(spk_fc_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


class GSCSNN(nn.Module):
    """Single-layer recurrent SNN for GSC (matches SHD v7 architecture).

    120 -> n_hidden (recurrent adLIF) -> 12 (non-spiking readout)

    Uses same regularization as SHD v7:
    - Recurrent dropout (dropped spikes fed back into recurrence)
    - Activity regularization (L2 penalty toward target firing rate)
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=512,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.3,
                 neuron_type='adlif', alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_out=0.9,
                 target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc_out = nn.Linear(n_hidden, n_output, bias=False)

        if neuron_type == 'adlif':
            self.lif1 = AdaptiveLIFNeuron(n_hidden, alpha_init=alpha_init,
                                           rho_init=rho_init, beta_a_init=beta_a_init,
                                           threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden, beta_init=0.95,
                                   threshold=threshold, learn_beta=True)

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
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden, device=device)
        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            # Recurrent connection uses DROPPED spikes
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1_d)
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

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


class TwoLayerGSCSNN(nn.Module):
    """Two-layer recurrent SNN for GSC (same approach as SSC v4 which hit 72.1%).

    input -> n_hidden1 (recurrent adLIF) -> n_hidden2 (adLIF) -> 12 (readout)

    Uses recurrent dropout + activity regularization from SHD v7.
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=512, n_hidden2=256,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 beta_out=0.9, target_rate=0.05, activity_lambda=0.01):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        self.fc1 = nn.Linear(n_input, n_hidden1, bias=False)
        self.fc_rec = nn.Linear(n_hidden1, n_hidden1, bias=False)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.fc_out = nn.Linear(n_hidden2, n_output, bias=False)

        self.lif1 = AdaptiveLIFNeuron(n_hidden1, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)
        self.lif2 = AdaptiveLIFNeuron(n_hidden2, alpha_init=alpha_init,
                                       rho_init=rho_init, beta_a_init=beta_a_init,
                                       threshold=threshold)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        a1 = torch.zeros(batch, self.n_hidden1, device=device)
        a2 = torch.zeros(batch, self.n_hidden2, device=device)
        spike_count = torch.zeros(batch, self.n_hidden1, device=device)

        for t in range(T):
            # Layer 1: recurrent adLIF with dropout
            I1 = self.fc1(x[:, t]) + self.fc_rec(
                self.dropout(spk1) if self.training else spk1)
            v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            spike_count = spike_count + spk1

            # Layer 2: feedforward adLIF
            spk1_d = self.dropout(spk1) if self.training else spk1
            I2 = self.fc2(spk1_d)
            v2, spk2, a2 = self.lif2(I2, v2, a2, torch.zeros_like(v2))

            # Readout (non-spiking)
            spk2_d = self.dropout(spk2) if self.training else spk2
            I_out = self.fc_out(spk2_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


class HybridGSCSNN(nn.Module):
    """N3 Hybrid ANN+SNN for GSC KWS.

    First layer uses ANNINT8Linear (N3 ANN INT8 MAC mode, non-spiking)
    to process continuous mel+delta+delta-delta features without the
    information loss of delta modulation. Subsequent layers are spiking.

    This maps to N3 hardware where core 0 operates in ANN INT8 MAC mode
    and cores 1+ operate in standard SNN mode.

    Input: (batch, T, n_mels*3) continuous mel features
      -> ANNINT8Linear(n_mels*3, n_proj) [non-spiking INT8 MAC]
      -> Linear(n_proj, n_hidden) + rec + AdaptiveLIF [spiking recurrent]
      -> Linear(n_hidden, 12) + leaky readout
    """

    def __init__(self, n_mels=40, n_proj=256, n_hidden=512,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.2,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 target_rate=0.05, activity_lambda=0.005):
        super().__init__()
        self.n_mels = n_mels
        n_input = n_mels * 3  # mel + delta + delta-delta
        self.n_proj = n_proj
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None

        # N3 ANN INT8 MAC input layer (non-spiking)
        self.int8_proj = ANNINT8Linear(n_input, n_proj, bias=True)

        # Spiking recurrent layer
        self.fc1 = nn.Linear(n_proj, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.lif1 = AdaptiveLIFNeuron(
            n_hidden, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)

        # Readout (non-spiking)
        self.fc_out = nn.Linear(n_hidden, n_output, bias=False)
        self.lif_out = LIFNeuron(n_output, beta_init=0.9,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden, device=device)
        a1 = torch.zeros(batch, self.n_hidden, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            # INT8 MAC projection (non-spiking, preserves continuous features)
            proj = self.int8_proj(x[:, t])

            # Spiking recurrent layer
            I1 = self.fc1(proj) + self.fc_rec(spk1_d)
            v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
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


def main():
    parser = argparse.ArgumentParser(description="Train SNN on GSC for KWS")
    parser.add_argument("--data-dir", default="data/gsc")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="gsc_kws_model.pt")
    parser.add_argument("--event-drop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-stretch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha-init", type=float, default=0.95)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05,
                        help="Target firing rate for activity regularization")
    parser.add_argument("--activity-lambda", type=float, default=0.01,
                        help="Activity regularization strength")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed precision training (2-3x speedup)")
    parser.add_argument("--conv", action="store_true", default=False,
                        help="Use Conv1D SNN architecture for mel spectrograms")
    parser.add_argument("--fc-hidden", type=int, default=256,
                        help="FC hidden size after conv layers (conv mode)")
    parser.add_argument("--two-layer", action="store_true", default=False,
                        help="Use two-layer architecture (like SSC v4)")
    parser.add_argument("--hidden2", type=int, default=256,
                        help="Second hidden layer size (two-layer mode)")
    parser.add_argument("--n-mels", type=int, default=40,
                        help="Number of mel bands (40 or 80)")
    parser.add_argument("--n3-hybrid", action="store_true", default=False,
                        help="Use N3 INT8 MAC hybrid ANN+SNN architecture")
    parser.add_argument("--n3-proj", type=int, default=256,
                        help="INT8 projection size for N3 hybrid mode")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    n_mels = args.n_mels
    encoding = 'n3' if args.n3_hybrid else 's2s'
    enc_name = "N3 INT8 mel" if encoding == 'n3' else "Speech2Spikes"
    print(f"Loading Google Speech Commands dataset ({enc_name}, {n_mels} mel bands)...")
    train_ds = GSCDataset(args.data_dir, split="training",
                           n_channels=n_mels,
                           max_time_bins=args.time_bins,
                           threshold=args.threshold,
                           encoding=encoding)
    test_ds = GSCDataset(args.data_dir, split="testing",
                          n_channels=n_mels,
                          max_time_bins=args.time_bins,
                          threshold=args.threshold,
                          encoding=encoding)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {args.time_bins}, Channels: {n_mels}")

    use_conv = args.conv
    use_two_layer = args.two_layer
    use_n3_hybrid = args.n3_hybrid

    if use_n3_hybrid:
        model = HybridGSCSNN(
            n_mels=n_mels,
            n_proj=args.n3_proj,
            n_hidden=args.hidden,
            threshold=args.threshold,
            dropout=args.dropout,
            alpha_init=args.alpha_init,
            rho_init=args.rho_init,
            beta_a_init=args.beta_a_init,
            target_rate=args.target_rate,
            activity_lambda=args.activity_lambda,
        ).to(device)
        print(f"Model: {n_mels*3}->[INT8 MAC {args.n3_proj}]->"
              f"{args.hidden}(rec adLIF)->{N_CLASSES} "
              f"(N3 HYBRID ANN+SNN, dropout={args.dropout})")
    elif use_two_layer:
        model = TwoLayerGSCSNN(
            n_input=n_mels,
            n_hidden1=args.hidden,
            n_hidden2=args.hidden2,
            threshold=args.threshold,
            dropout=args.dropout,
            alpha_init=args.alpha_init,
            rho_init=args.rho_init,
            beta_a_init=args.beta_a_init,
            target_rate=args.target_rate,
            activity_lambda=args.activity_lambda,
        ).to(device)
        print(f"Model: {n_mels}->{args.hidden}(rec adLIF)->{args.hidden2}(adLIF)->{N_CLASSES} "
              f"(TWO-LAYER, dropout={args.dropout})")
    elif use_conv:
        model = ConvGSCSNN(
            n_fc=args.fc_hidden,
            threshold=args.threshold,
            dropout=args.dropout,
        ).to(device)
        print(f"Model: Conv1d(3,32,5)->Pool->Conv1d(32,64,3)->Pool->"
              f"{args.fc_hidden}(rec adLIF)->{N_CLASSES} "
              f"(CONV+adLIF, dropout={args.dropout})")
    else:
        model = GSCSNN(
            n_input=n_mels,
            n_hidden=args.hidden,
            threshold=args.threshold,
            dropout=args.dropout,
            neuron_type=args.neuron,
            alpha_init=args.alpha_init,
            rho_init=args.rho_init,
            beta_a_init=args.beta_a_init,
            target_rate=args.target_rate,
            activity_lambda=args.activity_lambda,
        ).to(device)
        print(f"Model: {n_mels}->{args.hidden}(rec adLIF)->{N_CLASSES} "
              f"({args.neuron.upper()}, recurrent=on, dropout={args.dropout})")

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
        'benchmark': 'gsc_kws',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': 0,
        'model_config': {
            'conv': use_conv,
            'two_layer': use_two_layer,
            'n3_hybrid': use_n3_hybrid,
            'n_input': n_mels,
            'n_proj': args.n3_proj if use_n3_hybrid else None,
            'hidden': args.hidden,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': 'n3-hybrid' if use_n3_hybrid else ('two-layer-adlif' if use_two_layer else ('conv+adlif' if use_conv else args.neuron)),
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
