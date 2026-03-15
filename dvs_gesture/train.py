"""Train SNN on DVS128 Gesture benchmark.

Supports two architectures:
  FC:   2048 -> 512 (rec adLIF) -> 256 (adLIF) -> 11
  Conv: Conv(2,32,5)+LIF -> Pool -> Conv(32,64,5)+LIF -> Pool -> 256(adLIF) -> 11

Conv architecture is recommended — matches N-MNIST conv (98.8%).
DVS Gesture has only ~1342 training samples so conv features help generalize.

Usage:
    python dvs_gesture/train.py --conv --epochs 200 --device cuda:0 --amp
    python dvs_gesture/train.py --epochs 200  # FC fallback
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
from common.neurons_n3 import WTALayer
from common.training import run_training
from common.augmentation import event_drop, spatial_jitter

from dvs_gesture.loader import DVSGestureDataset, collate_fn, N_CHANNELS, N_CLASSES


class ConvDVSGestureSNN(nn.Module):
    """Conv SNN for DVS128 Gesture classification.

    Conv(2,32,5)+LIF -> Pool(2) -> Conv(32,64,5)+LIF -> Pool(2) ->
    Flatten(1600) -> 256 (adLIF) -> 11 (readout)

    Input: (batch, T, 2, 32, 32) from downsampled DVS frames.
    Conv1: (32, 28, 28) -> Pool -> (32, 14, 14)
    Conv2: (64, 10, 10) -> Pool -> (64, 5, 5) = 1600
    """

    def __init__(self, n_fc=256, n_output=N_CLASSES, threshold=1.0,
                 dropout=0.3, alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_conv=0.9, downsample=32):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output
        self.threshold = threshold

        # Conv layers
        self.conv1 = nn.Conv2d(2, 32, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # LIF membrane decay for conv layers (per-channel, learnable)
        init_bc = np.log(beta_conv / (1.0 - beta_conv))
        self.beta_conv1_raw = nn.Parameter(torch.full((32,), init_bc))
        self.beta_conv2_raw = nn.Parameter(torch.full((64,), init_bc))

        # Compute flatten size dynamically based on input spatial size
        s = downsample
        s = (s - 4) // 2   # conv1(5) + pool(2)
        s = (s - 4) // 2   # conv2(5) + pool(2)
        self.fc_flat = 64 * s * s

        # FC adLIF layer
        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        self.lif_fc = AdaptiveLIFNeuron(
            n_fc, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)

        # Readout
        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        init_beta_out = np.log(0.9 / 0.1)
        self.beta_out = nn.Parameter(torch.tensor(init_beta_out))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

    def forward(self, x):
        # x: (batch, T, 2, H, W)
        batch, T = x.shape[:2]
        device = x.device
        H, W = x.shape[3], x.shape[4]

        # Compute spatial sizes dynamically
        h1, w1 = (H - 4) // 2, (W - 4) // 2
        h2, w2 = (h1 - 4) // 2, (w1 - 4) // 2

        # Conv LIF membrane potentials
        v_c1 = torch.zeros(batch, 32, h1, w1, device=device)
        v_c2 = torch.zeros(batch, 64, h2, w2, device=device)

        # FC states
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc = torch.zeros(batch, self.n_fc, device=device)
        a_fc = torch.zeros(batch, self.n_fc, device=device)

        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        beta_c1 = torch.sigmoid(self.beta_conv1_raw).view(1, 32, 1, 1)
        beta_c2 = torch.sigmoid(self.beta_conv2_raw).view(1, 64, 1, 1)

        for t in range(T):
            # Conv layer 1: Conv -> BN -> Pool -> LIF
            c1 = self.bn1(self.conv1(x[:, t]))
            c1 = F.avg_pool2d(c1, 2)
            v_c1 = beta_c1 * v_c1 + (1.0 - beta_c1) * c1
            spk_c1 = surrogate_spike(v_c1 - self.threshold)
            v_c1 = v_c1 * (1.0 - spk_c1)

            # Conv layer 2: Conv -> BN -> Pool -> LIF
            c2 = self.bn2(self.conv2(spk_c1))
            c2 = F.avg_pool2d(c2, 2)
            v_c2 = beta_c2 * v_c2 + (1.0 - beta_c2) * c2
            spk_c2 = surrogate_spike(v_c2 - self.threshold)
            v_c2 = v_c2 * (1.0 - spk_c2)

            # Flatten + FC adLIF
            flat = spk_c2.view(batch, -1)              # (batch, 1600)
            flat = self.dropout(flat) if self.training else flat
            I_fc = self.fc1(flat)
            v_fc, spk_fc, a_fc = self.lif_fc(I_fc, v_fc, a_fc, spk_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc

            # Non-spiking readout
            I_out = self.fc_out(spk_fc_d)
            beta_o = torch.sigmoid(self.beta_out)
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        return out_sum / T


class DeepConvDVSGestureSNN(nn.Module):
    """Deep Conv SNN for DVS128 Gesture — v2 architecture for 90%+ accuracy.

    3 conv layers with more channels + recurrent FC adLIF.
    Designed for 48x48 input (more spatial detail than 32x32).

    Conv(2,64,3)+BN+Pool(2) -> Conv(64,128,3)+BN+Pool(2) ->
    Conv(128,256,3)+BN+Pool(2) -> Flatten -> 512(rec adLIF) -> 11

    Input: (batch, T, 2, 48, 48)
    Conv1: (64, 46, 46) -> Pool -> (64, 23, 23)
    Conv2: (128, 21, 21) -> Pool -> (128, 10, 10)
    Conv3: (256, 8, 8) -> Pool -> (256, 4, 4) = 4096
    """

    def __init__(self, n_fc=512, n_output=N_CLASSES, threshold=0.3,
                 dropout=0.4, alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_conv=0.9, downsample=48):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output
        self.threshold = threshold

        # 3 conv layers with increasing channels
        self.conv1 = nn.Conv2d(2, 64, 3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # Per-channel learnable membrane decay
        init_bc = np.log(beta_conv / (1.0 - beta_conv))
        self.beta_conv1_raw = nn.Parameter(torch.full((64,), init_bc))
        self.beta_conv2_raw = nn.Parameter(torch.full((128,), init_bc))
        self.beta_conv3_raw = nn.Parameter(torch.full((256,), init_bc))

        # Compute flatten size dynamically
        # 48 -> conv1(3): 46 -> pool(2): 23 -> conv2(3): 21 -> pool(2): 10
        # -> conv3(3): 8 -> pool(2): 4 => 256*4*4 = 4096
        s = downsample
        s = (s - 2) // 2   # conv1(3) + pool(2)
        s = (s - 2) // 2   # conv2(3) + pool(2)
        s = (s - 2) // 2   # conv3(3) + pool(2)
        self.fc_flat = 256 * s * s
        self._pool_sizes = []  # store for dynamic shape init

        # FC recurrent adLIF layer
        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        self.fc_rec = nn.Linear(n_fc, n_fc, bias=False)
        self.lif_fc = AdaptiveLIFNeuron(
            n_fc, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)

        # Readout
        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        init_beta_out = np.log(0.9 / 0.1)
        self.beta_out = nn.Parameter(torch.tensor(init_beta_out))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

    def forward(self, x):
        # x: (batch, T, 2, H, W) e.g. (batch, 30, 2, 48, 48)
        batch, T = x.shape[:2]
        device = x.device

        # Compute spatial sizes after each conv+pool
        H, W = x.shape[3], x.shape[4]
        h1, w1 = (H - 2) // 2, (W - 2) // 2
        h2, w2 = (h1 - 2) // 2, (w1 - 2) // 2
        h3, w3 = (h2 - 2) // 2, (w2 - 2) // 2

        # Conv membrane potentials
        v_c1 = torch.zeros(batch, 64, h1, w1, device=device)
        v_c2 = torch.zeros(batch, 128, h2, w2, device=device)
        v_c3 = torch.zeros(batch, 256, h3, w3, device=device)

        # FC recurrent states
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc = torch.zeros(batch, self.n_fc, device=device)
        a_fc = torch.zeros(batch, self.n_fc, device=device)

        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        beta_c1 = torch.sigmoid(self.beta_conv1_raw).view(1, 64, 1, 1)
        beta_c2 = torch.sigmoid(self.beta_conv2_raw).view(1, 128, 1, 1)
        beta_c3 = torch.sigmoid(self.beta_conv3_raw).view(1, 256, 1, 1)

        for t in range(T):
            # Conv layer 1
            c1 = self.bn1(self.conv1(x[:, t]))
            c1 = F.avg_pool2d(c1, 2)
            v_c1 = beta_c1 * v_c1 + (1.0 - beta_c1) * c1
            spk_c1 = surrogate_spike(v_c1 - self.threshold)
            v_c1 = v_c1 * (1.0 - spk_c1)

            # Conv layer 2
            c2 = self.bn2(self.conv2(spk_c1))
            c2 = F.avg_pool2d(c2, 2)
            v_c2 = beta_c2 * v_c2 + (1.0 - beta_c2) * c2
            spk_c2 = surrogate_spike(v_c2 - self.threshold)
            v_c2 = v_c2 * (1.0 - spk_c2)

            # Conv layer 3
            c3 = self.bn3(self.conv3(spk_c2))
            c3 = F.avg_pool2d(c3, 2)
            v_c3 = beta_c3 * v_c3 + (1.0 - beta_c3) * c3
            spk_c3 = surrogate_spike(v_c3 - self.threshold)
            v_c3 = v_c3 * (1.0 - spk_c3)

            # Flatten + recurrent FC adLIF
            flat = spk_c3.view(batch, -1)
            flat = self.dropout(flat) if self.training else flat
            I_fc = self.fc1(flat) + self.fc_rec(spk_fc)
            v_fc, spk_fc, a_fc = self.lif_fc(I_fc, v_fc, a_fc, spk_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc

            # Non-spiking readout
            I_out = self.fc_out(spk_fc_d)
            beta_o = torch.sigmoid(self.beta_out)
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        return out_sum / T


class DVSGestureSNN(nn.Module):
    """FC recurrent SNN for DVS128 Gesture classification (fallback).

    2048 -> 512 (recurrent adLIF) -> 256 (adLIF) -> 11 (non-spiking readout)
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=512, n_hidden2=256,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.3,
                 neuron_type='adlif', alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_init=0.95, beta_out=0.9):
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
            self.lif1 = AdaptiveLIFNeuron(n_hidden1, alpha_init=alpha_init,
                                           rho_init=rho_init, beta_a_init=beta_a_init,
                                           threshold=threshold)
            self.lif2 = AdaptiveLIFNeuron(n_hidden2, alpha_init=alpha_init,
                                           rho_init=rho_init, beta_a_init=beta_a_init,
                                           threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden1, beta_init=beta_init,
                                   threshold=threshold, learn_beta=True)
            self.lif2 = LIFNeuron(n_hidden2, beta_init=beta_init,
                                   threshold=threshold, learn_beta=True)

        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)
        self.dropout = nn.Dropout(p=dropout)

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
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden1, device=device)
            a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)

            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

            spk1_drop = self.dropout(spk1) if self.training else spk1
            I2 = self.fc2(spk1_drop)

            if self.neuron_type == 'adlif':
                v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            else:
                v2, spk2 = self.lif2(I2, v2)

            spk2_drop = self.dropout(spk2) if self.training else spk2

            I3 = self.fc3(spk2_drop)
            beta_out = self.lif_out.beta
            v_out = beta_out * v_out + (1.0 - beta_out) * I3
            out_sum = out_sum + v_out

        return out_sum / T


class WTAConvDVSGestureSNN(nn.Module):
    """Conv SNN with N3 WTA output for DVS128 Gesture.

    Same conv backbone as ConvDVSGestureSNN but replaces the FC adLIF
    layer with a WTALayer that enforces competitive sparse coding.
    Maps to N3 hardware WTA operation.

    Conv(2,32,5)+LIF -> Pool -> Conv(32,64,5)+LIF -> Pool ->
    Flatten(1600) -> 256 (WTA, groups=8, k=2) -> 11 (readout)
    """

    def __init__(self, n_fc=256, n_output=N_CLASSES, threshold=1.0,
                 dropout=0.3, beta_conv=0.9, n_wta_groups=8, wta_k=2):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output
        self.threshold = threshold
        self.aux_loss = None

        # Conv layers (same as ConvDVSGestureSNN)
        self.conv1 = nn.Conv2d(2, 32, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        init_bc = np.log(beta_conv / (1.0 - beta_conv))
        self.beta_conv1_raw = nn.Parameter(torch.full((32,), init_bc))
        self.beta_conv2_raw = nn.Parameter(torch.full((64,), init_bc))

        self.fc_flat = 64 * 5 * 5  # 1600

        # FC -> WTA layer (N3 hardware WTA operation)
        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        self.wta = WTALayer(n_fc, n_groups=n_wta_groups, k=wta_k,
                            beta_init=0.95, threshold=threshold)

        # Readout
        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        init_beta_out = np.log(0.9 / 0.1)
        self.beta_out = nn.Parameter(torch.tensor(init_beta_out))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

    def forward(self, x):
        batch, T = x.shape[:2]
        device = x.device

        v_c1 = torch.zeros(batch, 32, 14, 14, device=device)
        v_c2 = torch.zeros(batch, 64, 5, 5, device=device)
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        beta_c1 = torch.sigmoid(self.beta_conv1_raw).view(1, 32, 1, 1)
        beta_c2 = torch.sigmoid(self.beta_conv2_raw).view(1, 64, 1, 1)

        for t in range(T):
            c1 = self.bn1(self.conv1(x[:, t]))
            c1 = F.avg_pool2d(c1, 2)
            v_c1 = beta_c1 * v_c1 + (1.0 - beta_c1) * c1
            spk_c1 = surrogate_spike(v_c1 - self.threshold)
            v_c1 = v_c1 * (1.0 - spk_c1)

            c2 = self.bn2(self.conv2(spk_c1))
            c2 = F.avg_pool2d(c2, 2)
            v_c2 = beta_c2 * v_c2 + (1.0 - beta_c2) * c2
            spk_c2 = surrogate_spike(v_c2 - self.threshold)
            v_c2 = v_c2 * (1.0 - spk_c2)

            flat = spk_c2.view(batch, -1)
            flat = self.dropout(flat) if self.training else flat
            I_fc = self.fc1(flat)

            # WTA: competitive sparse coding
            v_fc, spk_fc = self.wta(I_fc, v_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc

            I_out = self.fc_out(spk_fc_d)
            beta_o = torch.sigmoid(self.beta_out)
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SNN on DVS128 Gesture")
    parser.add_argument("--data-dir", default="data/dvs_gesture")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--time-bins", type=int, default=20)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="dvs_gesture_model.pt")
    parser.add_argument("--event-drop", action="store_true", default=True)
    parser.add_argument("--spatial-jitter", action="store_true", default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed precision training (2-3x speedup)")
    parser.add_argument("--conv", action="store_true", default=False,
                        help="Use Conv SNN architecture instead of FC")
    parser.add_argument("--wta", action="store_true", default=False,
                        help="Use Conv + N3 WTA architecture")
    parser.add_argument("--wta-groups", type=int, default=8)
    parser.add_argument("--wta-k", type=int, default=2)
    parser.add_argument("--fc-hidden", type=int, default=256,
                        help="FC hidden size after conv layers (conv mode)")
    parser.add_argument("--deep", action="store_true", default=False,
                        help="Use deep 3-layer conv + recurrent FC (v2 architecture)")
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="Disable tonic event-level augmentation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    use_augment = not args.no_augment
    print(f"Loading DVS128 Gesture dataset... (event augment={'ON' if use_augment else 'OFF'})")
    use_conv = args.conv or args.wta or args.deep  # All conv variants
    train_ds = DVSGestureDataset(args.data_dir, train=True,
                                  n_time_bins=args.time_bins,
                                  downsample=args.downsample,
                                  flatten=not use_conv,
                                  augment=use_augment)
    test_ds = DVSGestureDataset(args.data_dir, train=False,
                                 n_time_bins=args.time_bins,
                                 downsample=args.downsample,
                                 flatten=not use_conv,
                                 augment=False)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    n_channels = 2 * args.downsample * args.downsample
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {args.time_bins}, Channels: {n_channels}")

    if args.deep:
        model = DeepConvDVSGestureSNN(
            n_fc=args.fc_hidden,
            threshold=args.threshold,
            dropout=args.dropout,
            downsample=args.downsample,
        ).to(device)
        print(f"Model: DEEP Conv(2,64,3)->Pool->Conv(64,128,3)->Pool->Conv(128,256,3)->Pool->"
              f"{args.fc_hidden}(rec adLIF)->{N_CLASSES} "
              f"(DEEP CONV+REC, dropout={args.dropout})")
    elif args.wta:
        model = WTAConvDVSGestureSNN(
            n_fc=args.fc_hidden,
            threshold=args.threshold,
            dropout=args.dropout,
            n_wta_groups=args.wta_groups,
            wta_k=args.wta_k,
        ).to(device)
        print(f"Model: Conv(2,32,5)->Pool->Conv(32,64,5)->Pool->"
              f"{args.fc_hidden}(WTA g={args.wta_groups} k={args.wta_k})->{N_CLASSES} "
              f"(N3 CONV+WTA, dropout={args.dropout})")
    elif use_conv:
        model = ConvDVSGestureSNN(
            n_fc=args.fc_hidden,
            threshold=args.threshold,
            dropout=args.dropout,
            downsample=args.downsample,
        ).to(device)
        print(f"Model: Conv(2,32,5)->Pool->Conv(32,64,5)->Pool->{args.fc_hidden}(adLIF)->{N_CLASSES} "
              f"(CONV+adLIF, dropout={args.dropout})")
    else:
        model = DVSGestureSNN(
            n_input=n_channels,
            n_hidden1=args.hidden1,
            n_hidden2=args.hidden2,
            threshold=args.threshold,
            dropout=args.dropout,
            neuron_type=args.neuron,
        ).to(device)
        print(f"Model: {n_channels}->{args.hidden1}->{args.hidden2}->{N_CLASSES} "
              f"({args.neuron.upper()}, recurrent=on, dropout={args.dropout})")

    spatial_dims = (args.downsample, args.downsample)

    def augment_fn(x):
        if args.event_drop:
            x = event_drop(x)
        if args.spatial_jitter:
            if use_conv:
                # Random spatial shift for conv mode (B, T, C, H, W)
                shift_h = random.randint(-2, 2)
                shift_w = random.randint(-2, 2)
                if shift_h != 0:
                    x = torch.roll(x, shifts=shift_h, dims=-2)
                if shift_w != 0:
                    x = torch.roll(x, shifts=shift_w, dims=-1)
            else:
                x = spatial_jitter(x, sigma=1.0, spatial_dims=spatial_dims)
        return x

    config = {
        'device': device,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_path': args.save,
        'benchmark': 'dvs_gesture',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': 0,
        'model_config': {
            'conv': use_conv,
            'wta': args.wta,
            'wta_groups': args.wta_groups if args.wta else None,
            'wta_k': args.wta_k if args.wta else None,
            'fc_hidden': args.fc_hidden,
            'n_input': n_channels,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': 'deep+rec+adlif' if args.deep else ('conv+wta' if args.wta else (args.neuron if not use_conv else 'conv+adlif')),
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
