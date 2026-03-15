"""Train SNN on N-MNIST benchmark.

Supports two architectures:
  FC:   2312 -> 512 (adLIF) -> 256 (adLIF) -> 10
  Conv: Conv(2,32,5)+LIF -> Pool -> Conv(32,64,5)+LIF -> Pool -> 256(adLIF) -> 10

Usage:
    python nmnist/train.py --conv --epochs 50 --device cuda:1 --amp
    python nmnist/train.py --epochs 50  # FC fallback
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
from common.training import run_training
from common.augmentation import event_drop

from nmnist.loader import NMNISTDataset, collate_fn, N_CHANNELS, N_CLASSES


class ConvNMNISTSNN(nn.Module):
    """Conv SNN for N-MNIST classification.

    Conv(2,32,5)+LIF -> Pool(2) -> Conv(32,64,5)+LIF -> Pool(2) ->
    Flatten(1600) -> 256 (adLIF) -> 10 (readout)
    """

    def __init__(self, n_fc=256, n_output=N_CLASSES, threshold=1.0,
                 dropout=0.3, alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_conv=0.9, neuron_type='adlif'):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output
        self.threshold = threshold
        self.neuron_type = neuron_type

        # Conv layers
        self.conv1 = nn.Conv2d(2, 32, 5, bias=False)   # (34,34) -> (30,30)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)  # (15,15) -> (11,11)
        self.bn2 = nn.BatchNorm2d(64)

        # LIF membrane decay for conv layers (per-channel, learnable)
        init_bc = np.log(beta_conv / (1.0 - beta_conv))
        self.beta_conv1_raw = nn.Parameter(torch.full((32,), init_bc))
        self.beta_conv2_raw = nn.Parameter(torch.full((64,), init_bc))

        # After conv2 + pool(2): 64 * 5 * 5 = 1600
        self.fc_flat = 64 * 5 * 5

        # FC layer
        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        if neuron_type == 'adlif':
            self.lif_fc = AdaptiveLIFNeuron(
                n_fc, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
        else:
            self.lif_fc = LIFNeuron(
                n_fc, beta_init=0.95, threshold=threshold, learn_beta=True)

        # Readout
        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        init_beta_out = np.log(0.9 / 0.1)  # logit so sigmoid(x) = 0.9
        self.beta_out = nn.Parameter(torch.tensor(init_beta_out))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

    def forward(self, x):
        # x: (batch, T, 2, 34, 34)
        batch, T = x.shape[:2]
        device = x.device

        # Conv LIF membrane potentials (per-channel decay)
        v_c1 = torch.zeros(batch, 32, 15, 15, device=device)
        v_c2 = torch.zeros(batch, 64, 5, 5, device=device)

        # FC states
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc = torch.zeros(batch, self.n_fc, device=device)
        if self.neuron_type == 'adlif':
            a_fc = torch.zeros(batch, self.n_fc, device=device)

        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        beta_c1 = torch.sigmoid(self.beta_conv1_raw).view(1, 32, 1, 1)
        beta_c2 = torch.sigmoid(self.beta_conv2_raw).view(1, 64, 1, 1)

        for t in range(T):
            # Conv layer 1: Conv -> BN -> Pool -> LIF
            c1 = self.bn1(self.conv1(x[:, t]))     # (batch, 32, 30, 30)
            c1 = F.avg_pool2d(c1, 2)                 # (batch, 32, 15, 15)
            v_c1 = beta_c1 * v_c1 + (1.0 - beta_c1) * c1
            spk_c1 = surrogate_spike(v_c1 - self.threshold)
            v_c1 = v_c1 * (1.0 - spk_c1)

            # Conv layer 2: Conv -> BN -> Pool -> LIF
            c2 = self.bn2(self.conv2(spk_c1))       # (batch, 64, 11, 11)
            c2 = F.avg_pool2d(c2, 2)                  # (batch, 64, 5, 5)
            v_c2 = beta_c2 * v_c2 + (1.0 - beta_c2) * c2
            spk_c2 = surrogate_spike(v_c2 - self.threshold)
            v_c2 = v_c2 * (1.0 - spk_c2)

            # Flatten + FC neuron
            flat = spk_c2.view(batch, -1)             # (batch, 1600)
            flat = self.dropout(flat) if self.training else flat
            I_fc = self.fc1(flat)
            if self.neuron_type == 'adlif':
                v_fc, spk_fc, a_fc = self.lif_fc(I_fc, v_fc, a_fc, spk_fc)
            else:
                v_fc, spk_fc = self.lif_fc(I_fc, v_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc

            # Non-spiking readout
            I_out = self.fc_out(spk_fc_d)
            beta_o = torch.sigmoid(self.beta_out)
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        return out_sum / T


class DeepConvNMNISTSNN(nn.Module):
    """3-layer Conv SNN for N-MNIST (N3 deep architecture).

    Conv(2,32,5)+LIF -> Pool -> Conv(32,64,3)+LIF -> Pool ->
    Conv(64,128,3,pad=1)+LIF -> Pool -> Flatten -> 256 (adLIF) -> 10

    Spatial flow: 34->30->15 -> 13->6 -> 6->3 -> flatten(128*3*3=1152)
    """

    def __init__(self, n_fc=256, n_output=N_CLASSES, threshold=1.0,
                 dropout=0.3, alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_conv=0.9):
        super().__init__()
        self.n_fc = n_fc
        self.n_output = n_output
        self.threshold = threshold

        self.conv1 = nn.Conv2d(2, 32, 5, bias=False)       # 34->30
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)      # 15->13
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)  # 6->6
        self.bn3 = nn.BatchNorm2d(128)

        init_bc = np.log(beta_conv / (1.0 - beta_conv))
        self.beta_conv1_raw = nn.Parameter(torch.full((32,), init_bc))
        self.beta_conv2_raw = nn.Parameter(torch.full((64,), init_bc))
        self.beta_conv3_raw = nn.Parameter(torch.full((128,), init_bc))

        # After pool: 30->15, 13->6, 6->3 => 128*3*3=1152
        self.fc_flat = 128 * 3 * 3

        self.fc1 = nn.Linear(self.fc_flat, n_fc, bias=False)
        self.lif_fc = AdaptiveLIFNeuron(
            n_fc, alpha_init=alpha_init, rho_init=rho_init,
            beta_a_init=beta_a_init, threshold=threshold)

        self.fc_out = nn.Linear(n_fc, n_output, bias=False)
        init_beta_out = np.log(0.9 / 0.1)
        self.beta_out = nn.Parameter(torch.tensor(init_beta_out))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

    def forward(self, x):
        batch, T = x.shape[:2]
        device = x.device

        v_c1 = torch.zeros(batch, 32, 15, 15, device=device)
        v_c2 = torch.zeros(batch, 64, 6, 6, device=device)
        v_c3 = torch.zeros(batch, 128, 3, 3, device=device)
        v_fc = torch.zeros(batch, self.n_fc, device=device)
        spk_fc = torch.zeros(batch, self.n_fc, device=device)
        a_fc = torch.zeros(batch, self.n_fc, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        beta_c1 = torch.sigmoid(self.beta_conv1_raw).view(1, 32, 1, 1)
        beta_c2 = torch.sigmoid(self.beta_conv2_raw).view(1, 64, 1, 1)
        beta_c3 = torch.sigmoid(self.beta_conv3_raw).view(1, 128, 1, 1)

        for t in range(T):
            c1 = self.bn1(self.conv1(x[:, t]))       # (B,32,30,30)
            c1 = F.avg_pool2d(c1, 2)                   # (B,32,15,15)
            v_c1 = beta_c1 * v_c1 + (1.0 - beta_c1) * c1
            spk_c1 = surrogate_spike(v_c1 - self.threshold)
            v_c1 = v_c1 * (1.0 - spk_c1)

            c2 = self.bn2(self.conv2(spk_c1))         # (B,64,13,13)
            c2 = F.avg_pool2d(c2, 2)                    # (B,64,6,6)
            v_c2 = beta_c2 * v_c2 + (1.0 - beta_c2) * c2
            spk_c2 = surrogate_spike(v_c2 - self.threshold)
            v_c2 = v_c2 * (1.0 - spk_c2)

            c3 = self.bn3(self.conv3(spk_c2))         # (B,128,6,6)
            c3 = F.avg_pool2d(c3, 2)                    # (B,128,3,3)
            v_c3 = beta_c3 * v_c3 + (1.0 - beta_c3) * c3
            spk_c3 = surrogate_spike(v_c3 - self.threshold)
            v_c3 = v_c3 * (1.0 - spk_c3)

            flat = spk_c3.view(batch, -1)
            flat = self.dropout(flat) if self.training else flat
            I_fc = self.fc1(flat)
            v_fc, spk_fc, a_fc = self.lif_fc(I_fc, v_fc, a_fc, spk_fc)
            spk_fc_d = self.dropout(spk_fc) if self.training else spk_fc

            I_out = self.fc_out(spk_fc_d)
            beta_o = torch.sigmoid(self.beta_out)
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        return out_sum / T


class NMNISTSNN(nn.Module):
    """Feedforward FC SNN for N-MNIST classification (original).

    2312 -> 512 (adLIF) -> 256 (adLIF) -> 10 (non-spiking readout)
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=512, n_hidden2=256,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.2,
                 neuron_type='adlif', alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_init=0.95, beta_out=0.9):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.neuron_type = neuron_type

        self.fc1 = nn.Linear(n_input, n_hidden1, bias=False)
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

        init_bo = np.log(beta_out / (1.0 - beta_out))
        self.beta_out_param = nn.Parameter(torch.tensor(init_bo))
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=3.0)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)

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
            I1 = self.fc1(x[:, t])
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
            beta_out = torch.sigmoid(self.beta_out_param)
            v_out = beta_out * v_out + (1.0 - beta_out) * I3
            out_sum = out_sum + v_out

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SNN on N-MNIST")
    parser.add_argument("--data-dir", default="data/nmnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--time-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="nmnist_model.pt")
    parser.add_argument("--event-drop", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--conv", action="store_true", default=False,
                        help="Use Conv SNN architecture instead of FC")
    parser.add_argument("--n3-deep", action="store_true", default=False,
                        help="Use 3-layer deep Conv SNN (N3 architecture)")
    parser.add_argument("--fc-hidden", type=int, default=256,
                        help="FC hidden size after conv layers (conv mode)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading N-MNIST dataset...")
    use_conv = args.conv or args.n3_deep
    train_ds = NMNISTDataset(args.data_dir, train=True, n_time_bins=args.time_bins,
                              flatten=not use_conv)
    test_ds = NMNISTDataset(args.data_dir, train=False, n_time_bins=args.time_bins,
                             flatten=not use_conv)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {args.time_bins}")

    if args.n3_deep:
        model = DeepConvNMNISTSNN(
            n_fc=args.fc_hidden,
            threshold=args.threshold,
            dropout=args.dropout,
        ).to(device)
        print(f"Model: Conv(2,32,5)->Pool->Conv(32,64,3)->Pool->Conv(64,128,3)->Pool->"
              f"{args.fc_hidden}(adLIF)->{N_CLASSES} (N3 DEEP CONV, dropout={args.dropout})")
    elif use_conv:
        model = ConvNMNISTSNN(
            n_fc=args.fc_hidden,
            threshold=args.threshold,
            dropout=args.dropout,
            neuron_type=args.neuron,
        ).to(device)
        ntype = args.neuron.upper()
        print(f"Model: Conv(2,32,5)->Pool->Conv(32,64,5)->Pool->{args.fc_hidden}({ntype})->{N_CLASSES} "
              f"(CONV+{ntype}, dropout={args.dropout})")
    else:
        model = NMNISTSNN(
            n_hidden1=args.hidden1,
            n_hidden2=args.hidden2,
            threshold=args.threshold,
            dropout=args.dropout,
            neuron_type=args.neuron,
        ).to(device)
        print(f"Model: {N_CHANNELS}->{args.hidden1}->{args.hidden2}->{N_CLASSES} "
              f"({args.neuron.upper()}, dropout={args.dropout})")

    augment_fn = (lambda x: event_drop(x)) if args.event_drop else None

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
        'warmup_epochs': 0,
        'model_config': {
            'conv': use_conv,
            'n3_deep': args.n3_deep,
            'fc_hidden': args.fc_hidden,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
