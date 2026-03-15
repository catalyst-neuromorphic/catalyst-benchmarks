"""Train delay-enhanced SNN on Google Speech Commands (35-class).

Implements learnable synaptic delays inspired by d-cAdLIF (Hammouamri et al.
2024, Frontiers in Neuroscience) which achieves 95.69% on GSC-35 with 0.61M
params. Our architecture adds delays to the N3 adLIF architecture that already
dominates SHD and SSC benchmarks.

Architecture: input -> DelayedLinear + rec adLIF -> Linear + rec adLIF -> 35
Target: 96%+ on GSC v2 35-class (beat d-cAdLIF 95.69%, compete with
SpikCommander 97.08%)

Usage:
    python gsc_kws/train_v2.py --epochs 300 --device cuda:1 --amp
    python gsc_kws/train_v2.py --hidden1 1024 --hidden2 512 --max-delay 30 --amp
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

from common.neurons import AdaptiveLIFNeuron, LIFNeuron, DelayedLinear, surrogate_spike
from common.training import run_training
from common.augmentation import event_drop, time_stretch, spec_augment

from gsc_kws.loader import (GSCDataset, collate_fn, N_CHANNELS,
                              N_CLASSES, N_CLASSES_35)


class DelayGSCSNN(nn.Module):
    """Delay-enhanced two-layer recurrent SNN for GSC-35.

    Learnable synaptic delays on the input layer allow the network to
    temporally align features across frequency bands -- critical for
    speech recognition where formant transitions span multiple mel bands
    with different onset times.

    Architecture:
        input -> DelayedLinear(in, H1) + rec(H1,H1) + adLIF [delays on input]
              -> Linear(H1, H2) + rec(H2,H2) + adLIF      [second recurrent]
              -> Linear(H2, n_classes) readout

    Inspired by d-cAdLIF (95.69%, 0.61M params) and RadLIF (94.51%).
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=512,
                 n_output=N_CLASSES_35, threshold=1.0, dropout=0.3,
                 alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 beta_out=0.9, max_delay=30,
                 target_rate=0.05, activity_lambda=0.01,
                 input_scale=1.0):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None
        self.max_delay = max_delay
        self.input_scale = input_scale

        # Layer 1: delayed input + recurrent adLIF + BN (BNTT-style)
        self.delay_fc1 = DelayedLinear(n_input, n_hidden1,
                                        max_delay=max_delay, bias=False)
        self.fc_rec1 = nn.Linear(n_hidden1, n_hidden1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.lif1 = AdaptiveLIFNeuron(n_hidden1, alpha_init=alpha_init,
                                        rho_init=rho_init, beta_a_init=beta_a_init,
                                        threshold=threshold)

        # Layer 2: feedforward + recurrent adLIF + BN
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.fc_rec2 = nn.Linear(n_hidden2, n_hidden2, bias=False)
        self.bn2 = nn.BatchNorm1d(n_hidden2)
        self.lif2 = AdaptiveLIFNeuron(n_hidden2, alpha_init=alpha_init,
                                        rho_init=rho_init, beta_a_init=beta_a_init,
                                        threshold=threshold)

        # Readout (non-spiking leaky integrator)
        self.fc_out = nn.Linear(n_hidden2, n_output, bias=False)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        # Init — gain tuned so ~30% neurons spike at init (for surrogate gradient flow)
        # S2S binary (40 ch): gain=2.0 (sparse binary input needs stronger weights)
        # N3 continuous (120 ch, scaled /128→[-1,1]): gain=1.0 (dense continuous input)
        input_gain = 1.0 if n_input > 50 else 2.0
        nn.init.xavier_uniform_(self.delay_fc1.weight, gain=input_gain)
        nn.init.orthogonal_(self.fc_rec1.weight, gain=0.3)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.fc_rec2.weight, gain=0.3)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

        # Initialize delays near zero (sigmoid(-3) ≈ 0.047 → delay ≈ 1.4 steps)
        # Default randn*0.1 gives sigmoid≈0.5 → delay≈max_delay/2, too much lag.
        with torch.no_grad():
            self.delay_fc1.delay_raw.fill_(-3.0)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # Scale input (N3 continuous features are [-128,127], need ~[-1,1])
        if self.input_scale != 1.0:
            x = x * self.input_scale

        # Pre-compute delayed input sequence (differentiable)
        x_delayed = self.delay_fc1.apply_delays(x)  # (batch, T, n_input)

        # State variables
        v1 = torch.zeros(batch, self.n_hidden1, device=device)
        spk1 = torch.zeros(batch, self.n_hidden1, device=device)
        a1 = torch.zeros(batch, self.n_hidden1, device=device)

        v2 = torch.zeros(batch, self.n_hidden2, device=device)
        spk2 = torch.zeros(batch, self.n_hidden2, device=device)
        a2 = torch.zeros(batch, self.n_hidden2, device=device)

        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count1 = torch.zeros(batch, self.n_hidden1, device=device)
        spike_count2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            # Layer 1: delayed input + recurrence + BN (BNTT)
            I1_raw = self.delay_fc1(x_delayed[:, t]) + self.fc_rec1(
                self.dropout(spk1) if self.training else spk1)
            I1 = self.bn1(I1_raw)
            v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            spike_count1 = spike_count1 + spk1

            # Layer 2: feedforward from layer 1 + recurrence + BN
            spk1_d = self.dropout(spk1) if self.training else spk1
            I2_raw = self.fc2(spk1_d) + self.fc_rec2(
                self.dropout(spk2) if self.training else spk2)
            I2 = self.bn2(I2_raw)
            v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            spike_count2 = spike_count2 + spk2

            # Non-spiking readout
            spk2_d = self.dropout(spk2) if self.training else spk2
            I_out = self.fc_out(spk2_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        # Activity regularization on both layers
        if self.training and self.activity_lambda > 0:
            rate1 = spike_count1 / T
            rate2 = spike_count2 / T
            self.aux_loss = self.activity_lambda * (
                ((rate1 - self.target_rate) ** 2).mean() +
                ((rate2 - self.target_rate) ** 2).mean())
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(
        description="Train delay-enhanced SNN on GSC-35")
    parser.add_argument("--data-dir", default="data/gsc")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--max-delay", type=int, default=30,
                        help="Maximum learnable delay in timesteps")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="checkpoints/gsc_n3_delay.pt")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--activity-lambda", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--twelve-class", action="store_true", default=False,
                        help="Use 12-class KWS task instead of 35-class")
    parser.add_argument("--encoding", choices=['s2s', 'n3'], default='s2s',
                        help="Input encoding: s2s (delta modulation) or n3 (continuous mel)")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--resume-weights", default=None,
                        help="Path to checkpoint to warm-start from")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    full_35 = not args.twelve_class
    n_classes = N_CLASSES_35 if full_35 else N_CLASSES
    encoding = args.encoding
    n_mels = args.n_mels

    enc_name = "continuous mel+d+d2" if encoding == 'n3' else "Speech2Spikes"
    class_str = "35-class" if full_35 else "12-class"
    print(f"Loading GSC v2 dataset ({class_str}, {enc_name}, {n_mels} mel)...")

    train_ds = GSCDataset(args.data_dir, split="training",
                           n_channels=n_mels, max_time_bins=args.time_bins,
                           threshold=args.threshold, encoding=encoding,
                           full_35=full_35)
    test_ds = GSCDataset(args.data_dir, split="testing",
                          n_channels=n_mels, max_time_bins=args.time_bins,
                          threshold=args.threshold, encoding=encoding,
                          full_35=full_35)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    n_input = n_mels * 3 if encoding == 'n3' else n_mels
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {args.time_bins}, Input: {n_input}")

    # N3 continuous features are [-128, 127] — scale to [-1, 1] for spiking nets
    input_scale = 1.0 / 128.0 if encoding == 'n3' else 1.0

    model = DelayGSCSNN(
        n_input=n_input,
        n_hidden1=args.hidden1,
        n_hidden2=args.hidden2,
        n_output=n_classes,
        threshold=args.threshold,
        dropout=args.dropout,
        max_delay=args.max_delay,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
        input_scale=input_scale,
    ).to(device)

    print(f"Model: {n_input}->[delay:{args.max_delay}]->"
          f"{args.hidden1}(rec adLIF)->{args.hidden2}(rec adLIF)->{n_classes} "
          f"(DELAY+DUAL-REC, dropout={args.dropout})")

    # Warm-start from existing checkpoint if specified
    if args.resume_weights and os.path.exists(args.resume_weights):
        print(f"Loading weights from {args.resume_weights}")
        ckpt = torch.load(args.resume_weights, map_location=device,
                           weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    def augment_fn(x):
        x = event_drop(x)
        x = time_stretch(x, factor_range=(0.9, 1.1))
        x = spec_augment(x, freq_mask_width=4, time_mask_width=8,
                          n_freq_masks=2, n_time_masks=2)
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
        'warmup_epochs': args.warmup_epochs,
        'gc_every': 5,
        'model_config': {
            'neuron_type': 'delay-dual-rec-adlif',
            'hidden': args.hidden1,
            'hidden2': args.hidden2,
            'max_delay': args.max_delay,
            'n_input': n_input,
            'n_output': n_classes,
            'encoding': encoding,
            'full_35': full_35,
            'threshold': args.threshold,
            'dropout': args.dropout,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
