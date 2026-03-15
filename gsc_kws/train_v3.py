"""Train d-cAdLIF-inspired SNN on Google Speech Commands (35-class).

Architecture inspired by Hammouamri et al. 2024 (d-cAdLIF, 95.69% GSC-35):
  - Pure feedforward (no recurrence — delays provide temporal memory)
  - Learnable synaptic delays on BOTH layers (DCLS-style)
  - BatchNorm through time on both layers
  - S2S binary spike encoding (40 mel, delta modulation)

Architecture: input -> DelayedLinear(40, 512) + BN + adLIF
                    -> DelayedLinear(512, 512) + BN + adLIF
                    -> Linear(512, 35) readout

Target: 95.4%+ on GSC v2 35-class (within 0.3% of d-cAdLIF 95.69%)

Usage:
    python gsc_kws/train_v3.py --epochs 300 --device cuda:1 --amp
    python gsc_kws/train_v3.py --hidden 512 --max-delay 60 --threshold 1.0 --amp
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


class DCGSC_SNN(nn.Module):
    """Delay-cAdLIF-inspired feedforward SNN for GSC-35.

    Pure feedforward architecture with learnable delays on BOTH layers.
    No recurrence — delays provide the temporal memory needed for
    speech recognition. Matches d-cAdLIF design (Hammouamri et al. 2024).

    Architecture:
        input -> DelayedLinear(in, H) + BN + adLIF [delays on input→hidden1]
              -> DelayedLinear(H, H) + BN + adLIF  [delays on hidden1→hidden2]
              -> Linear(H, n_classes) readout

    Key differences from our v2 (which collapsed):
    - No recurrence (cleaner gradient flow, delays handle temporal patterns)
    - Delays on both layers (d-cAdLIF's key innovation)
    - Same-size hidden layers (512/512 not 1024/512)
    - S2S binary encoding (not N3 continuous)
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden=512,
                 n_output=N_CLASSES_35, threshold=0.3, dropout=0.2,
                 alpha_init=0.85, rho_init=0.85, beta_a_init=0.05,
                 beta_out=0.9, max_delay=60,
                 target_rate=0.05, activity_lambda=0.005):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None
        self.max_delay = max_delay

        # Layer 1: delayed input + BN + adLIF
        self.delay_fc1 = DelayedLinear(n_input, n_hidden,
                                        max_delay=max_delay, bias=False)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.lif1 = AdaptiveLIFNeuron(n_hidden, alpha_init=alpha_init,
                                        rho_init=rho_init, beta_a_init=beta_a_init,
                                        threshold=threshold)

        # Layer 2: delayed hidden + BN + adLIF
        self.delay_fc2 = DelayedLinear(n_hidden, n_hidden,
                                        max_delay=max_delay, bias=False)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.lif2 = AdaptiveLIFNeuron(n_hidden, alpha_init=alpha_init,
                                        rho_init=rho_init, beta_a_init=beta_a_init,
                                        threshold=threshold)

        # Readout (non-spiking leaky integrator)
        self.fc_out = nn.Linear(n_hidden, n_output, bias=False)
        self.lif_out = LIFNeuron(n_output, beta_init=beta_out,
                                  threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        # Weight init — gain=2.0 for sparse binary S2S input
        nn.init.xavier_uniform_(self.delay_fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.delay_fc2.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

        # Initialize delays near zero (sigmoid(-3) ≈ 0.047 → delay ≈ 1.4 steps)
        # Default randn*0.1 gives sigmoid≈0.5 → delay≈max_delay/2, which is too
        # much for two delay layers on 100-timestep data. Start small, learn to grow.
        with torch.no_grad():
            self.delay_fc1.delay_raw.fill_(-3.0)
            self.delay_fc2.delay_raw.fill_(-3.0)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        # Pre-compute delayed sequences for both layers
        x_delayed = self.delay_fc1.apply_delays(x)  # (batch, T, n_input)

        # State variables — layer 1
        v1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        a1 = torch.zeros(batch, self.n_hidden, device=device)

        # State variables — layer 2
        v2 = torch.zeros(batch, self.n_hidden, device=device)
        spk2 = torch.zeros(batch, self.n_hidden, device=device)
        a2 = torch.zeros(batch, self.n_hidden, device=device)

        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)
        spike_count1 = torch.zeros(batch, self.n_hidden, device=device)
        spike_count2 = torch.zeros(batch, self.n_hidden, device=device)

        # Collect layer 1 output spikes for delay_fc2
        spk1_seq = []

        # Pass 1: compute layer 1 spikes for all timesteps
        for t in range(T):
            I1_raw = self.delay_fc1(x_delayed[:, t])
            I1 = self.bn1(I1_raw)
            v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            spike_count1 = spike_count1 + spk1
            # Dropout on OUTPUT going to L2 (not on internal spike feedback)
            spk1_out = self.dropout(spk1) if self.training else spk1
            spk1_seq.append(spk1_out)

        # Stack L1 output spikes and apply per-synapse delays
        spk1_tensor = torch.stack(spk1_seq, dim=1)  # (batch, T, n_hidden)
        spk1_delayed = self.delay_fc2.apply_delays(spk1_tensor)  # (batch, T, n_hidden)

        # Pass 2: layer 2 + readout using delayed L1 spikes
        for t in range(T):
            I2_raw = self.delay_fc2(spk1_delayed[:, t])
            I2 = self.bn2(I2_raw)
            v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            spike_count2 = spike_count2 + spk2

            # Non-spiking readout
            spk2_d = self.dropout(spk2) if self.training else spk2
            I_out = self.fc_out(spk2_d)
            beta_o = self.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

        # Activity regularization
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
        description="Train d-cAdLIF-inspired SNN on GSC-35")
    parser.add_argument("--data-dir", default="data/gsc")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--max-delay", type=int, default=60,
                        help="Maximum learnable delay in timesteps")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha-init", type=float, default=0.85)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="checkpoints/gsc_v3a.pt")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--activity-lambda", type=float, default=0.005)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--twelve-class", action="store_true", default=False,
                        help="Use 12-class KWS task instead of 35-class")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--encoding", choices=['s2s', 'n3'], default='s2s',
                        help="Input encoding: s2s (delta modulation, recommended) or n3")
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

    # N3 continuous features scale to [-1, 1]; S2S already in {-1, 0, 1}
    input_scale = 1.0 / 128.0 if encoding == 'n3' else 1.0

    model = DCGSC_SNN(
        n_input=n_input,
        n_hidden=args.hidden,
        n_output=n_classes,
        threshold=args.threshold,
        dropout=args.dropout,
        alpha_init=args.alpha_init,
        max_delay=args.max_delay,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
    ).to(device)

    print(f"Model: {n_input}->[delay:{args.max_delay}]->"
          f"{args.hidden}(BN+adLIF)->[delay:{args.max_delay}]->"
          f"{args.hidden}(BN+adLIF)->{n_classes} "
          f"(FF+DUAL-DELAY, dropout={args.dropout}, alpha={args.alpha_init})")

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
            'neuron_type': 'dc-adlif-ff',
            'hidden': args.hidden,
            'max_delay': args.max_delay,
            'n_input': n_input,
            'n_output': n_classes,
            'encoding': encoding,
            'full_35': full_35,
            'threshold': args.threshold,
            'dropout': args.dropout,
            'alpha_init': args.alpha_init,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
