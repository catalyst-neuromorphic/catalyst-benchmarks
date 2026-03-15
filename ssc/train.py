"""Train SNN on Spiking Speech Commands (SSC) benchmark.

v5 N3: Two-layer recurrent architecture with activity regularization.
  - v4 features: adLIF, recurrent, dropout, AMP
  - NEW: Recurrence on BOTH layers, recurrent dropout on layer 1
  - NEW: Activity regularization
  - Architecture: 700 -> 1024 (rec adLIF) -> 768 (rec adLIF) -> 35 (readout)
  - Target: 85%+ float (current SOTA: 85.98% SpikCommander)

Usage:
    python ssc/train.py --data-dir data/ssc --epochs 300 --amp --device cuda:0
    python ssc/train.py --recurrent2 --hidden2 768 --epochs 300 --amp
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

from ssc.loader import SSCDataset, collate_fn, N_CHANNELS, N_CLASSES


class SSCSNN(nn.Module):
    """Recurrent SNN for SSC classification.

    700 -> 1024 (recurrent adLIF) -> 512 (adLIF) -> 35 (non-spiking readout)
    """

    def __init__(self, n_input=N_CHANNELS, n_hidden1=1024, n_hidden2=512,
                 n_output=N_CLASSES, threshold=1.0, dropout=0.3,
                 neuron_type='adlif', alpha_init=0.95, rho_init=0.85,
                 beta_a_init=0.05, beta_init=0.95, beta_out=0.9,
                 target_rate=0.05, activity_lambda=0.01, recurrent2=False):
        super().__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.neuron_type = neuron_type
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss = None
        self.recurrent2 = recurrent2

        self.fc1 = nn.Linear(n_input, n_hidden1, bias=False)
        self.fc_rec = nn.Linear(n_hidden1, n_hidden1, bias=False)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        if recurrent2:
            self.fc_rec2 = nn.Linear(n_hidden2, n_hidden2, bias=False)
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
        if recurrent2:
            nn.init.orthogonal_(self.fc_rec2.weight, gain=0.2)
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
        spk2_d = torch.zeros(batch, self.n_hidden2, device=device)

        # Activity regularization
        spike_count1 = torch.zeros(batch, self.n_hidden1, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden1, device=device)
            a2 = torch.zeros(batch, self.n_hidden2, device=device)

        for t in range(T):
            # Recurrent connection uses DROPPED spikes (like SHD v7)
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1_d)

            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

            # Apply dropout BEFORE recurrent feedback
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count1 = spike_count1 + spk1

            # Layer 2 with optional recurrence
            I2 = self.fc2(spk1_d)
            if self.recurrent2:
                I2 = I2 + self.fc_rec2(spk2_d)

            if self.neuron_type == 'adlif':
                v2, spk2, a2 = self.lif2(I2, v2, a2, spk2)
            else:
                v2, spk2 = self.lif2(I2, v2)

            spk2_d = self.dropout(spk2) if self.training else spk2

            I3 = self.fc3(spk2_d)
            beta_out = self.lif_out.beta
            v_out = beta_out * v_out + (1.0 - beta_out) * I3
            out_sum = out_sum + v_out

        # Activity regularization
        if self.training and self.activity_lambda > 0:
            mean_rate = spike_count1 / T
            self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
        else:
            self.aux_loss = None

        return out_sum / T


def main():
    parser = argparse.ArgumentParser(description="Train SNN on SSC")
    parser.add_argument("--data-dir", default="data/ssc")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--recurrent2", action="store_true", default=False,
                        help="Enable recurrence on layer 2 (N3 v5 architecture)")
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="ssc_model.pt")
    parser.add_argument("--event-drop", action="store_true", default=True)
    parser.add_argument("--time-stretch", action="store_true", default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--activity-lambda", type=float, default=0.01)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--alpha-init", type=float, default=0.95)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--resume-weights", default=None,
                        help="Load model weights from a checkpoint (warm-start, no optimizer state)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed precision training (2-3x speedup)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading SSC dataset...")
    train_ds = SSCDataset(args.data_dir, "train", dt=args.dt)
    test_ds = SSCDataset(args.data_dir, "test", dt=args.dt)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {train_ds.n_bins} (dt={args.dt*1000:.1f}ms)")

    model = SSCSNN(
        n_hidden1=args.hidden1,
        n_hidden2=args.hidden2,
        threshold=args.threshold,
        dropout=args.dropout,
        neuron_type=args.neuron,
        recurrent2=args.recurrent2,
        alpha_init=args.alpha_init,
        rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
        target_rate=args.target_rate,
        activity_lambda=args.activity_lambda,
    ).to(device)

    if args.resume_weights:
        ckpt_data = torch.load(args.resume_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
        print(f"Loaded weights from {args.resume_weights} "
              f"(acc={ckpt_data.get('test_acc', 0)*100:.2f}%)")

    rec2_str = "+rec2" if args.recurrent2 else ""
    print(f"Model: {N_CHANNELS}->{args.hidden1}(rec)->{args.hidden2}{rec2_str}->{N_CLASSES} "
          f"({args.neuron.upper()}, dropout={args.dropout})")

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
        'benchmark': 'ssc',
        'augment_fn': augment_fn,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'warmup_epochs': args.warmup_epochs,
        'gc_every': 5,  # gc.collect() every 5 epochs to prevent numpy OOM
        'model_config': {
            'n_input': N_CHANNELS,
            'hidden': args.hidden1,
            'n_hidden2': args.hidden2,
            'n_output': N_CLASSES,
            'threshold': args.threshold,
            'neuron_type': args.neuron,
            'dropout': args.dropout,
            'recurrent2': args.recurrent2,
        },
    }

    run_training(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
