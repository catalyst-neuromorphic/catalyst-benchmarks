"""Train SNN on Primate Reaching motor decoding benchmark.

Architecture: 96 -> hidden (recurrent adLIF) -> 2
Regression: decode hand velocity from neural spikes.

Usage:
    python primate_reach/train.py --epochs 100 --device cuda:0
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

from primate_reach.loader import (PrimateReachDataset, collate_fn,
                                   N_CHANNELS, N_OUTPUT)


class PrimateReachSNN(nn.Module):
    """Recurrent SNN for neural motor decoding."""

    def __init__(self, n_input=N_CHANNELS, n_hidden=256, n_output=N_OUTPUT,
                 beta_hidden=0.95, threshold=1.0, dropout=0.1,
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

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        # Process sequence, accumulate readout
        out_sum = torch.zeros(batch, self.n_output, device=device)
        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)
            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

        # Final state readout
        spk1_d = self.dropout(v1) if self.training else v1
        return self.fc2(spk1_d)


def main():
    parser = argparse.ArgumentParser(description="Train SNN on Primate Reaching")
    parser.add_argument("--data-dir", default="data/primate_reach")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="primate_reach_model.pt")
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--alpha-init", type=float, default=0.93)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading Primate Reaching dataset...")
    train_ds = PrimateReachDataset(args.data_dir, train=True,
                                    synthetic=args.synthetic)
    test_ds = PrimateReachDataset(args.data_dir, train=False,
                                   synthetic=args.synthetic)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    model = PrimateReachSNN(
        n_hidden=args.hidden, dropout=args.dropout, neuron_type=args.neuron,
        alpha_init=args.alpha_init, rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {N_CHANNELS}->{args.hidden}->{N_OUTPUT} "
          f"({args.neuron.upper()}, recurrent=on)")
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    loss_fn = nn.MSELoss()
    best_mse = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)
        train_loss /= n_train

        model.eval()
        test_loss = 0
        n_test = 0
        # Also compute R² (coefficient of determination)
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                test_loss += loss_fn(output, targets).item() * inputs.size(0)
                n_test += inputs.size(0)
                all_preds.append(output.cpu())
                all_targets.append(targets.cpu())
        test_loss /= n_test

        preds = torch.cat(all_preds)
        targs = torch.cat(all_targets)
        ss_res = ((preds - targs) ** 2).sum()
        ss_tot = ((targs - targs.mean(dim=0)) ** 2).sum()
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8)).item()

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        marker = ""
        if test_loss < best_mse:
            best_mse = test_loss
            best_r2 = r2
            torch.save(model.state_dict(), args.save)
            marker = " *BEST*"
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Ep {epoch+1:3d}/{args.epochs} | "
              f"Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} | "
              f"R²: {r2:.4f} | LR: {lr:.2e}{marker}")

        if patience_counter >= 30:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nFinal best test MSE: {best_mse:.6f}")
    print(f"Final best R²: {best_r2:.4f}")


if __name__ == "__main__":
    main()
