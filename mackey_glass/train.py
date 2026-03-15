"""Train SNN on Mackey-Glass chaotic time series prediction.

Architecture: 1 -> hidden (recurrent adLIF) -> 1
Regression task: predict next value of Mackey-Glass equation.

Usage:
    python mackey_glass/train.py --epochs 100 --device cuda:0
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

from mackey_glass.loader import (MackeyGlassDataset, collate_fn,
                                  N_CHANNELS, N_OUTPUT)


class MackeyGlassSNN(nn.Module):
    """Recurrent SNN for Mackey-Glass time series prediction."""

    def __init__(self, n_input=N_CHANNELS, n_hidden=128, n_output=N_OUTPUT,
                 beta_hidden=0.95, beta_out=0.9, threshold=1.0, dropout=0.1,
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

        # Process full sequence, use final hidden state for prediction
        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)
            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

        # Use final membrane voltage for regression output
        spk1_d = self.dropout(v1) if self.training else v1
        out = self.fc2(spk1_d)
        return out


def train_epoch_regression(model, loader, optimizer, device, clip_norm=1.0):
    model.train()
    total_loss = 0
    n = 0
    loss_fn = nn.MSELoss()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        n += inputs.size(0)

    return total_loss / n


def evaluate_regression(model, loader, device):
    model.eval()
    total_mse = 0
    n = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            mse = ((output - targets) ** 2).sum().item()
            total_mse += mse
            n += inputs.size(0)

    return total_mse / n


def main():
    parser = argparse.ArgumentParser(description="Train SNN on Mackey-Glass")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="mackey_glass_model.pt")
    parser.add_argument("--neuron", choices=["lif", "adlif"], default="adlif")
    parser.add_argument("--alpha-init", type=float, default=0.93)
    parser.add_argument("--rho-init", type=float, default=0.85)
    parser.add_argument("--beta-a-init", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Generating Mackey-Glass time series...")
    train_ds = MackeyGlassDataset(train=True)
    test_ds = MackeyGlassDataset(train=False)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    model = MackeyGlassSNN(
        n_hidden=args.hidden, dropout=args.dropout, neuron_type=args.neuron,
        alpha_init=args.alpha_init, rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 1->{args.hidden}->1 ({args.neuron.upper()}, recurrent=on)")
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_mse = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_mse = train_epoch_regression(model, train_loader, optimizer, device)
        test_mse = evaluate_regression(model, test_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        marker = ""
        if test_mse < best_mse:
            best_mse = test_mse
            torch.save(model.state_dict(), args.save)
            marker = " *BEST*"
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Ep {epoch+1:3d}/{args.epochs} | "
              f"Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f} | "
              f"Best: {best_mse:.6f} | LR: {lr:.2e}{marker}")

        if patience_counter >= 30:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nFinal best test MSE: {best_mse:.6f}")
    print(f"Final best test RMSE: {best_mse**0.5:.6f}")


if __name__ == "__main__":
    main()
