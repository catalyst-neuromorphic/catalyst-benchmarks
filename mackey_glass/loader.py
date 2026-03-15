"""Mackey-Glass chaotic time series dataset.

NeuroBench benchmark: predict future values of the Mackey-Glass delay
differential equation. Regression task — predict y(t+1) from recent history.

The Mackey-Glass equation:
  dy/dt = beta * y(t-tau) / (1 + y(t-tau)^n) - gamma * y(t)

Standard parameters: beta=0.2, gamma=0.1, tau=17, n=10
"""

import numpy as np
import torch
from torch.utils.data import Dataset


N_CHANNELS = 1   # scalar time series (spike-encoded)
N_OUTPUT = 1     # regression target
HISTORY_LEN = 200  # number of past timesteps as input
PREDICT_AHEAD = 1


def generate_mackey_glass(n_samples=12000, tau=17, beta=0.2, gamma=0.1,
                          n=10, dt=1.0, warmup=500):
    """Generate Mackey-Glass time series via Euler integration."""
    total = n_samples + warmup + tau
    y = np.zeros(total)
    y[0] = 1.2  # initial condition

    for t in range(tau, total - 1):
        y_tau = y[t - tau]
        dy = beta * y_tau / (1.0 + y_tau ** n) - gamma * y[t]
        y[t + 1] = y[t] + dt * dy

    # Discard warmup
    return y[warmup:warmup + n_samples].astype(np.float32)


class MackeyGlassDataset(Dataset):
    """Mackey-Glass time series prediction.

    Input: spike-encoded history window of length HISTORY_LEN
    Target: next value y(t+1)
    """

    def __init__(self, train=True, history_len=HISTORY_LEN,
                 predict_ahead=PREDICT_AHEAD, n_samples=12000,
                 spike_threshold=0.02):
        self.history_len = history_len
        self.spike_threshold = spike_threshold

        # Generate full series
        series = generate_mackey_glass(n_samples=n_samples)

        # Normalize to [0, 1]
        self.series_min = series.min()
        self.series_max = series.max()
        series = (series - self.series_min) / (self.series_max - self.series_min + 1e-8)

        # Create windows
        n_windows = len(series) - history_len - predict_ahead
        self.inputs = np.zeros((n_windows, history_len, 1), dtype=np.float32)
        self.targets = np.zeros((n_windows, 1), dtype=np.float32)

        for i in range(n_windows):
            self.inputs[i, :, 0] = series[i:i + history_len]
            self.targets[i, 0] = series[i + history_len + predict_ahead - 1]

        # 80/20 train/test split
        n_train = int(0.8 * n_windows)
        if train:
            self.inputs = self.inputs[:n_train]
            self.targets = self.targets[:n_train]
        else:
            self.inputs = self.inputs[n_train:]
            self.targets = self.targets[n_train:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]  # (history_len, 1)

        # Spike encode: delta encoding
        spikes = np.zeros_like(x)
        for t in range(1, len(x)):
            delta = np.abs(x[t] - x[t - 1])
            spikes[t] = (delta > self.spike_threshold).astype(np.float32)

        return torch.from_numpy(spikes), torch.tensor(self.targets[idx])


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets
