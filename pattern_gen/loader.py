"""Spike Pattern Generation benchmark (self-generating).

Classify temporal spike patterns — tests the SNN's ability to learn
complex spatiotemporal features. No download needed.

8 distinct spike patterns with varying frequency, phase, and correlation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


N_CHANNELS = 50   # spike input channels
N_CLASSES = 8     # pattern classes
SEQ_LEN = 100     # timesteps


def generate_pattern(pattern_id, n_channels=N_CHANNELS, seq_len=SEQ_LEN):
    """Generate a distinct spatiotemporal spike pattern."""
    spikes = np.zeros((seq_len, n_channels), dtype=np.float32)
    rng = np.random.default_rng()

    if pattern_id == 0:  # Synchronous burst
        burst_times = rng.choice(seq_len, size=10, replace=False)
        for t in burst_times:
            spikes[t, :] = (rng.random(n_channels) < 0.8).astype(np.float32)

    elif pattern_id == 1:  # Wave (propagating front)
        for ch in range(n_channels):
            delay = int(ch * seq_len / n_channels)
            width = 5
            for t in range(max(0, delay - width), min(seq_len, delay + width)):
                spikes[t, ch] = 1.0 if rng.random() < 0.7 else 0.0

    elif pattern_id == 2:  # High frequency random
        spikes = (rng.random((seq_len, n_channels)) < 0.15).astype(np.float32)

    elif pattern_id == 3:  # Low frequency random
        spikes = (rng.random((seq_len, n_channels)) < 0.03).astype(np.float32)

    elif pattern_id == 4:  # Oscillating (alternating groups)
        period = 10
        for t in range(seq_len):
            phase = (t // period) % 2
            if phase == 0:
                spikes[t, :n_channels // 2] = (
                    rng.random(n_channels // 2) < 0.3).astype(np.float32)
            else:
                spikes[t, n_channels // 2:] = (
                    rng.random(n_channels - n_channels // 2) < 0.3).astype(
                    np.float32)

    elif pattern_id == 5:  # Ramping (increasing rate)
        for t in range(seq_len):
            rate = 0.01 + 0.3 * (t / seq_len)
            spikes[t] = (rng.random(n_channels) < rate).astype(np.float32)

    elif pattern_id == 6:  # Clustered (spatial groups)
        n_clusters = 5
        cluster_size = n_channels // n_clusters
        active_cluster = rng.integers(n_clusters)
        start = active_cluster * cluster_size
        end = start + cluster_size
        for t in range(seq_len):
            spikes[t, start:end] = (
                rng.random(end - start) < 0.15).astype(np.float32)

    elif pattern_id == 7:  # Decay (decreasing rate)
        for t in range(seq_len):
            rate = 0.3 * (1.0 - t / seq_len) + 0.01
            spikes[t] = (rng.random(n_channels) < rate).astype(np.float32)

    # Add noise
    noise_mask = rng.random((seq_len, n_channels)) < 0.02
    spikes = np.logical_xor(spikes > 0, noise_mask).astype(np.float32)

    return spikes


class PatternGenDataset(Dataset):
    """Spike pattern classification dataset (generated)."""

    def __init__(self, n_samples=5000, train=True):
        np.random.seed(42 if train else 123)
        n_per_class = n_samples // N_CLASSES
        self.inputs = []
        self.labels = []

        for cls in range(N_CLASSES):
            for _ in range(n_per_class):
                pattern = generate_pattern(cls)
                self.inputs.append(pattern)
                self.labels.append(cls)

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

        # Shuffle
        perm = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]),
                int(self.labels[idx]))


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
