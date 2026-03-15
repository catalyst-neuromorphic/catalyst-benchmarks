"""Primate Reaching neural motor decoding dataset.

NeuroBench benchmark: decode hand velocity from neural spike recordings.
Uses the Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex
Electrophysiology dataset (Makin et al.).

96 electrode channels -> predict 2D hand velocity (x, y).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None


N_CHANNELS = 96   # electrode channels
N_OUTPUT = 2      # velocity x, y
HISTORY_LEN = 50  # 50ms bins
BIN_SIZE_MS = 20  # 20ms time bins


class PrimateReachDataset(Dataset):
    """Primate reaching motor decoding dataset.

    If the NeuroBench HDF5 file is available, loads from it.
    Otherwise generates synthetic data matching the statistics
    for architecture validation.
    """

    def __init__(self, data_dir="data/primate_reach", train=True,
                 history_len=HISTORY_LEN, synthetic=True):
        self.history_len = history_len

        h5_path = os.path.join(data_dir, "indy_20160627_01.mat")
        if not synthetic and os.path.exists(h5_path) and h5py is not None:
            self._load_real(h5_path, train)
        else:
            self._generate_synthetic(train)

    def _generate_synthetic(self, train):
        """Generate synthetic reaching data for architecture validation."""
        np.random.seed(42 if train else 123)
        n_trials = 800 if train else 200

        self.inputs = []
        self.targets = []

        for _ in range(n_trials):
            # Simulate 96-channel neural spikes during a reach
            T = self.history_len
            # Base firing rates
            rates = np.random.uniform(0.05, 0.3, size=(1, N_CHANNELS))
            # Reach direction modulates rates
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.2, 1.0)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            # Preferred directions for each neuron
            pref = np.random.uniform(0, 2 * np.pi, size=(1, N_CHANNELS))
            modulation = 0.15 * np.cos(pref - angle)

            # Generate spikes
            spike_probs = np.clip(rates + modulation, 0, 1)
            spike_probs = np.tile(spike_probs, (T, 1))
            spikes = (np.random.random((T, N_CHANNELS)) < spike_probs).astype(np.float32)

            self.inputs.append(spikes)
            self.targets.append(np.array([vx, vy], dtype=np.float32))

        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

    def _load_real(self, h5_path, train):
        """Load real neural data from HDF5."""
        with h5py.File(h5_path, 'r') as f:
            spikes = np.array(f['spikes'])  # (n_bins, 96)
            cursor_vel = np.array(f['cursor_vel'])  # (n_bins, 2)

        # Window the data
        n = len(spikes) - self.history_len
        self.inputs = np.zeros((n, self.history_len, N_CHANNELS), dtype=np.float32)
        self.targets = np.zeros((n, N_OUTPUT), dtype=np.float32)

        for i in range(n):
            self.inputs[i] = spikes[i:i + self.history_len]
            self.targets[i] = cursor_vel[i + self.history_len]

        # 80/20 split
        n_train = int(0.8 * n)
        if train:
            self.inputs = self.inputs[:n_train]
            self.targets = self.targets[:n_train]
        else:
            self.inputs = self.inputs[n_train:]
            self.targets = self.targets[n_train:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]),
                torch.from_numpy(self.targets[idx]))


def collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)
