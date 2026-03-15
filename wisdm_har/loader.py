"""WISDM Human Activity Recognition dataset loader.

Accelerometer data from smartphones — 6-class activity classification.
Uses raw accelerometer (x, y, z) time series converted to spike trains.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


N_CHANNELS = 3  # x, y, z accelerometer
N_CLASSES = 6   # Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
N_TIME_BINS = 200  # time steps per sample (window of 200 readings at 20Hz = 10s)
WINDOW_SIZE = 200
STRIDE = 100  # 50% overlap


ACTIVITY_MAP = {
    'Walking': 0, 'Jogging': 1, 'Upstairs': 2,
    'Downstairs': 3, 'Sitting': 4, 'Standing': 5,
}


class WISDMDataset(Dataset):
    """PyTorch Dataset for WISDM accelerometer activity recognition.

    Downloads and parses WISDM_ar_v1.1_raw.txt, windows the data,
    and converts to spike trains via threshold encoding.
    """

    def __init__(self, data_dir="data/wisdm", train=True,
                 window_size=WINDOW_SIZE, stride=STRIDE, spike_threshold=0.5):
        self.window_size = window_size
        self.spike_threshold = spike_threshold

        raw_file = os.path.join(data_dir, "WISDM_ar_v1.1",
                                "WISDM_ar_v1.1_raw.txt")
        if not os.path.exists(raw_file):
            # Try alternate path
            raw_file = os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(
                f"WISDM dataset not found at {raw_file}. "
                "Download from: https://www.cis.fordham.edu/wisdm/dataset.php "
                f"and extract to {data_dir}/")

        # Parse raw accelerometer data
        samples = []
        labels = []
        for line in open(raw_file, 'r'):
            line = line.strip().rstrip(';').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                activity = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5].rstrip(';'))
                if activity in ACTIVITY_MAP:
                    samples.append([x, y, z])
                    labels.append(ACTIVITY_MAP[activity])
            except (ValueError, IndexError):
                continue

        samples = np.array(samples, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Normalize per-channel
        mean = samples.mean(axis=0)
        std = samples.std(axis=0) + 1e-8
        samples = (samples - mean) / std

        # Window the data
        self.windows = []
        self.labels = []
        for start in range(0, len(samples) - window_size, stride):
            window = samples[start:start + window_size]
            # Use majority label
            window_labels = labels[start:start + window_size]
            majority = np.bincount(window_labels).argmax()
            self.windows.append(window)
            self.labels.append(int(majority))

        self.windows = np.array(self.windows)
        self.labels = np.array(self.labels)

        # 80/20 train/test split
        n = len(self.windows)
        n_train = int(0.8 * n)
        if train:
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        window = self.windows[real_idx]  # (T, 3)

        # Convert to spike trains via step-forward encoding
        # Delta encoding: spike when change exceeds threshold
        spikes = np.zeros_like(window, dtype=np.float32)
        for t in range(1, len(window)):
            delta = np.abs(window[t] - window[t - 1])
            spikes[t] = (delta > self.spike_threshold).astype(np.float32)

        return torch.from_numpy(spikes), self.labels[real_idx]


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)  # All same size (window_size, 3)
    return inputs, torch.tensor(labels, dtype=torch.long)
