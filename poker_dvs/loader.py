"""Poker-DVS dataset loader using tonic.

DVS recordings of poker card pips (suits). 4-class classification.
35x35 pixels, binned into frames, flattened.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 2450  # 35 * 35 * 2
N_CLASSES = 4      # club, diamond, heart, spade
SENSOR_SIZE = (35, 35, 2)
N_TIME_BINS = 10


class PokerDVSDataset(Dataset):
    """PyTorch Dataset for Poker-DVS with frame conversion."""

    def __init__(self, data_dir="data/poker_dvs", train=True,
                 n_time_bins=N_TIME_BINS):
        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.POKERDVS(
            save_to=data_dir, train=train, transform=transform)
        self.n_time_bins = n_time_bins
        self.n_channels = N_CHANNELS

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames, label = self.dataset[idx]
        T = frames.shape[0]
        flat = frames.reshape(T, -1).astype(np.float32)
        flat = (flat > 0).astype(np.float32)
        return torch.from_numpy(flat), int(label)


def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_t = max(x.shape[0] for x in inputs)
    C = inputs[0].shape[1]
    padded = torch.zeros(len(inputs), max_t, C)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(labels, dtype=torch.long)
