"""N-Cars (Prophesee Automotive) dataset loader using tonic.

Binary classification: car vs. background from a DVS car dashcam.
Events binned into frames, downsampled, and flattened.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 1800  # 30 * 30 * 2 (downsampled + flattened)
N_CLASSES = 2
SENSOR_SIZE = (100, 120, 2)  # N-Cars native resolution
DOWNSAMPLE_SIZE = 30
N_TIME_BINS = 10


class NCarsDataset(Dataset):
    """PyTorch Dataset for N-Cars with frame conversion."""

    def __init__(self, data_dir="data/ncars", train=True,
                 n_time_bins=N_TIME_BINS, downsample=DOWNSAMPLE_SIZE):
        sensor = (downsample, downsample, 2)
        transform = transforms.Compose([
            transforms.Downsample(spatial_factor=downsample / 100),
            transforms.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.NCARS(
            save_to=data_dir, train=train, transform=transform)
        self.n_time_bins = n_time_bins
        self.n_channels = downsample * downsample * 2

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
