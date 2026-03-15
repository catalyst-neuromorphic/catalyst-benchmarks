"""ASL-DVS (American Sign Language) dataset loader using tonic.

Event-camera recordings of ASL letters. 24 classes (no J or Z — require motion).
240x180 pixels downsampled to 32x32, binned into frames, flattened.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 2048  # 32 * 32 * 2
N_CLASSES = 24
SENSOR_SIZE = (240, 180, 2)
DOWNSAMPLE_SIZE = 32
N_TIME_BINS = 10


class ASLDVSDataset(Dataset):
    """PyTorch Dataset for ASL-DVS with frame conversion."""

    def __init__(self, data_dir="data/asl_dvs", train=True,
                 n_time_bins=N_TIME_BINS, downsample=DOWNSAMPLE_SIZE):
        sensor = (downsample, downsample, 2)
        scale = downsample / 240  # downsample from 240x180

        transform = transforms.Compose([
            transforms.Downsample(spatial_factor=scale),
            transforms.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.ASLDVS(
            save_to=data_dir, transform=transform)
        self.n_time_bins = n_time_bins
        self.n_channels = downsample * downsample * 2

        # 80/20 train/test split (no built-in split)
        n = len(self.dataset)
        n_train = int(0.8 * n)
        if train:
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        frames, label = self.dataset[real_idx]
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
