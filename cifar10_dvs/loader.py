"""CIFAR10-DVS dataset loader using tonic.

CIFAR-10 images recorded with a 128x128 DVS camera. 10 classes.
Events are binned into frames and flattened for FC-SNN processing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 2048  # 32 * 32 * 2 (downsampled + flattened)
N_CLASSES = 10
SENSOR_SIZE = (128, 128, 2)
DOWNSAMPLE_SIZE = 32
N_TIME_BINS = 10


class CIFAR10DVSDataset(Dataset):
    """PyTorch Dataset for CIFAR10-DVS with frame conversion.

    Downsamples 128x128 -> 32x32, bins into frames, flattens to (T, 2048).
    """

    def __init__(self, data_dir="data/cifar10_dvs", train=True,
                 n_time_bins=N_TIME_BINS, downsample=DOWNSAMPLE_SIZE):
        self.downsample = downsample
        sensor = (downsample, downsample, 2)

        transform = transforms.Compose([
            transforms.Downsample(spatial_factor=downsample / 128),
            transforms.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.CIFAR10DVS(
            save_to=data_dir, transform=transform)
        self.n_time_bins = n_time_bins

        # CIFAR10-DVS doesn't have a built-in train/test split
        # Use 90/10 split
        n = len(self.dataset)
        n_train = int(0.9 * n)
        if train:
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        frames, label = self.dataset[real_idx]
        # frames: (T, 2, H, W) -> flatten to (T, 2*H*W)
        T = frames.shape[0]
        flat = frames.reshape(T, -1).astype(np.float32)
        flat = (flat > 0).astype(np.float32)
        return torch.from_numpy(flat), int(label)


def collate_fn(batch):
    """Collate with padding to max time length."""
    inputs, labels = zip(*batch)
    max_t = max(x.shape[0] for x in inputs)
    C = inputs[0].shape[1]
    padded = torch.zeros(len(inputs), max_t, C)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(labels, dtype=torch.long)
