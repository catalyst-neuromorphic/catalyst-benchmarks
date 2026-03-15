"""Sequential CIFAR-10 (sCIFAR-10) benchmark.

CIFAR-10 images fed one pixel at a time (R, G, B channels per step).
3072 timesteps (32*32*3), 3 channels per step, 10 classes.
Tests very long-range temporal dependency.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError("torchvision required: pip install torchvision")


N_CHANNELS = 3   # RGB per timestep
N_CLASSES = 10
SEQ_LEN = 1024   # 32 * 32 pixels, each with 3 channels


class SCIFAR10Dataset(Dataset):
    """Sequential CIFAR-10 — pixels fed one at a time with RGB channels."""

    def __init__(self, data_dir="data/scifar10", train=True,
                 spike_threshold=0.3):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True,
            transform=transform)
        self.spike_threshold = spike_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img: (3, 32, 32) -> reshape to (1024, 3) — pixel-by-pixel with RGB
        pixels = img.permute(1, 2, 0).reshape(-1, 3).numpy()  # (1024, 3)

        # Rate-code to spikes
        spikes = (pixels > self.spike_threshold).astype(np.float32)

        return torch.from_numpy(spikes), int(label)


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
