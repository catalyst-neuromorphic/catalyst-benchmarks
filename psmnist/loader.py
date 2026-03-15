"""Permuted Sequential MNIST (PS-MNIST) benchmark.

THE standard benchmark for long-range temporal dependency in SNNs.
Pixels presented one at a time in a FIXED RANDOM PERMUTATION order,
destroying all spatial structure. The SNN must rely purely on
temporal memory to classify the digit.

784 timesteps, 1 channel, 10 classes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError("torchvision required: pip install torchvision")


N_CHANNELS = 1   # one pixel at a time
N_CLASSES = 10
SEQ_LEN = 784    # 28 * 28


class PSMNISTDataset(Dataset):
    """Permuted Sequential MNIST — pixels in fixed random order."""

    def __init__(self, data_dir="data/psmnist", train=True,
                 spike_threshold=0.5, seed=42):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(
            root=data_dir, train=train, download=True,
            transform=transform)
        self.spike_threshold = spike_threshold

        # Fixed random permutation (same for all samples)
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(SEQ_LEN)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img: (1, 28, 28) -> flatten to (784,)
        pixels = img.view(-1).numpy()

        # Apply fixed permutation
        pixels = pixels[self.perm]

        # Rate-code to spikes: (784, 1)
        spikes = np.zeros((SEQ_LEN, 1), dtype=np.float32)
        spikes[:, 0] = (pixels > self.spike_threshold).astype(np.float32)

        return torch.from_numpy(spikes), int(label)


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
