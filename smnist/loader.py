"""Sequential MNIST (S-MNIST) dataset loader.

Standard MNIST images presented pixel-by-pixel as a temporal sequence.
Classic SNN/RNN benchmark — tests temporal memory over 784 timesteps.
Uses torchvision (reliable PyPI download, no tonic needed).
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


N_CHANNELS = 1   # one pixel at a time
N_CLASSES = 10   # digits 0-9
SEQ_LEN = 784    # 28 * 28 pixels presented sequentially


class SMNISTDataset(Dataset):
    """Sequential MNIST — pixels presented one at a time."""

    def __init__(self, data_dir="data/smnist", train=True,
                 spike_threshold=0.5):
        self.spike_threshold = spike_threshold
        self.dataset = datasets.MNIST(
            root=data_dir, train=train, download=True,
            transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img: (1, 28, 28) -> flatten to (784,) then to (784, 1)
        seq = img.flatten().unsqueeze(-1)  # (784, 1)
        # Rate-code to spikes
        spikes = (seq > self.spike_threshold).float()
        return spikes, label


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
