"""N-MNIST dataset loader using tonic.

N-MNIST is a neuromorphic version of MNIST captured with a DVS camera.
34x34 spatial resolution, 2 polarities, 10 digit classes.

Uses tonic for download and event-to-frame conversion.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 2312  # 34 * 34 * 2 (flattened DVS frames)
N_CLASSES = 10
SENSOR_SIZE = (34, 34, 2)
N_TIME_BINS = 20


class NMNISTDataset(Dataset):
    """PyTorch Dataset wrapping tonic N-MNIST with frame conversion.

    Each sample is converted to a dense tensor.
    - flatten=True (default): (T, 2312) for FC architectures
    - flatten=False: (T, 2, 34, 34) for Conv architectures
    """

    def __init__(self, data_dir="data/nmnist", train=True, n_time_bins=N_TIME_BINS,
                 flatten=True):
        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.NMNIST(
            save_to=data_dir, train=train, transform=transform)
        self.n_time_bins = n_time_bins
        self.flatten = flatten

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames, label = self.dataset[idx]
        # frames: (T, 2, 34, 34)
        T = frames.shape[0]
        if self.flatten:
            out = frames.reshape(T, -1).astype(np.float32)
            out = (out > 0).astype(np.float32)
        else:
            out = frames.astype(np.float32)
            out = (out > 0).astype(np.float32)
        return torch.from_numpy(out), int(label)


def collate_fn(batch):
    """Collate N-MNIST samples, padding to max time length."""
    inputs, labels = zip(*batch)
    max_t = max(x.shape[0] for x in inputs)
    shape = inputs[0].shape[1:]  # either (2312,) or (2, 34, 34)
    padded = torch.zeros(len(inputs), max_t, *shape)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(labels, dtype=torch.long)
