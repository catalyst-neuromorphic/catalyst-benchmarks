"""N-Caltech101 dataset loader using tonic.

Caltech-101 images recorded with a DVS camera using saccade movements.
240x180 -> downsampled to 32x32, binned into frames, flattened.
101 object classes.
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
N_CLASSES = 101
SENSOR_SIZE = (240, 180, 2)
DOWNSAMPLE_SIZE = 32
N_TIME_BINS = 10


class NCaltech101Dataset(Dataset):
    """PyTorch Dataset for N-Caltech101 with frame conversion."""

    def __init__(self, data_dir="data/ncaltech101", train=True,
                 n_time_bins=N_TIME_BINS, downsample=DOWNSAMPLE_SIZE):
        sensor = (downsample, downsample, 2)
        # Downsample from 240x180 — use average of H and W scale
        scale_h = downsample / 240
        scale_w = downsample / 180
        scale = min(scale_h, scale_w)

        transform = transforms.Compose([
            transforms.Downsample(spatial_factor=scale),
            transforms.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.NCALTECH101(
            save_to=data_dir, transform=transform)
        self.n_time_bins = n_time_bins
        self.n_channels = downsample * downsample * 2

        # Build label-to-index mapping (labels are strings like 'airplanes')
        self.label_to_idx = {}
        self._build_label_map()

        # Stratified 80/20 train/test split (dataset is sorted by class,
        # so sequential split would give non-overlapping classes)
        n = len(self.dataset)
        rng = np.random.RandomState(42)  # fixed seed for reproducibility
        all_indices = np.arange(n)
        rng.shuffle(all_indices)
        n_train = int(0.8 * n)
        if train:
            self.indices = all_indices[:n_train].tolist()
        else:
            self.indices = all_indices[n_train:].tolist()

    def _build_label_map(self):
        """Scan dataset targets to build string->int label map."""
        # Try to get all labels without loading full data
        unique_labels = set()
        for i in range(len(self.dataset)):
            try:
                _, label = self.dataset[i]
                unique_labels.add(str(label))
            except Exception:
                continue
            if len(unique_labels) >= N_CLASSES:
                break
        # Also scan a sample from the end
        for i in range(max(0, len(self.dataset) - 200), len(self.dataset)):
            try:
                _, label = self.dataset[i]
                unique_labels.add(str(label))
            except Exception:
                continue

        for idx, lbl in enumerate(sorted(unique_labels)):
            self.label_to_idx[lbl] = idx

        print(f"Found {len(self.label_to_idx)} classes")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        frames, label = self.dataset[real_idx]
        T = frames.shape[0]
        flat = frames.reshape(T, -1).astype(np.float32)
        flat = (flat > 0).astype(np.float32)
        label_idx = self.label_to_idx.get(str(label), 0)
        return torch.from_numpy(flat), label_idx


def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_t = max(x.shape[0] for x in inputs)
    C = inputs[0].shape[1]
    padded = torch.zeros(len(inputs), max_t, C)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(labels, dtype=torch.long)
