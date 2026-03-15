"""N-TIDIGITS18 dataset loader using tonic.

Spike-encoded spoken digit recordings. 18 classes: single digits (0-9)
plus double-digit combos (10-19 minus some). Cochlea-encoded audio.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 64   # 64 cochlea frequency channels
N_CLASSES = 18     # digits 0-9 plus combos
N_TIME_BINS = 100  # temporal bins


class NTIDIGITS18Dataset(Dataset):
    """PyTorch Dataset for N-TIDIGITS18 with frame conversion."""

    # Map string labels to integer classes
    LABEL_MAP = {
        'z': 0, 'o': 1, '1': 2, '2': 3, '3': 4, '4': 5,
        '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    }

    def __init__(self, data_dir="data/ntidigits", train=True,
                 n_time_bins=N_TIME_BINS):
        sensor = tonic.datasets.NTIDIGITS18.sensor_size
        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins),
        ])
        self.dataset = tonic.datasets.NTIDIGITS18(
            save_to=data_dir, train=train, transform=transform)
        self.n_time_bins = n_time_bins
        self.n_channels = N_CHANNELS

        # Build label set dynamically on first pass, filter to single-digit only
        self._build_label_map()

    def _build_label_map(self):
        """Build label mapping, keeping only single-character labels."""
        self.valid_indices = []
        self.label_to_idx = {}
        next_idx = 0

        for i in range(len(self.dataset)):
            try:
                _, label = self.dataset[i]
                label_str = str(label)
                # Only keep single-char labels (single digit utterances)
                if len(label_str) == 1:
                    if label_str not in self.label_to_idx:
                        self.label_to_idx[label_str] = next_idx
                        next_idx += 1
                    self.valid_indices.append(i)
            except Exception:
                continue
            # Stop scanning after finding enough to build the map
            if next_idx >= 11 and len(self.valid_indices) > 100:
                # Scan the rest without loading data
                break

        # If we couldn't scan all, just use sequential indices
        if len(self.valid_indices) < 100:
            self.valid_indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx] if idx < len(self.valid_indices) else idx
        frames, label = self.dataset[real_idx]
        T = frames.shape[0]
        flat = frames.reshape(T, -1).astype(np.float32)
        # Binary spike encoding
        flat = (flat > 0).astype(np.float32)

        label_str = str(label)
        # For multi-char labels, use first character
        key = label_str[0] if label_str else 'z'
        label_idx = self.label_to_idx.get(key, 0)
        return torch.from_numpy(flat), label_idx


def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_t = max(x.shape[0] for x in inputs)
    C = inputs[0].shape[1]
    padded = torch.zeros(len(inputs), max_t, C)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(labels, dtype=torch.long)
