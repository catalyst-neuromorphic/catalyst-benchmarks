"""DVS128 Gesture dataset loader using tonic.

DVS128 Gesture is an event-based gesture recognition dataset captured
with a 128x128 DVS camera. 11 gesture classes.

Events are downsampled to 32x32, binned into frames, and flattened.

Disk caching: frames are processed once via tonic and cached as .pt files
to prevent numpy OOM on repeated epoch access (tonic re-processes raw events
every __getitem__ call, causing memory fragmentation after ~60 epochs).
"""

import gc
import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")


N_CHANNELS = 2048  # 32 * 32 * 2 (downsampled + flattened)
N_CLASSES = 11
SENSOR_SIZE = (128, 128, 2)
DOWNSAMPLE_SIZE = 32
N_TIME_BINS = 20


class DVSGestureDataset(Dataset):
    """PyTorch Dataset for DVS128 Gesture with frame conversion.

    Downsamples 128x128 -> 32x32, bins into frames.
    Returns (T, 2, 32, 32) for conv mode or (T, 2048) for FC mode.

    Uses disk caching to avoid tonic re-processing OOM.
    """

    def __init__(self, data_dir="data/dvs_gesture", train=True,
                 n_time_bins=N_TIME_BINS, downsample=DOWNSAMPLE_SIZE,
                 flatten=True, augment=False):
        self.downsample = downsample
        self.flatten = flatten
        self.augment = augment and train
        sensor = (downsample, downsample, 2)

        # Cache directory
        split = "train" if train else "test"
        cache_dir = os.path.join(data_dir, f"cache_{split}_{downsample}_{n_time_bins}")
        self.cache_dir = cache_dir

        if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
            # Load from cache
            self.n_samples = len([f for f in os.listdir(cache_dir) if f.endswith('.pt')])
            print(f"DVS Gesture: loaded {self.n_samples} cached frames from {cache_dir}")
            self.dataset = None
        else:
            # Build tonic dataset and cache all frames
            os.makedirs(cache_dir, exist_ok=True)
            base_transforms = [
                transforms.Downsample(spatial_factor=downsample / 128),
                transforms.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins),
            ]
            transform = transforms.Compose(base_transforms)
            dataset = tonic.datasets.DVSGesture(
                save_to=data_dir, train=train, transform=transform)

            print(f"DVS Gesture: caching {len(dataset)} samples to {cache_dir}...")
            for i in range(len(dataset)):
                frames, label = dataset[i]
                out = (frames.astype(np.float32) > 0).astype(np.float32)
                tensor = torch.from_numpy(out)
                torch.save((tensor, int(label)), os.path.join(cache_dir, f"{i:05d}.pt"))
                if (i + 1) % 100 == 0:
                    gc.collect()
                    print(f"  cached {i+1}/{len(dataset)}")

            self.n_samples = len(dataset)
            self.dataset = None
            del dataset
            gc.collect()
            print(f"DVS Gesture: caching complete ({self.n_samples} samples)")

        self.n_time_bins = n_time_bins

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        tensor, label = torch.load(
            os.path.join(self.cache_dir, f"{idx:05d}.pt"),
            weights_only=True)
        # tensor: (T, 2, 32, 32)
        if self.flatten:
            T = tensor.shape[0]
            tensor = tensor.reshape(T, -1)  # (T, 2048)
        return tensor, label


def collate_fn(batch):
    """Collate with padding to max time length.

    Handles both flat (T, C) and conv (T, 2, H, W) inputs.
    """
    inputs, labels = zip(*batch)
    max_t = max(x.shape[0] for x in inputs)
    rest_shape = inputs[0].shape[1:]
    padded = torch.zeros(len(inputs), max_t, *rest_shape)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(labels, dtype=torch.long)
