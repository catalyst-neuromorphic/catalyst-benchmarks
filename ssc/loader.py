"""SSC (Spiking Speech Commands) dataset loader.

SSC is the 35-class version of the SHD benchmark with more speakers and
more classes. Same 700 cochlea channels, same HDF5 format.

700 input channels, 35 classes.
Download: https://zenodo.org/records/3515935
"""

import os
import urllib.request
import gzip
import shutil
import numpy as np

import torch
from torch.utils.data import Dataset


SSC_URLS = {
    "train": "https://compneuro.net/datasets/ssc_train.h5.gz",
    "valid": "https://compneuro.net/datasets/ssc_valid.h5.gz",
    "test": "https://compneuro.net/datasets/ssc_test.h5.gz",
}

N_CHANNELS = 700
N_CLASSES = 35


def download_ssc(data_dir="data/ssc"):
    """Download SSC HDF5 files if not present."""
    import h5py  # noqa: F401
    os.makedirs(data_dir, exist_ok=True)

    for split, url in SSC_URLS.items():
        h5_path = os.path.join(data_dir, f"ssc_{split}.h5")
        gz_path = h5_path + ".gz"

        if os.path.exists(h5_path):
            continue

        print(f"Downloading SSC {split} set from {url} ...")
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {url}: {e}\n"
                f"Download manually from https://zenodo.org/records/3515935 "
                f"and place files in {data_dir}/")

        print(f"Extracting {gz_path} ...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(h5_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  Saved to {h5_path}")

    return data_dir


def spikes_to_dense(times, units, n_channels=N_CHANNELS, dt=4e-3, max_time=1.0):
    """Convert spike events to dense binary tensor (T, n_channels)."""
    n_bins = int(max_time / dt)
    dense = np.zeros((n_bins, n_channels), dtype=np.float32)

    if len(times) == 0:
        return dense

    bin_indices = np.clip((times / dt).astype(int), 0, n_bins - 1)
    unit_indices = np.clip(units.astype(int), 0, n_channels - 1)
    np.add.at(dense, (bin_indices, unit_indices), 1.0)
    return dense


class SSCDataset(Dataset):
    """PyTorch Dataset for Spiking Speech Commands."""

    def __init__(self, data_dir="data/ssc", split="train", dt=4e-3, max_time=1.0):
        import h5py

        h5_path = os.path.join(data_dir, f"ssc_{split}.h5")
        if not os.path.exists(h5_path):
            download_ssc(data_dir)

        with h5py.File(h5_path, 'r') as f:
            self.times = [np.array(t) for t in f['spikes']['times']]
            self.units = [np.array(u) for u in f['spikes']['units']]
            self.labels = np.array(f['labels'])

        self.dt = dt
        self.max_time = max_time
        self.n_bins = int(max_time / dt)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dense = spikes_to_dense(
            self.times[idx], self.units[idx],
            dt=self.dt, max_time=self.max_time,
        )
        return torch.from_numpy(dense), int(self.labels[idx])


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
