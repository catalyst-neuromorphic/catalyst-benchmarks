"""MIT-BIH Arrhythmia benchmark (real ECG from PhysioNet).

THE standard real-world ECG classification benchmark. Used by Loihi
for heartbeat classification. 5 AAMI superclasses.

Uses the preprocessed Kaggle version (187-sample heartbeats, 5 classes).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import kagglehub
except ImportError:
    kagglehub = None

N_CHANNELS = 1   # single-lead ECG
N_CLASSES = 5    # N, S, V, F, Q (AAMI superclasses)
SEQ_LEN = 187    # samples per heartbeat

CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)',
               'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']


class MITBIHDataset(Dataset):
    """MIT-BIH Arrhythmia — real ECG heartbeat classification.

    Uses the preprocessed Kaggle dataset (187 samples per beat, 5 classes).
    Download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
    Place mitbih_train.csv and mitbih_test.csv in data_dir.
    """

    def __init__(self, data_dir="data/mitbih", train=True,
                 spike_threshold=0.3):
        self.spike_threshold = spike_threshold

        if train:
            path = os.path.join(data_dir, "mitbih_train.csv")
        else:
            path = os.path.join(data_dir, "mitbih_test.csv")

        if not os.path.exists(path):
            # Try downloading via kagglehub
            if kagglehub is not None:
                print("Downloading MIT-BIH dataset via kagglehub...")
                try:
                    dl_path = kagglehub.dataset_download(
                        "shayanfazeli/heartbeat")
                    # Copy files to data_dir
                    import shutil
                    os.makedirs(data_dir, exist_ok=True)
                    for f in ['mitbih_train.csv', 'mitbih_test.csv']:
                        src = os.path.join(dl_path, f)
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(data_dir, f))
                    print(f"Dataset saved to {data_dir}")
                except Exception as e:
                    print(f"kagglehub download failed: {e}")

        if not os.path.exists(path):
            # Generate from wfdb if available
            try:
                self._generate_from_wfdb(data_dir, train)
                assert os.path.exists(path), \
                    f"wfdb generation didn't create {path}"
            except ImportError:
                raise FileNotFoundError(
                    f"MIT-BIH dataset not found at {path}. "
                    "Download from https://www.kaggle.com/datasets/"
                    "shayanfazeli/heartbeat and place CSV files in "
                    f"{data_dir}/, or install kagglehub: "
                    "pip install kagglehub")

        # Load CSV: last column is label, rest is signal
        data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.signals = data[:, :-1]  # (N, 187)
        self.labels = data[:, -1].astype(np.int64)  # (N,)

        # Normalize signals
        self.signals = (self.signals - self.signals.mean()) / (
            self.signals.std() + 1e-8)

        print(f"MIT-BIH {'train' if train else 'test'}: "
              f"{len(self.signals)} samples, "
              f"{len(np.unique(self.labels))} classes")

    def _generate_from_wfdb(self, data_dir, train):
        """Generate CSV from PhysioNet using wfdb library."""
        import wfdb
        from scipy.signal import find_peaks

        os.makedirs(data_dir, exist_ok=True)

        # AAMI class mapping
        aami_map = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
            'A': 1, 'a': 1, 'J': 1, 'S': 1,  # Supraventricular
            'V': 2, 'E': 2,  # Ventricular
            'F': 3,  # Fusion
            '/': 4, 'f': 4, 'Q': 4,  # Unknown
        }

        records = [
            '100', '101', '102', '103', '104', '105', '106', '107',
            '108', '109', '111', '112', '113', '114', '115', '116',
            '117', '118', '119', '121', '122', '123', '124',
            '200', '201', '202', '203', '205', '207', '208', '209',
            '210', '212', '213', '214', '215', '217', '219',
            '220', '221', '222', '223', '228', '230', '231',
            '232', '233', '234',
        ]

        # DS1 for train, DS2 for test (standard split)
        ds1 = records[:23]   # first 23 records
        ds2 = records[23:]   # remaining records
        rec_set = ds1 if train else ds2

        all_beats = []
        for rec_id in rec_set:
            try:
                record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
                annotation = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
            except Exception:
                continue

            signal = record.p_signal[:, 0]  # Lead II
            for i, (samp, sym) in enumerate(
                    zip(annotation.sample, annotation.symbol)):
                if sym not in aami_map:
                    continue
                label = aami_map[sym]

                # Extract 187-sample window centered on R-peak
                start = max(0, samp - 93)
                end = start + 187
                if end > len(signal):
                    continue

                beat = signal[start:end].astype(np.float32)
                if len(beat) == 187:
                    row = np.append(beat, label)
                    all_beats.append(row)

        data = np.array(all_beats, dtype=np.float32)
        fname = "mitbih_train.csv" if train else "mitbih_test.csv"
        np.savetxt(os.path.join(data_dir, fname), data, delimiter=',')
        print(f"Generated {fname}: {len(data)} beats")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        ecg = self.signals[idx]  # (187,)

        # Delta spike encoding
        spikes = np.zeros((len(ecg), 1), dtype=np.float32)
        for t in range(1, len(ecg)):
            delta = abs(ecg[t] - ecg[t - 1])
            spikes[t, 0] = float(delta > self.spike_threshold)

        return torch.from_numpy(spikes), int(self.labels[idx])


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
