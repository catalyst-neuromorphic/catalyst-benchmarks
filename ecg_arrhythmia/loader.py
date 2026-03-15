"""ECG Arrhythmia classification benchmark (synthetic).

Synthetic ECG-like waveforms with 5 arrhythmia classes.
Tests SNN's ability to classify biomedical time series.
Each class has distinct morphological features (QRS width, rhythm, amplitude).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


N_CHANNELS = 1   # single-lead ECG
N_CLASSES = 5    # Normal, Bradycardia, Tachycardia, PVC, AFib
SEQ_LEN = 300    # ~3 seconds at 100 Hz


CLASS_NAMES = ['Normal', 'Bradycardia', 'Tachycardia', 'PVC', 'AFib']


def _qrs_complex(t, center, width=0.04, amplitude=1.0):
    """Generate a QRS-like spike."""
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _p_wave(t, center, width=0.08, amplitude=0.15):
    """Generate a P wave."""
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _t_wave(t, center, width=0.12, amplitude=0.3):
    """Generate a T wave."""
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def generate_ecg(class_id, seq_len=SEQ_LEN, fs=100):
    """Generate synthetic ECG waveform for a given arrhythmia class."""
    t = np.linspace(0, seq_len / fs, seq_len)
    signal = np.zeros(seq_len)
    rng = np.random.default_rng()

    if class_id == 0:  # Normal sinus rhythm (~72 BPM)
        rr = 0.83 + rng.normal(0, 0.02)
        beats = np.arange(0, t[-1], rr)
        for beat in beats:
            signal += _p_wave(t, beat - 0.16)
            signal += _qrs_complex(t, beat)
            signal += _t_wave(t, beat + 0.22)

    elif class_id == 1:  # Bradycardia (~48 BPM)
        rr = 1.25 + rng.normal(0, 0.03)
        beats = np.arange(0, t[-1], rr)
        for beat in beats:
            signal += _p_wave(t, beat - 0.16)
            signal += _qrs_complex(t, beat, amplitude=0.9)
            signal += _t_wave(t, beat + 0.24)

    elif class_id == 2:  # Tachycardia (~120 BPM)
        rr = 0.5 + rng.normal(0, 0.02)
        beats = np.arange(0, t[-1], rr)
        for beat in beats:
            signal += _p_wave(t, beat - 0.12, amplitude=0.1)
            signal += _qrs_complex(t, beat, amplitude=0.8)
            signal += _t_wave(t, beat + 0.18, amplitude=0.2)

    elif class_id == 3:  # PVC (premature ventricular contraction)
        rr = 0.83
        beats = np.arange(0, t[-1], rr)
        for i, beat in enumerate(beats):
            if i % 3 == 2:  # Every 3rd beat is PVC
                signal += _qrs_complex(t, beat - 0.15, width=0.08,
                                        amplitude=1.5)
                signal += _t_wave(t, beat + 0.1, amplitude=-0.4)
            else:
                signal += _p_wave(t, beat - 0.16)
                signal += _qrs_complex(t, beat)
                signal += _t_wave(t, beat + 0.22)

    elif class_id == 4:  # Atrial fibrillation (irregular rhythm)
        pos = 0.2
        while pos < t[-1] - 0.3:
            rr = rng.uniform(0.4, 1.2)  # Highly irregular
            signal += _qrs_complex(t, pos, amplitude=0.7 + rng.normal(0, 0.15))
            signal += _t_wave(t, pos + 0.2, amplitude=0.2)
            # No P wave (fibrillating baseline)
            signal += 0.05 * np.sin(2 * np.pi * rng.uniform(4, 8) * t)
            pos += rr

    # Add noise
    signal += rng.normal(0, 0.03, seq_len)

    return signal.astype(np.float32)


class ECGArrhythmiaDataset(Dataset):
    """Synthetic ECG arrhythmia classification."""

    def __init__(self, n_samples=4000, train=True, spike_threshold=0.3):
        np.random.seed(42 if train else 123)
        n_per_class = n_samples // N_CLASSES
        self.spike_threshold = spike_threshold
        self.inputs = []
        self.labels = []

        for cls in range(N_CLASSES):
            for _ in range(n_per_class):
                ecg = generate_ecg(cls)
                self.inputs.append(ecg)
                self.labels.append(cls)

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

        # Normalize
        self.inputs = (self.inputs - self.inputs.mean()) / (
            self.inputs.std() + 1e-8)

        # Shuffle
        perm = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        ecg = self.inputs[idx]  # (SEQ_LEN,)
        # Delta encoding for spikes
        spikes = np.zeros((len(ecg), 1), dtype=np.float32)
        for t in range(1, len(ecg)):
            delta = abs(ecg[t] - ecg[t - 1])
            spikes[t, 0] = float(delta > self.spike_threshold)

        return torch.from_numpy(spikes), int(self.labels[idx])


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
