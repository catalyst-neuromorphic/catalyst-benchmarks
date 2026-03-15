"""Google Speech Commands (GSC) dataset loader for keyword spotting.

Speech2Spikes-style encoding: raw audio → mel spectrogram → log → delta modulation
→ binary spike trains {-1, 0, 1}. This produces native SNN inputs matching the
NeuroBench standard (https://doi.org/10.1145/3584954.3584995).

12-class task: yes, no, up, down, left, right, on, off, stop, go, unknown, silence.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile required: pip install soundfile")


N_CHANNELS = 40    # Default: 40 mel bands, delta-modulated to binary spikes
                    # Use n_channels=80 for higher frequency resolution
N_CLASSES = 12     # 10 keywords + unknown + silence (12-class KWS task)
N_CLASSES_35 = 35  # All 35 words (standard benchmark, used by SOTA papers)

KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
LABEL_MAP = {kw: i for i, kw in enumerate(KEYWORDS)}
LABEL_MAP['_unknown_'] = 10
LABEL_MAP['_silence_'] = 11

# All GSC v2 command words (for mapping non-keyword to unknown)
ALL_COMMANDS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
    'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
    'wow', 'yes', 'zero',
]

# 35-class label map: each word gets its own class (sorted alphabetically)
LABEL_MAP_35 = {cmd: i for i, cmd in enumerate(sorted(ALL_COMMANDS))}


def _mel_filterbank(sr, n_fft, n_mels, f_min=20, f_max=4000):
    """Simple mel filterbank matrix."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low_mel = hz_to_mel(f_min)
    high_mel = hz_to_mel(min(f_max, sr / 2))
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            filters[i, j] = (j - bin_points[i]) / max(1, bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            filters[i, j] = (bin_points[i + 2] - j) / max(1, bin_points[i + 2] - bin_points[i + 1])
    return filters


def _delta_modulate(signal, threshold=1.0):
    """Delta modulation: convert continuous signal to binary spikes.

    Matches Speech2Spikes algorithm (https://doi.org/10.1145/3584954.3584995).
    At each timestep, emits +1 if signal exceeds level+threshold, -1 if below
    level-threshold, 0 otherwise. Level tracks the signal via spike feedback.

    Args:
        signal: (T, channels) numpy array
        threshold: delta threshold (default 1.0)

    Returns:
        spikes: (T, channels) float32 array with values {-1, 0, 1}
    """
    T, C = signal.shape
    spikes = np.zeros((T, C), dtype=np.float32)
    levels = np.round(signal[0]).astype(np.float32)

    for t in range(T):
        up = (signal[t] - levels > threshold).astype(np.float32)
        down = (signal[t] - levels < -threshold).astype(np.float32)
        spikes[t] = up - down
        levels += spikes[t] * threshold

    return spikes


def audio_to_spikes_s2s(waveform_np, sr=16000, n_mels=40, hop_ms=10,
                         n_fft=512, threshold=1.0):
    """Speech2Spikes-style encoding: mel spectrogram → log → delta modulation.

    Produces binary spike trains {-1, 0, 1} that are native SNN inputs.

    Args:
        waveform_np: (samples,) numpy array
        sr: sample rate (default 16000)
        n_mels: number of mel bands (default 40)
        hop_ms: hop length in milliseconds (default 10 → 100 frames/sec)
        n_fft: FFT window size (default 512)
        threshold: delta modulation threshold (default 1.0)

    Returns:
        spikes: (T, n_mels) float32 numpy array with values {-1, 0, 1}
    """
    hop_length = int(sr * hop_ms / 1000)  # 160 for 10ms at 16kHz

    # STFT
    if len(waveform_np) < n_fft:
        waveform_np = np.pad(waveform_np, (0, n_fft - len(waveform_np)))

    num_frames = 1 + (len(waveform_np) - n_fft) // hop_length
    frames = np.zeros((num_frames, n_fft))
    window = np.hanning(n_fft)
    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        if end <= len(waveform_np):
            frames[i] = waveform_np[start:end] * window
        else:
            chunk = waveform_np[start:]
            frames[i, :len(chunk)] = chunk[:n_fft] * window[:len(chunk)]

    # Power spectrum → mel filterbank
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel_fb = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = spectrum @ mel_fb.T  # (T, n_mels)

    # Log (matching S2S pipeline — natural log, not log10)
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    # Delta modulation → binary spikes {-1, 0, 1}
    spikes = _delta_modulate(log_mel, threshold=threshold)

    return spikes


def audio_to_mel_int8(waveform_np, sr=16000, n_mels=40, hop_ms=10,
                       n_fft=512):
    """N3-compatible encoding: log mel + delta + delta-delta features.

    Bypasses delta modulation to preserve continuous features for N3's
    ANN INT8 MAC input layer. Returns (T, n_mels*3) features normalized
    to [-128, 127] range per channel.
    """
    hop_length = int(sr * hop_ms / 1000)

    if len(waveform_np) < n_fft:
        waveform_np = np.pad(waveform_np, (0, n_fft - len(waveform_np)))

    num_frames = 1 + (len(waveform_np) - n_fft) // hop_length
    frames = np.zeros((num_frames, n_fft))
    window = np.hanning(n_fft)
    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        if end <= len(waveform_np):
            frames[i] = waveform_np[start:end] * window
        else:
            chunk = waveform_np[start:]
            frames[i, :len(chunk)] = chunk[:n_fft] * window[:len(chunk)]

    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel_fb = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = spectrum @ mel_fb.T
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    # Delta and delta-delta (first and second temporal differences)
    delta = np.zeros_like(log_mel)
    delta[1:] = log_mel[1:] - log_mel[:-1]
    delta2 = np.zeros_like(delta)
    delta2[1:] = delta[1:] - delta[:-1]

    # Stack: (T, n_mels*3)
    features = np.concatenate([log_mel, delta, delta2], axis=1).astype(np.float32)

    # Per-channel normalize to [-128, 127] for INT8 MAC compatibility
    feat_max = np.abs(features).max(axis=0, keepdims=True)
    feat_max = np.maximum(feat_max, 1e-5)
    features = features * (127.0 / feat_max)

    return features


class GSCDataset(Dataset):
    """Google Speech Commands dataset with Speech2Spikes encoding.

    Loads wav files directly from extracted GSC v2 directory.
    Converts audio to binary spike trains via delta modulation on log mel
    spectrograms, matching the NeuroBench standard encoding.
    """

    def __init__(self, data_dir="data/gsc", split="training", n_channels=N_CHANNELS,
                 max_time_bins=100, balance_unknown=True, max_unknown_per_cmd=300,
                 threshold=1.0, encoding='s2s', full_35=False, cache=True):
        self.n_channels = n_channels
        self.max_time_bins = max_time_bins
        self.threshold = threshold
        self.encoding = encoding  # 's2s' (delta modulation) or 'n3' (INT8 mel features)
        self.full_35 = full_35
        self.label_map = LABEL_MAP_35 if full_35 else LABEL_MAP
        self.n_classes = N_CLASSES_35 if full_35 else N_CLASSES
        self.cache = cache
        self._feature_cache = {}  # idx -> features tensor (populated on first access)

        # Find the extracted directory
        base = os.path.join(data_dir, "SpeechCommands", "speech_commands_v0.02")
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"GSC data not found at {base}. "
                "Download from http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
            )

        # Read validation/test lists
        val_list = set()
        test_list = set()
        val_path = os.path.join(base, "validation_list.txt")
        test_path = os.path.join(base, "testing_list.txt")
        if os.path.exists(val_path):
            with open(val_path) as f:
                val_list = {line.strip() for line in f}
        if os.path.exists(test_path):
            with open(test_path) as f:
                test_list = {line.strip() for line in f}

        # Build samples list with class balancing
        self.samples = []  # (filepath, label_idx)
        for cmd in ALL_COMMANDS:
            cmd_dir = os.path.join(base, cmd)
            if not os.path.isdir(cmd_dir):
                continue
            if self.full_35:
                label = LABEL_MAP_35.get(cmd)
                if label is None:
                    continue  # skip commands not in the 35 (shouldn't happen)
            elif cmd in LABEL_MAP:
                label = LABEL_MAP[cmd]
            else:
                label = LABEL_MAP['_unknown_']

            cmd_samples = []
            for wav_path in glob.glob(os.path.join(cmd_dir, "*.wav")):
                # Determine split
                rel = cmd + "/" + os.path.basename(wav_path)
                if split == "training" and rel not in val_list and rel not in test_list:
                    cmd_samples.append((wav_path, label))
                elif split == "validation" and rel in val_list:
                    cmd_samples.append((wav_path, label))
                elif split == "testing" and rel in test_list:
                    cmd_samples.append((wav_path, label))

            # Undersample unknown class for training to prevent class imbalance
            # (only in 12-class mode where non-keywords are collapsed)
            if not full_35 and balance_unknown and label == LABEL_MAP['_unknown_'] and split == "training":
                import random as _rand
                if len(cmd_samples) > max_unknown_per_cmd:
                    _rand.shuffle(cmd_samples)
                    cmd_samples = cmd_samples[:max_unknown_per_cmd]

            self.samples.extend(cmd_samples)

        # Count class distribution
        from collections import Counter
        label_counts = Counter(label for _, label in self.samples)
        enc_str = "N3 INT8 mel+delta+delta2" if encoding == 'n3' else "S2S spike"
        n_ch = n_channels * 3 if encoding == 'n3' else n_channels
        class_str = "35-class" if full_35 else "12-class"
        print(f"GSC {split} ({class_str}): {len(self.samples)} samples, {n_ch} {enc_str} channels")
        if split == "training":
            lmap = LABEL_MAP_35 if full_35 else LABEL_MAP
            for label_idx in sorted(label_counts.keys()):
                name = [k for k, v in lmap.items() if v == label_idx][0]
                print(f"  class {label_idx:2d} ({name:>10s}): {label_counts[label_idx]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return cached features if available (skips all wav I/O + mel computation)
        if self.cache and idx in self._feature_cache:
            return self._feature_cache[idx]

        wav_path, label = self.samples[idx]

        # Load audio with soundfile
        waveform, sr = sf.read(wav_path, dtype='float32')
        if waveform.ndim > 1:
            waveform = waveform[:, 0]  # mono

        # Pad to 1 second if shorter
        target_len = sr  # 16000 samples = 1 second
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        elif len(waveform) > target_len:
            waveform = waveform[:target_len]

        if self.encoding == 'n3':
            features = audio_to_mel_int8(
                waveform, sr=sr, n_mels=self.n_channels)
            n_feat = self.n_channels * 3
        else:
            features = audio_to_spikes_s2s(
                waveform, sr=sr, n_mels=self.n_channels,
                threshold=self.threshold)
            n_feat = self.n_channels

        # Pad or truncate to fixed time bins
        T = features.shape[0]
        if T < self.max_time_bins:
            pad = np.zeros((self.max_time_bins - T, n_feat), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
        elif T > self.max_time_bins:
            features = features[:self.max_time_bins]

        result = (torch.from_numpy(features), label)

        # Cache in memory after first computation
        if self.cache:
            self._feature_cache[idx] = result

        return result


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
