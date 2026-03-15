"""Data augmentation for spiking neural network training.

Provides event-based augmentation strategies for spike train data:
- event_drop: randomly drop time bins or channels
- time_stretch: stretch or compress temporal dimension
- spatial_jitter: add Gaussian noise to spatial coordinates (DVS)
"""

import random

import torch
import torch.nn.functional as F


def event_drop(spikes_batch, drop_time_prob=0.1, drop_neuron_prob=0.05):
    """Randomly drop entire time bins or channels for regularization.

    With 50% probability drops random time bins; otherwise drops random
    input channels. Operates on full batch for efficiency.
    Supports both (B, T, C) and (B, T, C, H, W) tensor shapes.
    Typically gives ~1% accuracy boost.

    Args:
        spikes_batch: (B, T, ...) tensor
        drop_time_prob: probability of dropping each time bin
        drop_neuron_prob: probability of dropping each channel

    Returns:
        augmented tensor with same shape
    """
    B, T = spikes_batch.shape[:2]
    if random.random() < 0.5:
        # Drop random time bins — mask shape (1, T, 1, ...) to broadcast
        mask_shape = [1, T] + [1] * (spikes_batch.ndim - 2)
        mask = (torch.rand(mask_shape, device=spikes_batch.device)
                > drop_time_prob).float()
        return spikes_batch * mask
    else:
        # Drop random channels — mask shape (1, 1, C, 1, ...) to broadcast
        C = spikes_batch.shape[2]
        mask_shape = [1, 1, C] + [1] * (spikes_batch.ndim - 3)
        mask = (torch.rand(mask_shape, device=spikes_batch.device)
                > drop_neuron_prob).float()
        return spikes_batch * mask


def time_stretch(spikes_batch, factor_range=(0.8, 1.2)):
    """Stretch or compress the temporal dimension of spike trains.

    Randomly selects a stretch factor and uses linear interpolation
    to resize the time axis. Preserves batch and channel dimensions.

    Args:
        spikes_batch: (B, T, C) tensor
        factor_range: (min, max) range for random stretch factor

    Returns:
        (B, T, C) tensor with stretched/compressed time axis
    """
    B, T, C = spikes_batch.shape
    factor = random.uniform(*factor_range)
    new_T = max(1, int(T * factor))

    # Interpolate: reshape to (B, C, T) for F.interpolate, then back
    x = spikes_batch.permute(0, 2, 1)  # (B, C, T)
    x = F.interpolate(x, size=new_T, mode='linear', align_corners=False)

    # Pad or crop back to original T
    if new_T < T:
        x = F.pad(x, (0, T - new_T))
    elif new_T > T:
        x = x[:, :, :T]

    return x.permute(0, 2, 1)  # (B, T, C)


def spec_augment(x, freq_mask_width=5, time_mask_width=10, n_freq_masks=2, n_time_masks=2):
    """SpecAugment: frequency and time masking for speech data.

    Applies random contiguous masks along frequency and time dimensions.
    Standard technique for speech recognition (Park et al. 2019).

    Args:
        x: (B, T, C) tensor (time bins × frequency channels)
        freq_mask_width: max width of each frequency mask
        time_mask_width: max width of each time mask
        n_freq_masks: number of frequency masks to apply
        n_time_masks: number of time masks to apply

    Returns:
        masked tensor with same shape
    """
    B, T, C = x.shape
    x = x.clone()

    for _ in range(n_freq_masks):
        f = random.randint(0, min(freq_mask_width, C - 1))
        f0 = random.randint(0, C - f)
        x[:, :, f0:f0 + f] = 0

    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_width, T - 1))
        t0 = random.randint(0, T - t)
        x[:, t0:t0 + t, :] = 0

    return x


def spatial_jitter(events_batch, sigma=1.0, spatial_dims=(34, 34)):
    """Add Gaussian noise to spatial coordinates of DVS-style flattened input.

    Unflatten the channel dimension into (polarity, H, W), apply random
    spatial shifts to the spatial coordinates, and re-flatten.

    Args:
        events_batch: (B, T, C) tensor where C = polarity * H * W
        sigma: standard deviation of Gaussian jitter in pixels
        spatial_dims: (H, W) spatial dimensions

    Returns:
        (B, T, C) tensor with jittered spatial positions
    """
    B, T, C = events_batch.shape
    H, W = spatial_dims
    polarity = C // (H * W)

    # Reshape to spatial
    x = events_batch.view(B, T, polarity, H, W)

    # Generate random shift offsets
    shift_h = int(round(random.gauss(0, sigma)))
    shift_w = int(round(random.gauss(0, sigma)))

    # Apply shift via roll (wraps around edges)
    if shift_h != 0:
        x = torch.roll(x, shifts=shift_h, dims=3)
    if shift_w != 0:
        x = torch.roll(x, shifts=shift_w, dims=4)

    return x.view(B, T, C)
