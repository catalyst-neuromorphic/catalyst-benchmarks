"""Spiking neuron models for surrogate gradient training.

Provides LIF, Adaptive LIF (SE discretization), and Conv+LIF layers.
All models use a configurable surrogate gradient for BPTT.

Hardware mapping (Catalyst CUBA neuron):
    decay_v = round(beta * 4096)   (12-bit fractional)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate gradient
# ---------------------------------------------------------------------------

class SurrogateSpikeFunction(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid backward (surrogate gradient).

    Forward:  spike = 1 if x >= 0 else 0
    Backward: grad = 1 / (1 + scale * |x|)^2

    The scale parameter controls gradient sharpness. Higher values give
    sharper gradients (closer to true Heaviside) but harder optimization.
    """

    scale = 25.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = grad_output / (SurrogateSpikeFunction.scale * torch.abs(x) + 1.0) ** 2
        return grad


surrogate_spike = SurrogateSpikeFunction.apply


# ---------------------------------------------------------------------------
# LIF neuron — multiplicative decay (maps to CUBA hardware neuron)
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire with multiplicative (exponential) decay.

    Dynamics per timestep:
        v = beta * v_prev + (1 - beta) * I   # exponential decay + scaled input
        spike = Heaviside(v - threshold)       # surrogate in backward
        v = v * (1 - spike)                    # hard reset

    Hardware mapping (CUBA neuron):
        decay_v = round(beta * 4096)   (12-bit fractional)
    """

    def __init__(self, size, beta_init=0.95, threshold=1.0, learn_beta=True):
        super().__init__()
        self.size = size
        self.threshold = threshold

        init_val = np.log(beta_init / (1.0 - beta_init))
        if learn_beta:
            self.beta_raw = nn.Parameter(torch.full((size,), init_val))
        else:
            self.register_buffer('beta_raw', torch.full((size,), init_val))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def forward(self, input_current, v_prev):
        beta = self.beta
        v = beta * v_prev + (1.0 - beta) * input_current
        spikes = surrogate_spike(v - self.threshold)
        v = v * (1.0 - spikes)
        return v, spikes


# ---------------------------------------------------------------------------
# Adaptive LIF neuron — Symplectic Euler discretization
# ---------------------------------------------------------------------------

class AdaptiveLIFNeuron(nn.Module):
    """Adaptive LIF with Symplectic Euler (SE) discretization.

    Key: adaptation is updated BEFORE threshold computation, so the neuron
    can anticipate its own spike — greatly improves temporal coding.

    Dynamics per timestep (SE order):
        a = rho * a_prev + spike_prev          # 1. adaptation update FIRST
        theta = threshold_base + beta_a * a    # 2. adaptive threshold
        v = alpha * v_prev + (1-alpha) * I     # 3. membrane update
        spike = Heaviside(v - theta)            # 4. spike decision
        v = v * (1 - spike)                     # 5. hard reset

    Hardware note: adaptation is training-only. Only alpha (membrane decay)
    deploys to CUBA hardware as decay_v = round(alpha * 4096).
    """

    def __init__(self, size, alpha_init=0.95, rho_init=0.85, beta_a_init=0.05,
                 threshold=1.0):
        super().__init__()
        self.size = size
        self.threshold_base = nn.Parameter(torch.full((size,), threshold))

        init_alpha = np.log(alpha_init / (1.0 - alpha_init))
        self.alpha_raw = nn.Parameter(torch.full((size,), init_alpha))

        init_rho = np.log(rho_init / (1.0 - rho_init))
        self.rho_raw = nn.Parameter(torch.full((size,), init_rho))

        init_beta_a = np.log(np.exp(beta_a_init) - 1.0)
        self.beta_a_raw = nn.Parameter(torch.full((size,), init_beta_a))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, input_current, v_prev, a_prev, spike_prev):
        alpha = torch.sigmoid(self.alpha_raw)
        rho = torch.sigmoid(self.rho_raw)
        beta_a = F.softplus(self.beta_a_raw)

        # SE discretization: adaptation FIRST
        a_new = rho * a_prev + spike_prev
        theta = self.threshold_base + beta_a * a_new

        # Membrane dynamics
        v = alpha * v_prev + (1.0 - alpha) * input_current
        spikes = surrogate_spike(v - theta)
        v = v * (1.0 - spikes)

        return v, spikes, a_new


# ---------------------------------------------------------------------------
# Conv + LIF layer (training only — deploys as flattened weight matrix)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Delayed Linear — learnable per-input synaptic delays
# ---------------------------------------------------------------------------

class DelayedLinear(nn.Module):
    """Linear layer with learnable per-input synaptic delays.

    Each input feature has a learnable delay d_i in [0, max_delay].
    Delays are applied to the full input sequence via differentiable
    linear interpolation, enabling gradient flow through delay parameters.

    Implements the core mechanism from:
        Hammouamri et al. (2024) "Co-learning Synaptic Delays, Weights
        and Adaptation in SNNs", Frontiers in Neuroscience.

    Hardware mapping (Catalyst N3):
        Maps directly to per-axon delay queue (4 KB SRAM per core).
        delay_ticks = round(d_i / dt)  where dt = simulation timestep.
    """

    def __init__(self, in_features, out_features, max_delay=30, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Per-input learnable delay, initialized uniformly across delay range
        # (sigmoid maps [-2,2] to ~[0.12,0.88] of max_delay — good spread)
        self.delay_raw = nn.Parameter(torch.linspace(-2, 2, in_features))

        nn.init.xavier_uniform_(self.weight, gain=0.5)

    @property
    def delays(self):
        """Constrained delays in [0, max_delay]."""
        return torch.sigmoid(self.delay_raw) * self.max_delay

    def apply_delays(self, x_seq):
        """Apply per-channel delays to an input sequence.

        Args:
            x_seq: (batch, T, in_features) input sequence

        Returns:
            (batch, T, in_features) delayed sequence
        """
        batch, T, C = x_seq.shape
        delays = self.delays  # (C,)
        max_d = self.max_delay

        # Zero-pad time dimension at start (silence before signal)
        x_pad = F.pad(x_seq, (0, 0, max_d, 0))  # (batch, T+max_d, C)

        # Floor/ceil delay indices for linear interpolation
        d_floor = delays.long().clamp(0, max_d)
        d_frac = (delays - d_floor.float()).unsqueeze(0).unsqueeze(0)  # (1, 1, C)

        # For each timestep t, fetch from padded at index (t + max_d - d_floor[c])
        t_idx = torch.arange(T, device=x_seq.device)  # (T,)
        idx_floor = (t_idx.unsqueeze(1) + max_d - d_floor.unsqueeze(0))  # (T, C)
        idx_ceil = (idx_floor - 1).clamp(min=0)

        # Expand for batch gather
        idx_f = idx_floor.unsqueeze(0).expand(batch, -1, -1)  # (batch, T, C)
        idx_c = idx_ceil.unsqueeze(0).expand(batch, -1, -1)

        x_f = torch.gather(x_pad, 1, idx_f)
        x_c = torch.gather(x_pad, 1, idx_c)

        return x_f + d_frac * (x_c - x_f)

    def forward(self, x):
        """Standard linear on (already delayed) input. x: (batch, in_features)."""
        return F.linear(x, self.weight, self.bias)


class ConvLIFLayer(nn.Module):
    """Conv2D followed by LIF spiking activation.

    For spatial feature extraction on DVS data. Trains in PyTorch with
    surrogate gradients; for hardware deployment, pre-compute conv output
    as a fixed feature extraction stage, then deploy only FC SNN layers.

    Input:  (batch, T, C_in, H, W)
    Output: (batch, T, C_out, H', W'), spikes at each timestep
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, beta_init=0.9, threshold=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.threshold = threshold

        init_val = np.log(beta_init / (1.0 - beta_init))
        self.beta_raw = nn.Parameter(torch.tensor(init_val))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def forward(self, x):
        """Process temporal input through conv + LIF.

        Args:
            x: (batch, T, C_in, H, W) input tensor

        Returns:
            spikes: (batch, T, C_out, H', W') output spike tensor
        """
        batch, T = x.shape[:2]
        spatial_shape = x.shape[2:]

        # Initialize membrane potential
        # Do a dry run to get output spatial dims
        with torch.no_grad():
            dummy = torch.zeros(1, *spatial_shape, device=x.device)
            out_shape = self.conv(dummy).shape[1:]  # (C_out, H', W')

        v = torch.zeros(batch, *out_shape, device=x.device)
        beta = self.beta
        all_spikes = []

        for t in range(T):
            I = self.bn(self.conv(x[:, t]))
            v = beta * v + (1.0 - beta) * I
            spikes = surrogate_spike(v - self.threshold)
            v = v * (1.0 - spikes)
            all_spikes.append(spikes)

        return torch.stack(all_spikes, dim=1)  # (batch, T, C_out, H', W')
