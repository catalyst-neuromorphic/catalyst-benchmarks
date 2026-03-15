"""N3-specific neuron models for benchmark training.

Extends common/neurons.py with N3 hardware features:
  - ANNINT8Linear/Conv2d: ANN INT8 MAC mode with QAT
  - GatedLIFNeuron: Multiplicative gating from modulatory input
  - WTALayer: Winner-Take-All (N3 two-pass WTA)

These simulate N3 hardware behavior during training. For deployment,
see common/deploy_n3.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.neurons import surrogate_spike, LIFNeuron


# ---------------------------------------------------------------------------
# INT8 Quantization-Aware Training (for N3 ANN mode)
# ---------------------------------------------------------------------------

class INT8Quantize(torch.autograd.Function):
    """Quantize to INT8 range [-128, 127] with STE (Straight-Through Estimator)."""

    @staticmethod
    def forward(ctx, x, scale):
        return (torch.clamp(torch.round(x / scale), -128, 127) * scale)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


int8_quantize = INT8Quantize.apply


class ANNINT8Linear(nn.Module):
    """Linear layer simulating N3 ANN INT8 MAC mode.

    Weights and activations are quantized to INT8 during forward pass
    using STE for gradient flow. No spiking threshold.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.w_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.a_scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        # Compute scales from running stats
        with torch.no_grad():
            w_absmax = self.linear.weight.abs().max().clamp(min=1e-5)
            self.w_scale.fill_(w_absmax / 127.0)
            a_absmax = x.abs().max().clamp(min=1e-5)
            self.a_scale.fill_(a_absmax / 127.0)

        w_q = int8_quantize(self.linear.weight, self.w_scale)
        x_q = int8_quantize(x, self.a_scale)
        out = F.linear(x_q, w_q, self.linear.bias)
        return out


class ANNINT8Conv2d(nn.Module):
    """Conv2d layer simulating N3 ANN INT8 MAC mode with QAT.

    Conv -> BN -> ReLU -> INT8 quantize. No spiking.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.a_scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        with torch.no_grad():
            a_absmax = out.abs().max().clamp(min=1e-5)
            self.a_scale.fill_(a_absmax / 127.0)
        return int8_quantize(out, self.a_scale)


# ---------------------------------------------------------------------------
# Gated LIF Neuron (for approximate computing simulation)
# ---------------------------------------------------------------------------

class GatedLIFNeuron(nn.Module):
    """LIF with multiplicative gating from a modulatory signal.

    Simulates N3 approximate computing: the gate can suppress
    far-from-threshold neurons to save power. During training, the gate
    is learned; during deployment, it maps to the N3 approximate skip logic.

    Dynamics:
        v = beta * v_prev + (1-beta) * I * gate
        spike = Heaviside(v - threshold)
        v = v * (1 - spike)
    """

    def __init__(self, size, beta_init=0.95, threshold=1.0):
        super().__init__()
        self.size = size
        self.threshold = threshold
        self.beta_raw = nn.Parameter(torch.full((size,), 2.944))  # sigmoid(2.944) ≈ 0.95
        self.gate_weight = nn.Parameter(torch.ones(size))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def forward(self, input_current, v_prev, gate=None):
        beta = self.beta
        if gate is not None:
            input_current = input_current * (torch.sigmoid(self.gate_weight) * gate)
        v = beta * v_prev + (1.0 - beta) * input_current
        spikes = surrogate_spike(v - self.threshold)
        v = v * (1.0 - spikes)
        return v, spikes


# ---------------------------------------------------------------------------
# TDM (Time-Division Multiplexing) LIF Neuron — N3 Shadow Banks
# ---------------------------------------------------------------------------

class TDMLIFNeuron(nn.Module):
    """LIF neuron with Time-Division Multiplexing (shadow banks).

    Simulates N3's TDM mode where one physical neuron handles multiple
    virtual time slots using independent membrane potential banks.
    At timestep t, bank (t % n_banks) is active. This allows processing
    long sequences (e.g. 784 timesteps in psMNIST) with n_banks× fewer
    physical neurons by time-sharing.

    Args:
        size: number of neurons per bank
        n_banks: number of TDM banks (physical neuron reuse factor)
        beta_init: initial membrane decay constant
        threshold: spike threshold
    """

    def __init__(self, size, n_banks=4, beta_init=0.95, threshold=1.0):
        super().__init__()
        self.size = size
        self.n_banks = n_banks
        self.threshold = threshold
        self.beta_raw = nn.Parameter(torch.full((size,), 2.944))  # sigmoid→0.95

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def forward(self, input_current, v_banks, t):
        """Forward pass for one timestep.

        Args:
            input_current: (batch, size) input
            v_banks: (batch, n_banks, size) membrane potentials for all banks
            t: current timestep index (used for bank selection)

        Returns:
            v_banks: updated (batch, n_banks, size)
            spikes: (batch, size) output spikes from active bank
        """
        bank_idx = t % self.n_banks
        beta = self.beta
        v = beta * v_banks[:, bank_idx] + (1.0 - beta) * input_current
        spikes = surrogate_spike(v - self.threshold)
        v = v * (1.0 - spikes)
        v_banks = v_banks.clone()
        v_banks[:, bank_idx] = v
        return v_banks, spikes


# ---------------------------------------------------------------------------
# Winner-Take-All layer (N3 two-pass WTA)
# ---------------------------------------------------------------------------

class WTALayer(nn.Module):
    """Winner-Take-All layer simulating N3 two-pass WTA.

    In N3, WTA is a hardware-level operation where only the top-k
    neurons in a group are allowed to fire per timestep. This encourages
    sparse, competitive representations.

    Args:
        size: number of neurons
        n_groups: number of WTA groups
        k: number of winners per group (default 1)
    """

    def __init__(self, size, n_groups=8, k=1, beta_init=0.95, threshold=1.0):
        super().__init__()
        self.size = size
        self.n_groups = n_groups
        self.k = k
        self.group_size = size // n_groups
        self.threshold = threshold
        self.lif = LIFNeuron(size, beta_init=beta_init, threshold=threshold,
                             learn_beta=True)

    def forward(self, input_current, v_prev):
        v, spikes = self.lif(input_current, v_prev)

        if self.training:
            # Soft WTA via group-wise top-k masking with STE
            batch = spikes.shape[0]
            spikes_grouped = spikes.view(batch, self.n_groups, self.group_size)
            # Get top-k values per group
            topk_vals, _ = spikes_grouped.topk(self.k, dim=2)
            threshold_per_group = topk_vals[:, :, -1:].detach()
            mask = (spikes_grouped >= threshold_per_group).float()
            spikes = (spikes_grouped * mask).view(batch, self.size)
        else:
            # Hard WTA at inference
            batch = spikes.shape[0]
            spikes_grouped = spikes.view(batch, self.n_groups, self.group_size)
            topk_vals, topk_idx = spikes_grouped.topk(self.k, dim=2)
            mask = torch.zeros_like(spikes_grouped)
            mask.scatter_(2, topk_idx, 1.0)
            spikes = (spikes_grouped * mask).view(batch, self.size)

        return v, spikes
