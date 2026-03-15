"""Shared utilities for Catalyst neuromorphic benchmarks."""

from .neurons import (
    SurrogateSpikeFunction,
    surrogate_spike,
    LIFNeuron,
    AdaptiveLIFNeuron,
    ConvLIFLayer,
)
from .training import train_epoch, evaluate, run_training
from .deploy import quantize_weights, compute_hardware_params, build_hardware_network
from .augmentation import event_drop, time_stretch, spatial_jitter

__all__ = [
    "SurrogateSpikeFunction",
    "surrogate_spike",
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "ConvLIFLayer",
    "train_epoch",
    "evaluate",
    "run_training",
    "quantize_weights",
    "compute_hardware_params",
    "build_hardware_network",
    "event_drop",
    "time_stretch",
    "spatial_jitter",
]
