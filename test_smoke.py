"""Smoke test: verify all benchmark scripts import and models forward-pass correctly.

Run: python test_smoke.py
"""

import sys
import torch
import numpy as np


def test_common():
    """Test common package imports and basic functionality."""
    print("Testing common package...")

    from common.neurons import LIFNeuron, AdaptiveLIFNeuron, ConvLIFLayer, surrogate_spike
    from common.training import train_epoch, evaluate
    from common.deploy import quantize_weights, compute_hardware_params
    from common.augmentation import event_drop, time_stretch, spatial_jitter

    # LIF forward pass
    lif = LIFNeuron(64)
    x = torch.randn(4, 64)
    v = torch.zeros(4, 64)
    v_new, spk = lif(x, v)
    assert v_new.shape == (4, 64)
    assert spk.shape == (4, 64)

    # adLIF forward pass
    adlif = AdaptiveLIFNeuron(64)
    a = torch.zeros(4, 64)
    v_new, spk, a_new = adlif(x, v, a, torch.zeros(4, 64))
    assert v_new.shape == (4, 64)
    assert a_new.shape == (4, 64)

    # Quantization
    w = np.random.randn(20, 512).astype(np.float32) * 0.1
    w_q = quantize_weights(w, 1.0, 1000)
    assert w_q.shape == (512, 20)
    assert w_q.dtype == np.int32

    # Augmentation
    batch = torch.rand(4, 100, 700)
    assert event_drop(batch).shape == batch.shape
    assert time_stretch(batch).shape == batch.shape

    print("  common: PASS")


def test_shd_model():
    """Test SHD model forward pass."""
    print("Testing SHD model...")
    from shd.train import SHDSNN

    # LIF model
    model = SHDSNN(n_input=700, n_hidden=64, n_output=20, neuron_type='lif', dropout=0.0)
    x = torch.rand(2, 50, 700)
    out = model(x)
    assert out.shape == (2, 20), f"SHD LIF output: {out.shape}"

    # adLIF model
    model = SHDSNN(n_input=700, n_hidden=64, n_output=20, neuron_type='adlif', dropout=0.0)
    out = model(x)
    assert out.shape == (2, 20), f"SHD adLIF output: {out.shape}"

    print("  SHD: PASS")


def test_nmnist_model():
    """Test N-MNIST model forward pass."""
    print("Testing N-MNIST model...")
    from nmnist.train import NMNISTSNN

    model = NMNISTSNN(n_input=2312, n_hidden1=64, n_hidden2=32, n_output=10,
                       neuron_type='adlif', dropout=0.0)
    x = torch.rand(2, 20, 2312)
    out = model(x)
    assert out.shape == (2, 10), f"N-MNIST output: {out.shape}"
    print("  N-MNIST: PASS")


def test_ssc_model():
    """Test SSC model forward pass."""
    print("Testing SSC model...")
    from ssc.train import SSCSNN

    model = SSCSNN(n_input=700, n_hidden1=64, n_hidden2=32, n_output=35,
                    neuron_type='adlif', dropout=0.0)
    x = torch.rand(2, 50, 700)
    out = model(x)
    assert out.shape == (2, 35), f"SSC output: {out.shape}"
    print("  SSC: PASS")


def test_dvs_model():
    """Test DVS Gesture model forward pass."""
    print("Testing DVS Gesture model...")
    from dvs_gesture.train import DVSGestureSNN

    model = DVSGestureSNN(n_input=2048, n_hidden1=64, n_hidden2=32, n_output=11,
                           neuron_type='adlif', dropout=0.0)
    x = torch.rand(2, 20, 2048)
    out = model(x)
    assert out.shape == (2, 11), f"DVS output: {out.shape}"
    print("  DVS Gesture: PASS")


def test_gsc_model():
    """Test GSC KWS model forward pass."""
    print("Testing GSC KWS model...")
    from gsc_kws.train import GSCSNN

    model = GSCSNN(n_input=80, n_hidden=64, n_output=12,
                    neuron_type='adlif', dropout=0.0)
    x = torch.rand(2, 50, 80)
    out = model(x)
    assert out.shape == (2, 12), f"GSC output: {out.shape}"
    print("  GSC KWS: PASS")


def test_gradient_flow():
    """Test that gradients flow through surrogate spike function."""
    print("Testing gradient flow...")
    from shd.train import SHDSNN

    model = SHDSNN(n_input=700, n_hidden=64, n_output=20, neuron_type='adlif', dropout=0.0)
    x = torch.rand(2, 50, 700)
    out = model(x)
    loss = out.sum()
    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters())
    assert grad_count > 0, "No gradients flowing!"
    print(f"  Gradient flow: {grad_count}/{total_params} params have gradients: PASS")


def main():
    print("=" * 50)
    print("Catalyst Benchmarks — Smoke Test")
    print("=" * 50)

    tests = [
        test_common,
        test_shd_model,
        test_nmnist_model,
        test_ssc_model,
        test_dvs_model,
        test_gsc_model,
        test_gradient_flow,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All smoke tests passed!")


if __name__ == "__main__":
    main()
