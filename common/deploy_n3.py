"""N3-specific deployment utilities.

Extends common/deploy.py with N3 hardware features:
  - Multi-precision quantization (8/16/24-bit)
  - FACTOR low-rank synapse compression
  - Approximate computing simulation
  - Precision sweep (run at all bit-widths and report)

Usage:
    from common.deploy_n3 import run_precision_sweep, run_factor_compressed_inference
"""

import numpy as np
import torch
import torch.nn.functional as F


# Precision ranges for N3
PRECISION_RANGES = {
    8:  (-128, 127),
    16: (-32768, 32767),
    24: (-8388608, 8388607),
}


def quantize_weights_n3(w_float, precision=24, threshold_float=1.0, threshold_hw=1000):
    """N3 multi-precision weight quantization.

    Args:
        w_float: (out, in) float32 weight matrix
        precision: 8, 16, or 24 bits
        threshold_float: training threshold
        threshold_hw: hardware threshold

    Returns:
        w_int: (in, out) int32 weight matrix (transposed for SDK)
    """
    w_min, w_max = PRECISION_RANGES[precision]
    scale = threshold_hw / threshold_float
    w_scaled = w_float * scale
    w_int = np.clip(np.round(w_scaled), w_min, w_max).astype(np.int32)
    return w_int.T


def run_quantized_inference_n3(model_class, checkpoint, test_loader,
                                precision=24, device='cpu', threshold_hw=1000):
    """Run inference with N3 multi-precision quantized weights.

    Args:
        model_class: SNN model class
        checkpoint: loaded checkpoint dict
        test_loader: test DataLoader
        precision: 8, 16, or 24 bits
        device: torch device
        threshold_hw: hardware threshold

    Returns:
        accuracy (float)
    """
    from common.deploy import _build_model_kwargs

    config = checkpoint.get('config', checkpoint.get('args', {}))
    threshold_float = config.get('threshold', 1.0)
    scale = threshold_hw / threshold_float
    w_min, w_max = PRECISION_RANGES[precision]

    model = model_class(**_build_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Quantize and de-quantize at specified precision
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and not any(k in name for k in
                    ('beta', 'alpha', 'rho', 'threshold_base')):
                q = torch.round(param * scale).clamp(w_min, w_max) / scale
                param.copy_(q)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            correct += (output.argmax(1) == labels).sum().item()
            total += inputs.size(0)

    acc = correct / total
    print(f"N3 {precision}-bit quantized: {correct}/{total} = {acc*100:.2f}%")
    return acc


def run_precision_sweep(model_class, checkpoint, test_loader, device='cpu',
                        threshold_hw=1000):
    """Run inference at all N3 precision levels and report results.

    Returns:
        dict: {precision: accuracy} e.g. {8: 0.89, 16: 0.91, 24: 0.915}
    """
    results = {}
    for precision in [24, 16, 8]:
        acc = run_quantized_inference_n3(
            model_class, checkpoint, test_loader,
            precision=precision, device=device, threshold_hw=threshold_hw)
        results[precision] = acc
    return results


def run_factor_compressed_inference(model_class, checkpoint, test_loader,
                                     rank=32, device='cpu'):
    """Run inference with FACTOR low-rank compressed weights.

    Simulates N3 FACTOR synapse compression by decomposing weight matrices
    via SVD and truncating to the specified rank.

    Memory savings: rank/min(rows,cols) of original. E.g. rank=32 on a
    1024x512 matrix saves 32/512 = 93.75% of synapse memory.

    Args:
        model_class: SNN model class
        checkpoint: loaded checkpoint dict
        test_loader: test DataLoader
        rank: truncation rank (lower = more compression)
        device: torch device

    Returns:
        (accuracy, compression_ratio) tuple
    """
    from common.deploy import _build_model_kwargs

    config = checkpoint.get('config', checkpoint.get('args', {}))
    model = model_class(**_build_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    total_original = 0
    total_compressed = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2 and not any(k in name for k in
                    ('beta', 'alpha', 'rho', 'threshold_base')):
                rows, cols = param.shape
                original_params = rows * cols
                total_original += original_params

                effective_rank = min(rank, min(rows, cols))
                U, S, Vt = torch.linalg.svd(param, full_matrices=False)
                U_r = U[:, :effective_rank]
                S_r = S[:effective_rank]
                Vt_r = Vt[:effective_rank, :]
                W_approx = U_r @ torch.diag(S_r) @ Vt_r
                param.copy_(W_approx)

                compressed_params = rows * effective_rank + effective_rank + effective_rank * cols
                total_compressed += compressed_params

    compression_ratio = total_compressed / total_original if total_original > 0 else 1.0

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            correct += (output.argmax(1) == labels).sum().item()
            total += inputs.size(0)

    acc = correct / total
    savings = (1 - compression_ratio) * 100
    print(f"FACTOR rank-{rank}: {correct}/{total} = {acc*100:.2f}% "
          f"(synapse memory: {savings:.1f}% savings)")
    return acc, compression_ratio


def run_approximate_inference(model_class, checkpoint, test_loader,
                               quality=1.0, device='cpu'):
    """Simulate N3 approximate computing (skip-far-from-threshold).

    In N3, neurons whose membrane potential is far from threshold can
    skip computation to save power. This simulates that by zeroing out
    spikes from neurons whose pre-spike membrane is below quality*threshold.

    quality=1.0: full accuracy (no approximation)
    quality=0.5: skip neurons below 50% of threshold
    quality=0.25: aggressive approximation

    Args:
        model_class: SNN model class
        checkpoint: loaded checkpoint dict
        test_loader: test DataLoader
        quality: 0.0 to 1.0 (fraction of neurons to keep)
        device: torch device

    Returns:
        accuracy (float)
    """
    from common.deploy import _build_model_kwargs

    config = checkpoint.get('config', checkpoint.get('args', {}))
    model = model_class(**_build_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Hook to zero out spikes from far-from-threshold neurons
    hooks = []
    def make_approx_hook(quality_level):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                v, spikes = output[0], output[1]
                # Only keep spikes from neurons that were close to threshold
                threshold = getattr(module, 'threshold', 1.0)
                mask = (v.abs() >= quality_level * threshold).float()
                output_list = list(output)
                output_list[1] = spikes * mask
                return tuple(output_list)
        return hook

    for name, m in model.named_modules():
        if hasattr(m, 'threshold') and hasattr(m, 'beta'):
            hooks.append(m.register_forward_hook(make_approx_hook(quality)))

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            correct += (output.argmax(1) == labels).sum().item()
            total += inputs.size(0)

    for h in hooks:
        h.remove()

    acc = correct / total
    print(f"Approximate (quality={quality:.0%}): {correct}/{total} = {acc*100:.2f}%")
    return acc


def run_approximate_sweep(model_class, checkpoint, test_loader, device='cpu'):
    """Run approximate computing at multiple quality levels.

    Returns:
        dict: {quality: accuracy}
    """
    results = {}
    for quality in [1.0, 0.75, 0.5, 0.25]:
        acc = run_approximate_inference(
            model_class, checkpoint, test_loader,
            quality=quality, device=device)
        results[quality] = acc
    return results
