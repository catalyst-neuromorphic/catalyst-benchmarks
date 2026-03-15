"""Quantization and hardware deployment utilities.

Converts trained PyTorch models to Catalyst Neurocore SDK networks
for FPGA deployment. Handles weight quantization (float32 -> int16),
hardware parameter mapping, and quantized accuracy evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Neurocore SDK weight range (int16)
WEIGHT_MIN = -32768
WEIGHT_MAX = 32767


def quantize_weights(w_float, threshold_float, threshold_hw=1000):
    """Quantize float weight matrix to int16 for hardware deployment.

    Maps float weights so hardware dynamics match training dynamics:
        weight_hw = round(w_float * threshold_hw / threshold_float)
        clamped to [WEIGHT_MIN, WEIGHT_MAX]

    Args:
        w_float: (out, in) float32 weight matrix from nn.Linear
        threshold_float: threshold used in training (e.g. 1.0)
        threshold_hw: hardware threshold (default 1000)

    Returns:
        w_int: (in, out) int32 weight matrix (transposed for src->tgt convention)
    """
    scale = threshold_hw / threshold_float
    w_scaled = w_float * scale
    w_int = np.clip(np.round(w_scaled), WEIGHT_MIN, WEIGHT_MAX).astype(np.int32)
    # nn.Linear stores (out, in), SDK wants (src, tgt) = (in, out)
    return w_int.T


def compute_hardware_params(checkpoint, threshold_hw=1000):
    """Compute hardware neuron parameters from trained model.

    Maps membrane decay to CUBA neuron decay_v:
        decay_v = round(decay * 4096)  (12-bit fractional)

    For LIF: decay = beta (from lif1.beta_raw)
    For adLIF: decay = alpha (from lif1.alpha_raw)

    Returns:
        dict with hardware parameters for each layer
    """
    state = checkpoint['model_state_dict']
    neuron_type = _detect_neuron_type(state)
    params = {'neuron_type': neuron_type}

    if neuron_type == 'adlif':
        alpha_raw = state.get('lif1.alpha_raw')
        if alpha_raw is not None:
            alpha = torch.sigmoid(alpha_raw).cpu().numpy()
            params['hidden_decay_mean'] = float(alpha.mean())
            params['hidden_decay_v'] = int(round(alpha.mean() * 4096))

        rho_raw = state.get('lif1.rho_raw')
        if rho_raw is not None:
            rho = torch.sigmoid(rho_raw).cpu().numpy()
            params['hidden_rho_mean'] = float(rho.mean())
            params['hidden_rho_note'] = 'training-only'

        beta_a_raw = state.get('lif1.beta_a_raw')
        if beta_a_raw is not None:
            beta_a = F.softplus(beta_a_raw).cpu().numpy()
            params['hidden_beta_a_mean'] = float(beta_a.mean())
            params['hidden_beta_a_note'] = 'training-only'
    else:
        beta_raw = state.get('lif1.beta_raw')
        if beta_raw is not None:
            beta = torch.sigmoid(beta_raw).cpu().numpy()
            params['hidden_decay_mean'] = float(beta.mean())
            params['hidden_decay_v'] = int(round(beta.mean() * 4096))

    # Output layer is always standard LIF
    beta_out_raw = state.get('lif2.beta_raw')
    if beta_out_raw is not None:
        beta_out = torch.sigmoid(beta_out_raw).cpu().numpy()
        params['output_decay_mean'] = float(beta_out.mean())
        params['output_decay_v'] = int(round(beta_out.mean() * 4096))

    params['threshold_hw'] = threshold_hw
    return params


def build_hardware_network(checkpoint, threshold_hw=1000):
    """Build Neurocore SDK Network from a trained PyTorch checkpoint.

    Requires neurocore to be installed. Returns a Network ready for deploy().

    Args:
        checkpoint: loaded torch checkpoint dict
        threshold_hw: hardware spike threshold

    Returns:
        (net, hw_params) tuple
    """
    from neurocore import Network

    config = checkpoint.get('config', checkpoint.get('args', {}))
    threshold_float = config.get('threshold', 1.0)
    n_input = config.get('n_input', 700)
    n_hidden = config.get('hidden', config.get('n_hidden', 256))
    n_output = config.get('n_output', config.get('n_classes', 20))

    state = checkpoint['model_state_dict']
    w_fc1 = state['fc1.weight'].cpu().numpy()
    w_fc2 = state['fc2.weight'].cpu().numpy()
    w_rec = state.get('fc_rec.weight')
    if w_rec is not None:
        w_rec = w_rec.cpu().numpy()

    # Quantize
    wm_fc1 = quantize_weights(w_fc1, threshold_float, threshold_hw)
    wm_fc2 = quantize_weights(w_fc2, threshold_float, threshold_hw)
    wm_rec = quantize_weights(w_rec, threshold_float, threshold_hw) if w_rec is not None else None

    hw_params = compute_hardware_params(checkpoint, threshold_hw)
    decay_mean = hw_params.get('hidden_decay_mean', 0.95)
    decay_out = hw_params.get('output_decay_mean', 0.9)
    leak_hid = max(1, int(round((1 - decay_mean) * threshold_hw)))
    leak_out = max(1, int(round((1 - decay_out) * threshold_hw)))

    net = Network()
    inp = net.population(n_input,
                         params={'threshold': 65535, 'leak': 0, 'refrac': 0},
                         label="input")
    hid = net.population(n_hidden,
                         params={'threshold': threshold_hw, 'leak': leak_hid, 'refrac': 0},
                         label="hidden")
    out = net.population(n_output,
                         params={'threshold': threshold_hw, 'leak': leak_out, 'refrac': 0},
                         label="output")

    net.connect(inp, hid, weight_matrix=wm_fc1)
    net.connect(hid, out, weight_matrix=wm_fc2)
    if wm_rec is not None:
        net.connect(hid, hid, weight_matrix=wm_rec)

    # Report
    nonzero_fc1 = np.count_nonzero(wm_fc1)
    nonzero_fc2 = np.count_nonzero(wm_fc2)
    nonzero_rec = np.count_nonzero(wm_rec) if wm_rec is not None else 0
    total_conn = nonzero_fc1 + nonzero_fc2 + nonzero_rec
    print(f"Quantized network (threshold_hw={threshold_hw}):")
    print(f"  fc1: {wm_fc1.shape}, {nonzero_fc1:,} nonzero, "
          f"range [{wm_fc1.min()}, {wm_fc1.max()}]")
    print(f"  fc2: {wm_fc2.shape}, {nonzero_fc2:,} nonzero, "
          f"range [{wm_fc2.min()}, {wm_fc2.max()}]")
    if wm_rec is not None:
        print(f"  rec: {wm_rec.shape}, {nonzero_rec:,} nonzero, "
              f"range [{wm_rec.min()}, {wm_rec.max()}]")
    print(f"  Total connections: {total_conn:,}")

    return net, hw_params


def run_quantized_inference(model_class, checkpoint, test_loader, device='cpu',
                            threshold_hw=1000):
    """Run inference with quantized weights in PyTorch.

    Loads the model, replaces float weights with quantized versions
    (round-trip through int16), and runs normal forward pass.

    Args:
        model_class: SNN model class to instantiate
        checkpoint: loaded checkpoint dict
        test_loader: test DataLoader
        device: torch device
        threshold_hw: hardware threshold for quantization scale

    Returns:
        quantized accuracy (float)
    """
    config = checkpoint.get('config', checkpoint.get('args', {}))
    threshold_float = config.get('threshold', 1.0)
    scale = threshold_hw / threshold_float

    model = model_class(**_build_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Quantize and de-quantize weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and not any(k in name for k in
                    ('beta', 'alpha', 'rho', 'threshold_base')):
                q = torch.round(param * scale).clamp(WEIGHT_MIN, WEIGHT_MAX) / scale
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
    print(f"Quantized accuracy: {correct}/{total} = {acc*100:.1f}%")
    return acc


def _detect_neuron_type(state_dict):
    """Auto-detect neuron type from checkpoint state dict."""
    if 'lif1.alpha_raw' in state_dict:
        return 'adlif'
    return 'lif'


def _build_model_kwargs(config):
    """Build model constructor kwargs from config dict."""
    kwargs = {}
    field_map = {
        'n_input': 'n_input',
        'n_hidden': 'hidden',
        'n_output': 'n_output',
        'threshold': 'threshold',
        'dropout': 'dropout',
        'neuron_type': 'neuron_type',
        'alpha_init': 'alpha_init',
        'rho_init': 'rho_init',
        'beta_a_init': 'beta_a_init',
        'beta_hidden': 'beta_hidden',
        'beta_out': 'beta_out',
    }
    for key, alt_key in field_map.items():
        val = config.get(key, config.get(alt_key))
        if val is not None:
            kwargs[key] = val
    # Disable dropout at inference
    kwargs['dropout'] = 0.0
    return kwargs
