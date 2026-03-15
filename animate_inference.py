"""Animate real SNN inference on a spoken digit (SHD benchmark).

Loads the trained 90.7% adLIF model, runs inference on a real test sample,
captures all spikes at every timestep, and creates a real-time animation.

All data is real — actual trained weights, actual audio-derived spikes,
actual neuron firing patterns during inference.

Usage:
    python animate_inference.py                     # Random test sample
    python animate_inference.py --sample 42         # Specific sample index
    python animate_inference.py --save demo.mp4     # Save as video
    python animate_inference.py --save demo.gif     # Save as GIF
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

from common.neurons import AdaptiveLIFNeuron, LIFNeuron, surrogate_spike
from shd.loader import SHDDataset, N_CHANNELS, N_CLASSES

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


# SHD class labels (digits 0-9 in German and English)
CLASS_LABELS = [
    'null', 'eins', 'zwei', 'drei', 'vier',
    'fünf', 'sechs', 'sieben', 'acht', 'neun',
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine',
]


def load_model(checkpoint_path):
    """Load trained SHD model from checkpoint."""
    from shd.train import SHDSNN

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt['config']

    model = SHDSNN(
        n_input=config.get('n_input', N_CHANNELS),
        n_hidden=config.get('hidden', 1024),
        n_output=config.get('n_output', N_CLASSES),
        threshold=config.get('threshold', 1.0),
        beta_out=config.get('beta_out', 0.9),
        dropout=0.0,  # no dropout at inference
        neuron_type=config.get('neuron_type', 'adlif'),
        alpha_init=config.get('alpha_init', 0.95),
        rho_init=config.get('rho_init', 0.85),
        beta_a_init=config.get('beta_a_init', 0.05),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, config, ckpt.get('test_acc', 0)


def run_inference_with_recording(model, input_tensor):
    """Run inference and record all internal state at every timestep.

    Returns dict with numpy arrays of all spike/voltage data.
    """
    model.eval()
    x = input_tensor.unsqueeze(0)  # add batch dim: (1, T, 700)
    T = x.shape[1]
    n_hidden = model.n_hidden
    n_output = model.n_output

    # Storage
    input_spikes = x[0].numpy()  # (T, 700)
    hidden_spikes = np.zeros((T, n_hidden), dtype=np.float32)
    hidden_voltage = np.zeros((T, n_hidden), dtype=np.float32)
    output_voltage = np.zeros((T, n_output), dtype=np.float32)
    output_cumsum = np.zeros((T, n_output), dtype=np.float32)

    with torch.no_grad():
        v1 = torch.zeros(1, n_hidden)
        v_out = torch.zeros(1, n_output)
        spk1 = torch.zeros(1, n_hidden)
        out_sum = torch.zeros(1, n_output)

        a1 = torch.zeros(1, n_hidden)

        for t in range(T):
            I1 = model.fc1(x[:, t]) + model.fc_rec(spk1)

            if model.neuron_type == 'adlif':
                v1, spk1, a1 = model.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = model.lif1(I1, v1)

            I_out = model.fc_out(spk1)
            beta_o = model.lif_out.beta
            v_out = beta_o * v_out + (1.0 - beta_o) * I_out
            out_sum = out_sum + v_out

            hidden_spikes[t] = spk1[0].numpy()
            hidden_voltage[t] = v1[0].numpy()
            output_voltage[t] = v_out[0].numpy()
            output_cumsum[t] = (out_sum[0] / (t + 1)).numpy()

    prediction = int(np.argmax(output_cumsum[-1]))

    return {
        'input_spikes': input_spikes,
        'hidden_spikes': hidden_spikes,
        'hidden_voltage': hidden_voltage,
        'output_voltage': output_voltage,
        'output_cumsum': output_cumsum,
        'prediction': prediction,
        'T': T,
    }


def create_animation(data, true_label, save_path=None, fps=30, speed=4):
    """Create animated visualization of SNN inference.

    Args:
        data: dict from run_inference_with_recording
        true_label: ground truth class index
        save_path: path to save (mp4/gif), or None for display
        fps: frames per second
        speed: timesteps per frame (higher = faster animation)
    """
    T = data['T']
    prediction = data['prediction']
    input_spikes = data['input_spikes']
    hidden_spikes = data['hidden_spikes']
    output_cumsum = data['output_cumsum']

    # Subsample neurons for visual clarity
    n_input_show = 200  # show 200 of 700 input channels
    n_hidden_show = 300  # show 300 of 1024 hidden neurons

    # Pick input channels that actually fire (most interesting visually)
    input_activity = input_spikes.sum(axis=0)
    active_input_idx = np.argsort(input_activity)[-n_input_show:]
    active_input_idx = np.sort(active_input_idx)

    # Pick hidden neurons that fire (skip dead neurons)
    hidden_activity = hidden_spikes.sum(axis=0)
    active_hidden_idx = np.argsort(hidden_activity)[-n_hidden_show:]
    active_hidden_idx = np.sort(active_hidden_idx)

    input_sub = input_spikes[:, active_input_idx]
    hidden_sub = hidden_spikes[:, active_hidden_idx]

    # Dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor='#080810')

    # Layout: input raster | hidden raster | output bars
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 0.8],
                          hspace=0.35, left=0.08, right=0.95,
                          top=0.92, bottom=0.06)

    ax_input = fig.add_subplot(gs[0])
    ax_hidden = fig.add_subplot(gs[1])
    ax_output = fig.add_subplot(gs[2])

    for ax in [ax_input, ax_hidden, ax_output]:
        ax.set_facecolor('#080810')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333340')
        ax.spines['bottom'].set_color('#333340')
        ax.tick_params(colors='#666680')

    # Title
    title = fig.suptitle('', fontsize=16, fontweight='bold',
                         color='#e0e0f0', y=0.97)

    # ── Input raster (accumulated scatter) ──
    ax_input.set_xlim(0, T)
    ax_input.set_ylim(-1, n_input_show)
    ax_input.set_ylabel('Input\nChannel', fontsize=10, color='#8888aa',
                        rotation=0, labelpad=50, va='center')
    ax_input.set_xticks([])

    # ── Hidden raster (accumulated scatter) ──
    ax_hidden.set_xlim(0, T)
    ax_hidden.set_ylim(-1, n_hidden_show)
    ax_hidden.set_ylabel('Hidden\nNeuron', fontsize=10, color='#8888aa',
                         rotation=0, labelpad=50, va='center')
    ax_hidden.set_xlabel('Timestep', fontsize=10, color='#8888aa')

    # ── Output bars ──
    bar_colors = ['#333350'] * N_CLASSES
    bars = ax_output.bar(range(N_CLASSES), [0] * N_CLASSES, color=bar_colors,
                         edgecolor='none', width=0.7)
    ax_output.set_xlim(-0.5, N_CLASSES - 0.5)
    ax_output.set_xticks(range(N_CLASSES))
    ax_output.set_xticklabels(CLASS_LABELS, fontsize=7, rotation=45,
                              ha='right', color='#8888aa')
    ax_output.set_ylabel('Class\nScore', fontsize=10, color='#8888aa',
                         rotation=0, labelpad=50, va='center')

    # Pre-compute all spike coordinates for scatter
    in_t_all, in_n_all = np.where(input_sub > 0)
    hid_t_all, hid_n_all = np.where(hidden_sub > 0)

    # Persistent scatter objects (start empty)
    in_scatter = ax_input.scatter([], [], s=6, c='#00ddff', alpha=0.9,
                                  marker='.', linewidths=0)
    hid_scatter = ax_hidden.scatter([], [], s=6, c='#33ff99', alpha=0.9,
                                    marker='.', linewidths=0)

    # "Now" line
    in_line = ax_input.axvline(0, color='#ffffff', alpha=0.15, linewidth=0.8)
    hid_line = ax_hidden.axvline(0, color='#ffffff', alpha=0.15, linewidth=0.8)

    n_frames = T // speed + 1

    def update(frame):
        t = min(frame * speed, T - 1)
        progress = t / T

        # Title with time and progress
        title.set_text(
            f'Catalyst SNN Inference  —  '
            f'Timestep {t}/{T}  '
            f'({t * 4:.0f}ms / {T * 4:.0f}ms)'
        )

        # Update input scatter (show all spikes up to current timestep)
        mask_in = in_t_all <= t
        if mask_in.any():
            in_scatter.set_offsets(
                np.column_stack([in_t_all[mask_in], in_n_all[mask_in]]))
        in_line.set_xdata([t, t])

        # Update hidden scatter
        mask_hid = hid_t_all <= t
        if mask_hid.any():
            hid_scatter.set_offsets(
                np.column_stack([hid_t_all[mask_hid], hid_n_all[mask_hid]]))
        hid_line.set_xdata([t, t])

        # Update output bars — use raw scores, not normalized
        scores = output_cumsum[t]
        # Shift so minimum is 0, keeps relative differences visible
        scores_shifted = scores - scores.min()

        current_pred = int(np.argmax(scores))
        for i, bar in enumerate(bars):
            bar.set_height(scores_shifted[i])
            if i == current_pred and progress > 0.2:
                bar.set_color('#4488ff')
                bar.set_alpha(1.0)
            else:
                bar.set_color('#333350')
                bar.set_alpha(0.6)

        ymax = max(0.01, scores_shifted.max() * 1.2)
        ax_output.set_ylim(0, ymax)

        # Final frame: show prediction result
        if t >= T - 1:
            correct = prediction == true_label
            result_color = '#44ff88' if correct else '#ff4444'
            result_text = f'Prediction: {CLASS_LABELS[prediction]}'
            if correct:
                result_text += '  ✓'

            title.set_text(
                f'Catalyst SNN Inference  —  {result_text}'
            )
            title.set_color(result_color)

            # Highlight winning bar
            bars[prediction].set_color(result_color)
            bars[prediction].set_alpha(1.0)

        return [in_scatter, hid_scatter, in_line, hid_line, title] + list(bars)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.gif':
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=120)
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
                anim.save(save_path, writer=writer, dpi=120)
            except Exception as e:
                print(f"FFmpeg save failed ({e}), trying Pillow GIF...")
                gif_path = save_path.rsplit('.', 1)[0] + '.gif'
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer, dpi=120)
                save_path = gif_path

        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Animate real SNN inference on spoken digit")
    parser.add_argument("--checkpoint",
                        default="checkpoints/shd_adlif_v7.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--sample", type=int, default=None,
                        help="Test sample index (random if not specified)")
    parser.add_argument("--save", default="figures/inference_animation.mp4",
                        help="Output path (mp4 or gif)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=int, default=3,
                        help="Timesteps per frame (higher = faster)")
    args = parser.parse_args()

    # Load model
    ckpt_path = os.path.join(os.path.dirname(__file__), args.checkpoint)
    print(f"Loading model from {ckpt_path}...")
    model, config, test_acc = load_model(ckpt_path)
    print(f"  Model: 700 -> {config.get('hidden', 1024)} (adLIF recurrent) -> 20")
    print(f"  Test accuracy: {test_acc * 100:.1f}%")

    # Load test data
    print("Loading SHD test set...")
    test_ds = SHDDataset(args.data_dir, "test", dt=4e-3)
    print(f"  {len(test_ds)} test samples, {test_ds.n_bins} timesteps each")

    # Pick sample
    if args.sample is not None:
        idx = args.sample
    else:
        # Pick a random sample that the model gets RIGHT (more interesting)
        import random
        random.seed(42)
        candidates = list(range(len(test_ds)))
        random.shuffle(candidates)
        idx = candidates[0]
        for c in candidates[:50]:
            x, y = test_ds[c]
            data = run_inference_with_recording(model, x)
            if data['prediction'] == y:
                idx = c
                break

    x, true_label = test_ds[idx]
    print(f"  Sample {idx}: true label = {CLASS_LABELS[true_label]} ({true_label})")

    # Run inference with full recording
    print("Running inference...")
    data = run_inference_with_recording(model, x)
    pred = data['prediction']
    correct = pred == true_label
    print(f"  Prediction: {CLASS_LABELS[pred]} ({pred}) "
          f"{'✓ CORRECT' if correct else '✗ WRONG'}")

    # Count spikes
    n_input_spikes = int(data['input_spikes'].sum())
    n_hidden_spikes = int(data['hidden_spikes'].sum())
    print(f"  Input spikes: {n_input_spikes:,}")
    print(f"  Hidden spikes: {n_hidden_spikes:,}")
    print(f"  Hidden firing rate: "
          f"{n_hidden_spikes / (data['T'] * model.n_hidden) * 100:.1f}%")

    # Create animation
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    print(f"Creating animation ({data['T'] // args.speed} frames at {args.fps} fps)...")
    create_animation(data, true_label, save_path=args.save,
                     fps=args.fps, speed=args.speed)


if __name__ == "__main__":
    main()
