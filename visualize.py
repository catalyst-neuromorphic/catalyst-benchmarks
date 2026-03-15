"""Generate benchmark comparison charts from results.json.

Creates publication-quality charts comparing Catalyst vs other neuromorphic processors.

Usage:
    python visualize.py                    # Generate all charts
    python visualize.py --output figures/  # Custom output directory
"""

import json
import os
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib required: pip install matplotlib")
    exit(1)


# Competitive results (from published papers)
COMPETITOR_RESULTS = {
    'SHD': {
        'Catalyst (LIF)': 90.7,
        'Loihi 1': 89.04,
        'Loihi 2': 90.9,
    },
    'SSC': {
        'Loihi 2': 69.8,
    },
    'N-MNIST': {
        'Loihi 1': 99.52,
    },
    'DVS Gesture': {
        'Loihi 1': 89.64,
        'Akida 2': 97.12,
    },
    'GSC KWS': {
        'Loihi 1': 88.5,
        'Akida 2': 92.83,
        'SpiNNaker 2': 91.12,
    },
}

BENCHMARK_MAP = {
    'shd': 'SHD',
    'ssc': 'SSC',
    'nmnist': 'N-MNIST',
    'dvs_gesture': 'DVS Gesture',
    'gsc_kws': 'GSC KWS',
}

COLORS = {
    'Catalyst (float)': '#2563eb',
    'Catalyst (quant)': '#60a5fa',
    'Catalyst (LIF)': '#93c5fd',
    'Loihi 1': '#dc2626',
    'Loihi 2': '#ef4444',
    'Akida 2': '#f59e0b',
    'SpiNNaker 2': '#10b981',
}


def load_results():
    """Load benchmark results from results.json."""
    results_path = os.path.join(os.path.dirname(__file__), 'results.json')
    if not os.path.exists(results_path):
        print(f"No results.json found at {results_path}")
        return {}
    with open(results_path) as f:
        return json.load(f)


def best_per_benchmark(results):
    """Return best (highest accuracy) result per benchmark."""
    best_float = {}
    best_quant = {}
    for r in results:
        bm_name = BENCHMARK_MAP.get(r['benchmark'], r['benchmark'])
        if bm_name not in BENCHMARK_MAP.values():
            continue
        acc = r.get('accuracy_float', 0)
        if acc and acc > best_float.get(bm_name, 0):
            best_float[bm_name] = acc
        qacc = r.get('accuracy_quant', 0)
        if qacc and qacc > best_quant.get(bm_name, 0):
            best_quant[bm_name] = qacc
    return best_float, best_quant


def plot_comparison_bars(results, output_dir):
    """Generate grouped bar chart comparing Catalyst vs competitors."""
    benchmarks_order = ['SHD', 'SSC', 'N-MNIST', 'DVS Gesture', 'GSC KWS']

    # Collect best result per benchmark
    catalyst_float, catalyst_quant = best_per_benchmark(results)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(benchmarks_order))
    width = 0.12
    offset = 0

    bars_plotted = {}

    # Plot Catalyst float
    vals = [catalyst_float.get(b, 0) for b in benchmarks_order]
    if any(v > 0 for v in vals):
        mask = [v > 0 for v in vals]
        positions = x[mask]
        heights = [v for v, m in zip(vals, mask) if m]
        bar = ax.bar(positions + offset * width, heights, width, label='Catalyst (float)',
                     color=COLORS['Catalyst (float)'], edgecolor='white', linewidth=0.5)
        bars_plotted['Catalyst (float)'] = bar
        offset += 1

    # Plot Catalyst quant
    vals = [catalyst_quant.get(b, 0) for b in benchmarks_order]
    if any(v > 0 for v in vals):
        mask = [v > 0 for v in vals]
        positions = x[mask]
        heights = [v for v, m in zip(vals, mask) if m]
        bar = ax.bar(positions + offset * width, heights, width, label='Catalyst (quant)',
                     color=COLORS['Catalyst (quant)'], edgecolor='white', linewidth=0.5)
        bars_plotted['Catalyst (quant)'] = bar
        offset += 1

    # Plot competitors
    all_competitors = set()
    for bm_results in COMPETITOR_RESULTS.values():
        all_competitors.update(bm_results.keys())

    for comp in sorted(all_competitors):
        if comp.startswith('Catalyst'):
            continue
        vals = [COMPETITOR_RESULTS.get(b, {}).get(comp, 0) for b in benchmarks_order]
        if any(v > 0 for v in vals):
            mask = [v > 0 for v in vals]
            positions = x[mask]
            heights = [v for v, m in zip(vals, mask) if m]
            color = COLORS.get(comp, '#6b7280')
            bar = ax.bar(positions + offset * width, heights, width, label=comp,
                         color=color, edgecolor='white', linewidth=0.5)
            bars_plotted[comp] = bar
            offset += 1

    # Center the bar groups
    center_offset = (offset - 1) * width / 2
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels(benchmarks_order, fontsize=12)

    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Neuromorphic Benchmark Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim(50, 102)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_quantization_scatter(results, output_dir):
    """Generate scatter plot of float vs quantized accuracy per benchmark."""
    fig, ax = plt.subplots(figsize=(8, 8))

    benchmarks = []
    float_accs = []
    quant_accs = []

    best_float, best_quant = best_per_benchmark(results)
    for bm_name in ['SHD', 'SSC', 'N-MNIST', 'DVS Gesture', 'GSC KWS']:
        if bm_name in best_float and bm_name in best_quant:
            benchmarks.append(bm_name)
            float_accs.append(best_float[bm_name])
            quant_accs.append(best_quant[bm_name])

    if not benchmarks:
        print("No quantized results to plot")
        return

    # Perfect line
    min_val = min(min(float_accs), min(quant_accs)) - 2
    max_val = max(max(float_accs), max(quant_accs)) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect quantization')

    # 1% loss line
    ax.plot([min_val, max_val], [min_val - 1, max_val - 1], 'r--', alpha=0.2, label='1% loss')

    colors = plt.cm.Set2(np.linspace(0, 1, len(benchmarks)))
    for i, (bm, fa, qa) in enumerate(zip(benchmarks, float_accs, quant_accs)):
        ax.scatter(fa, qa, s=150, c=[colors[i]], edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(bm, (fa, qa), textcoords="offset points", xytext=(8, 8),
                    fontsize=11, fontweight='bold')

    ax.set_xlabel('Float Accuracy (%)', fontsize=13)
    ax.set_ylabel('Quantized Accuracy (int16) (%)', fontsize=13)
    ax.set_title('Quantization Impact: Float vs Hardware Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'quantization_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_training_progress(results, output_dir):
    """Generate a timeline/progress chart showing benchmark status."""
    benchmarks = ['SHD', 'SSC', 'N-MNIST', 'DVS Gesture', 'GSC KWS']
    targets = {
        'SHD': 91.0, 'SSC': 72.0, 'N-MNIST': 98.0,
        'DVS Gesture': 92.0, 'GSC KWS': 91.0,
    }

    achieved, _ = best_per_benchmark(results)

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(benchmarks))

    # Target bars (lighter)
    target_vals = [targets.get(b, 0) for b in benchmarks]
    ax.barh(y_pos, target_vals, height=0.4, color='#e5e7eb', edgecolor='#9ca3af',
            linewidth=0.5, label='Target')

    # Achieved bars (on top)
    achieved_vals = [achieved.get(b, 0) for b in benchmarks]
    colors = ['#2563eb' if v > 0 else '#d1d5db' for v in achieved_vals]
    ax.barh(y_pos, achieved_vals, height=0.4, color=colors, edgecolor='white',
            linewidth=0.5, label='Achieved')

    # Labels
    for i, (bm, target, ach) in enumerate(zip(benchmarks, target_vals, achieved_vals)):
        if ach > 0:
            ax.text(ach + 0.5, i, f'{ach:.1f}%', va='center', fontsize=10, fontweight='bold',
                    color='#2563eb')
        ax.text(target + 0.5, i - 0.15, f'target: {target:.0f}%', va='center', fontsize=8,
                color='#6b7280')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(benchmarks, fontsize=12)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Benchmark Progress', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, 'benchmark_progress.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison charts")
    parser.add_argument("--output", default="figures", help="Output directory for charts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = load_results()

    print("Generating benchmark charts...")
    plot_comparison_bars(results, args.output)
    plot_quantization_scatter(results, args.output)
    plot_training_progress(results, args.output)
    print(f"\nAll charts saved to {args.output}/")


if __name__ == "__main__":
    main()
