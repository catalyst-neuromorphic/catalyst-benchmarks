"""SNN Benchmark Leaderboard — community comparison across neuromorphic hardware.

Maintained by Catalyst Neuromorphic (https://catalyst-neuromorphic.com).
Submit results via the Discussions tab.
"""

import gradio as gr
import pandas as pd
import json

# ── Benchmark results database ──────────────────────────────────────────────
# Each entry: platform, benchmark, accuracy, neuron model, params, source, year
# Sources are published papers, official repos, or reproducible benchmarks.

RESULTS = [
    # ── SHD (Spiking Heidelberg Digits) ── 20 classes, 700ch cochlea
    {"Platform": "Catalyst N3", "Benchmark": "SHD", "Accuracy (%)": 91.0,
     "Neuron": "adLIF", "Params": "3.47M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N2", "Benchmark": "SHD", "Accuracy (%)": 84.5,
     "Neuron": "adLIF", "Params": "759K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N1", "Benchmark": "SHD", "Accuracy (%)": 90.6,
     "Neuron": "LIF", "Params": "1.79M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Intel Loihi 2", "Benchmark": "SHD", "Accuracy (%)": 90.9,
     "Neuron": "ALIF", "Params": "—", "Type": "ASIC",
     "Source": "Mészáros et al. 2025", "Year": 2025},
    {"Platform": "Intel Loihi 1", "Benchmark": "SHD", "Accuracy (%)": 89.0,
     "Neuron": "ALIF", "Params": "—", "Type": "ASIC",
     "Source": "Cramer et al. 2022", "Year": 2022},
    {"Platform": "Software (SRNN)", "Benchmark": "SHD", "Accuracy (%)": 90.4,
     "Neuron": "LIF+attn", "Params": "—", "Type": "GPU",
     "Source": "Yin et al. 2021", "Year": 2021},
    {"Platform": "Software (SuperSpike)", "Benchmark": "SHD", "Accuracy (%)": 84.4,
     "Neuron": "LIF", "Params": "—", "Type": "GPU",
     "Source": "Zenke & Vogels 2021", "Year": 2021},
    {"Platform": "Software (Cramer)", "Benchmark": "SHD", "Accuracy (%)": 83.2,
     "Neuron": "LIF", "Params": "—", "Type": "GPU",
     "Source": "Cramer et al. 2020", "Year": 2020},

    # ── SSC (Spiking Speech Commands) ── 35 classes, 700ch
    {"Platform": "Catalyst N3", "Benchmark": "SSC", "Accuracy (%)": 76.4,
     "Neuron": "adLIF", "Params": "2.31M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N2", "Benchmark": "SSC", "Accuracy (%)": 72.1,
     "Neuron": "adLIF", "Params": "2.31M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Intel Loihi 2", "Benchmark": "SSC", "Accuracy (%)": 69.8,
     "Neuron": "ALIF", "Params": "—", "Type": "ASIC",
     "Source": "Mészáros et al. 2025", "Year": 2025},
    {"Platform": "Software (Bittar)", "Benchmark": "SSC", "Accuracy (%)": 74.2,
     "Neuron": "adLIF", "Params": "—", "Type": "GPU",
     "Source": "Bittar & Garner 2022", "Year": 2022},

    # ── N-MNIST (Neuromorphic MNIST) ── 10 classes, 34x34 DVS
    {"Platform": "Catalyst N3", "Benchmark": "N-MNIST", "Accuracy (%)": 99.2,
     "Neuron": "LIF", "Params": "691K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N2", "Benchmark": "N-MNIST", "Accuracy (%)": 97.8,
     "Neuron": "adLIF", "Params": "466K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N1", "Benchmark": "N-MNIST", "Accuracy (%)": 99.2,
     "Neuron": "LIF", "Params": "466K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Intel Loihi 1", "Benchmark": "N-MNIST", "Accuracy (%)": 99.5,
     "Neuron": "LIF", "Params": "—", "Type": "ASIC",
     "Source": "Shrestha & Orchard 2018", "Year": 2018},
    {"Platform": "Software (SLAYER)", "Benchmark": "N-MNIST", "Accuracy (%)": 99.2,
     "Neuron": "SRM", "Params": "—", "Type": "GPU",
     "Source": "Shrestha & Orchard 2018", "Year": 2018},

    # ── GSC KWS (Google Speech Commands — Keyword Spotting) ── 12 classes
    {"Platform": "Catalyst N3", "Benchmark": "GSC KWS", "Accuracy (%)": 88.0,
     "Neuron": "adLIF", "Params": "291K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N2", "Benchmark": "GSC KWS", "Accuracy (%)": 88.0,
     "Neuron": "adLIF", "Params": "291K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "BrainChip Akida 2", "Benchmark": "GSC KWS", "Accuracy (%)": 92.8,
     "Neuron": "IF", "Params": "—", "Type": "ASIC",
     "Source": "BrainChip 2024", "Year": 2024},
    {"Platform": "SpiNNaker 2", "Benchmark": "GSC KWS", "Accuracy (%)": 91.1,
     "Neuron": "LIF", "Params": "—", "Type": "ASIC",
     "Source": "Yan et al. 2024", "Year": 2024},

    # ── GSC KWS — N1
    {"Platform": "Catalyst N1", "Benchmark": "GSC KWS", "Accuracy (%)": 86.4,
     "Neuron": "LIF", "Params": "291K", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},

    # ── DVS128 Gesture ── 11 classes, 128x128 DVS
    {"Platform": "Catalyst N3", "Benchmark": "DVS Gesture", "Accuracy (%)": 89.0,
     "Neuron": "adLIF", "Params": "~1.2M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N2", "Benchmark": "DVS Gesture", "Accuracy (%)": 81.4,
     "Neuron": "adLIF", "Params": "~1.2M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "Catalyst N1", "Benchmark": "DVS Gesture", "Accuracy (%)": 69.7,
     "Neuron": "LIF", "Params": "~1.2M", "Type": "FPGA",
     "Source": "catalyst-benchmarks", "Year": 2026},
    {"Platform": "BrainChip Akida 2", "Benchmark": "DVS Gesture", "Accuracy (%)": 97.1,
     "Neuron": "IF", "Params": "—", "Type": "ASIC",
     "Source": "BrainChip 2024", "Year": 2024},
    {"Platform": "Intel Loihi 1", "Benchmark": "DVS Gesture", "Accuracy (%)": 89.6,
     "Neuron": "LIF", "Params": "—", "Type": "ASIC",
     "Source": "Shrestha & Orchard 2018", "Year": 2018},
]

BENCHMARKS = {
    "SHD": {
        "name": "Spiking Heidelberg Digits (SHD)",
        "classes": 20,
        "input": "700-channel cochlea spikes",
        "description": "Spoken digits classification using spike-encoded audio from the Heidelberg auditory model. The standard benchmark for temporal SNN processing.",
        "paper": "Cramer et al., IEEE TNNLS, 2020",
    },
    "SSC": {
        "name": "Spiking Speech Commands (SSC)",
        "classes": 35,
        "input": "700-channel cochlea spikes",
        "description": "35-class spoken command recognition. Larger and harder than SHD — tests temporal processing at scale.",
        "paper": "Cramer et al., IEEE TNNLS, 2020",
    },
    "N-MNIST": {
        "name": "Neuromorphic MNIST (N-MNIST)",
        "classes": 10,
        "input": "34x34 DVS events (2 polarities)",
        "description": "Event-camera recording of MNIST digits via saccadic eye movements. Standard spatial SNN benchmark.",
        "paper": "Orchard et al., Frontiers in Neuroscience, 2015",
    },
    "GSC KWS": {
        "name": "Google Speech Commands — Keyword Spotting",
        "classes": 12,
        "input": "Mel spectrogram → spike encoding",
        "description": "12-class keyword spotting (yes, no, up, down, etc.). Tests audio classification for edge deployment.",
        "paper": "Warden, 2018 (dataset); NeuroBench Task",
    },
    "DVS Gesture": {
        "name": "DVS128 Gesture Recognition",
        "classes": 11,
        "input": "128x128 DVS events",
        "description": "11 hand/arm gestures recorded with a DVS128 event camera. Tests spatiotemporal feature extraction.",
        "paper": "Amir et al., CVPR, 2017",
    },
}


def build_overview_df():
    """Build the main comparison table: best result per platform per benchmark."""
    df = pd.DataFrame(RESULTS)

    # Pivot: rows=Platform, cols=Benchmark, values=Accuracy
    benchmarks_order = ["SHD", "SSC", "N-MNIST", "GSC KWS", "DVS Gesture"]
    platforms_order = [
        "Catalyst N3", "Catalyst N2", "Catalyst N1", "Intel Loihi 2", "Intel Loihi 1",
        "BrainChip Akida 2", "SpiNNaker 2",
        "Software (SRNN)", "Software (Bittar)",
        "Software (SuperSpike)", "Software (Cramer)",
        "Software (SLAYER)",
    ]

    # Best accuracy per platform per benchmark
    best = df.groupby(["Platform", "Benchmark"])["Accuracy (%)"].max().reset_index()
    pivot = best.pivot(index="Platform", columns="Benchmark", values="Accuracy (%)")
    pivot = pivot.reindex(columns=benchmarks_order)

    # Reorder rows
    existing = [p for p in platforms_order if p in pivot.index]
    extra = [p for p in pivot.index if p not in platforms_order]
    pivot = pivot.reindex(existing + extra)

    # Format: add % sign, replace NaN with —
    for col in pivot.columns:
        pivot[col] = pivot[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"Platform": "Platform / Chip"})
    return pivot


def build_benchmark_df(benchmark_key):
    """Build per-benchmark detail table sorted by accuracy."""
    df = pd.DataFrame(RESULTS)
    bdf = df[df["Benchmark"] == benchmark_key].copy()
    bdf = bdf.sort_values("Accuracy (%)", ascending=False)
    bdf["Accuracy (%)"] = bdf["Accuracy (%)"].apply(lambda x: f"{x:.1f}%")
    bdf = bdf[["Platform", "Accuracy (%)", "Neuron", "Params", "Type", "Source", "Year"]]
    return bdf.reset_index(drop=True)


def build_hardware_df():
    """Hardware-only comparison (exclude software baselines)."""
    df = pd.DataFrame(RESULTS)
    hw = df[df["Type"].isin(["FPGA", "ASIC"])].copy()

    benchmarks_order = ["SHD", "SSC", "N-MNIST", "GSC KWS", "DVS Gesture"]
    platforms_order = [
        "Catalyst N3", "Catalyst N2", "Catalyst N1", "Intel Loihi 2", "Intel Loihi 1",
        "BrainChip Akida 2", "SpiNNaker 2",
    ]

    best = hw.groupby(["Platform", "Benchmark"])["Accuracy (%)"].max().reset_index()
    pivot = best.pivot(index="Platform", columns="Benchmark", values="Accuracy (%)")
    pivot = pivot.reindex(columns=benchmarks_order)

    existing = [p for p in platforms_order if p in pivot.index]
    extra = [p for p in pivot.index if p not in platforms_order]
    pivot = pivot.reindex(existing + extra)

    for col in pivot.columns:
        pivot[col] = pivot[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"Platform": "Hardware"})
    return pivot


# ── Build the Gradio app ────────────────────────────────────────────────────

HEADER_MD = """
# SNN Benchmark Leaderboard

Community benchmark comparison for **spiking neural networks** across neuromorphic hardware platforms.

All results from published papers, official repositories, or reproducible open-source benchmarks.
[Papers With Code](https://paperswithcode.com) shut down in July 2025 — this leaderboard fills that gap for neuromorphic computing.

**Submit your results** → open a [Discussion](https://huggingface.co/spaces/Catalyst-Neuromorphic/snn-benchmark-leaderboard/discussions) with your platform, benchmark, accuracy, neuron model, and source paper/repo.

---
"""

FOOTER_MD = """
---

### Sources & Methodology

All results use **test set accuracy** from the original published papers or official benchmark repositories.
Hardware results (FPGA/ASIC) include quantization to the target platform's precision.
Software baselines are float32 GPU results for reference.

**Benchmark datasets:**
- **SHD / SSC**: Cramer et al., *The Heidelberg Spiking Data Sets*, IEEE TNNLS 2020 — [zenodo.org/record/4677068](https://zenodo.org/record/4677068)
- **N-MNIST**: Orchard et al., *Converting Static Image Datasets to Spiking Neuromorphic Datasets*, Frontiers 2015
- **GSC KWS**: Warden, *Speech Commands*, 2018 — [arxiv.org/abs/1804.03209](https://arxiv.org/abs/1804.03209)
- **DVS Gesture**: Amir et al., *A Low Power, Fully Event-Based Gesture Recognition System*, CVPR 2017

**Catalyst benchmarks:** Fully reproducible at [github.com/catalyst-neuromorphic/catalyst-benchmarks](https://github.com/catalyst-neuromorphic/catalyst-benchmarks)

*Last updated: March 2026. Maintained by [Catalyst Neuromorphic](https://catalyst-neuromorphic.com).*
"""

with gr.Blocks(
    title="SNN Benchmark Leaderboard",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
) as demo:

    gr.Markdown(HEADER_MD)

    with gr.Tabs():
        # ── Tab 1: Hardware comparison ──
        with gr.TabItem("Hardware Comparison"):
            gr.Markdown("### Neuromorphic Hardware — Head to Head")
            gr.Markdown("Best published accuracy per platform. Hardware results only (FPGA + ASIC).")
            hw_df = build_hardware_df()
            gr.Dataframe(
                value=hw_df,
                interactive=False,
                wrap=True,
            )

        # ── Tab 2: All results ──
        with gr.TabItem("All Results"):
            gr.Markdown("### All Platforms (Hardware + Software Baselines)")
            overview_df = build_overview_df()
            gr.Dataframe(
                value=overview_df,
                interactive=False,
                wrap=True,
            )

        # ── Per-benchmark tabs ──
        for key, info in BENCHMARKS.items():
            with gr.TabItem(key):
                gr.Markdown(f"### {info['name']}")
                gr.Markdown(
                    f"**Classes:** {info['classes']} · "
                    f"**Input:** {info['input']}\n\n"
                    f"{info['description']}\n\n"
                    f"*Dataset paper: {info['paper']}*"
                )
                bdf = build_benchmark_df(key)
                gr.Dataframe(
                    value=bdf,
                    interactive=False,
                    wrap=True,
                )

        # ── Submit tab ──
        with gr.TabItem("Submit Results"):
            gr.Markdown("""
### Submit Your Results

We welcome benchmark submissions from any neuromorphic platform, SNN framework, or research group.

**To submit:**

1. Open a [Discussion](https://huggingface.co/spaces/Catalyst-Neuromorphic/snn-benchmark-leaderboard/discussions) with:
   - **Platform** (e.g., "Loihi 2", "Akida", "SpikingJelly on GPU")
   - **Benchmark** (SHD, SSC, N-MNIST, GSC KWS, DVS Gesture, or propose a new one)
   - **Test accuracy** (%)
   - **Neuron model** (LIF, adLIF, IF, Izhikevich, etc.)
   - **Parameter count** (if available)
   - **Hardware type** (ASIC, FPGA, GPU, CPU)
   - **Source** (paper DOI, arXiv link, or GitHub repo with reproducible training code)
   - **Year** of publication

2. We verify and add the result within 48 hours.

**Requirements:**
- Results must be reproducible (published paper or open-source code)
- Test set accuracy only (no validation set results)
- Hardware results should include quantization to target precision

**Suggest a new benchmark:**
We are happy to add new benchmark tasks. Open a Discussion with the dataset paper and proposed metrics.
""")

    gr.Markdown(FOOTER_MD)

demo.launch()
