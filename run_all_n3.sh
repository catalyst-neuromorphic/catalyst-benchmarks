#!/bin/bash
# N3 Benchmark Campaign — Comprehensive orchestrator for all 4 tiers.
#
# Dual GPU strategy (nvidia-smi indexing):
#   CUDA_VISIBLE_DEVICES=0 → RTX 3080 (20GB):    Heavy models (SSC, sCIFAR-10, psMNIST)
#   CUDA_VISIBLE_DEVICES=1 → RTX 3080 Ti (12GB):  Standard models (SHD, GSC, NTiDigits)
#
# CUDA_VISIBLE_DEVICES trick: setting CUDA_VISIBLE_DEVICES=X makes that
# physical GPU appear as cuda:0 in the subprocess. So --device cuda:0 always.
# NOTE: PyTorch cuda:0 = 3080 Ti, cuda:1 = 3080. But CUDA_VISIBLE_DEVICES
# uses nvidia-smi indices which are REVERSED.
#
# Usage:
#   bash run_all_n3.sh tier1       # Headline benchmarks only
#   bash run_all_n3.sh tier2       # Feature showcase benchmarks
#   bash run_all_n3.sh tier3       # Breadth benchmarks
#   bash run_all_n3.sh tier4       # Ablation studies
#   bash run_all_n3.sh all         # Everything (10-12 days)
#   bash run_all_n3.sh smoke       # 5-epoch smoke test on all tiers
#   bash run_all_n3.sh --dry-run   # Print commands without executing

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p checkpoints logs/n3 data

TIER="${1:-all}"
DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

STATUS_FILE="checkpoints/n3_campaign_status.log"

# ── Utilities ──────────────────────────────────────────────────────────────

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$STATUS_FILE"
}

# Run a paired GPU job: one on GPU 0, one on GPU 1, wait for both
run_paired() {
    local name0="$1" cmd0="$2" name1="$3" cmd1="$4"
    local log0="logs/n3/${name0}.log" log1="logs/n3/${name1}.log"

    if $DRY_RUN; then
        echo "[DRY-RUN] GPU0: $cmd0"
        echo "[DRY-RUN] GPU1: $cmd1"
        return 0
    fi

    log "START: $name0 (GPU0) + $name1 (GPU1)"

    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 bash -c "$cmd0" > "$log0" 2>&1 &
    local pid0=$!
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 bash -c "$cmd1" > "$log1" 2>&1 &
    local pid1=$!

    local exit0=0 exit1=0
    wait $pid0 || exit0=$?
    wait $pid1 || exit1=$?

    [ $exit0 -eq 0 ] && log "DONE: $name0 (success)" || log "FAIL: $name0 (exit=$exit0)"
    [ $exit1 -eq 0 ] && log "DONE: $name1 (success)" || log "FAIL: $name1 (exit=$exit1)"
}

# Run a single GPU job
run_single() {
    local gpu="$1" name="$2" cmd="$3"
    local logfile="logs/n3/${name}.log"

    if $DRY_RUN; then
        echo "[DRY-RUN] GPU$gpu: $cmd"
        return 0
    fi

    log "START: $name (GPU$gpu)"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$logfile" 2>&1
    local exit_code=$?
    [ $exit_code -eq 0 ] && log "DONE: $name (success)" || log "FAIL: $name (exit=$exit_code)"
}

# ── Tier 1: Headline Benchmarks ───────────────────────────────────────────

tier1() {
    log "===== TIER 1: HEADLINE BENCHMARKS ====="

    # Wave 1: SSC N3 run 1 (GPU0=20GB) + SHD N3 sweep (GPU1=12GB)
    run_paired \
        "ssc_n3_run1" \
        "python ssc/train.py --neuron adlif --hidden1 1024 --hidden2 768 --recurrent2 --epochs 300 --dropout 0.3 --label-smoothing 0.05 --amp --save checkpoints/ssc_n3_rec2_h768.pt --device cuda:0" \
        "shd_n3_sweep" \
        "python sweep.py --benchmark shd_n3 --device cuda:0"

    # Wave 2: SSC N3 run 2 (GPU0=20GB) + GSC KWS sweep (GPU1=12GB)
    run_paired \
        "ssc_n3_run2" \
        "python ssc/train.py --neuron adlif --hidden1 1024 --hidden2 512 --recurrent2 --epochs 300 --dropout 0.4 --label-smoothing 0.05 --amp --save checkpoints/ssc_n3_rec2_h512_do40.pt --device cuda:0" \
        "gsc_sweep" \
        "python sweep.py --benchmark gsc_kws --device cuda:0"

    # Wave 3: N-MNIST push (GPU0=20GB) + DVS Gesture (GPU1=12GB)
    run_paired \
        "nmnist_push" \
        "python nmnist/train.py --neuron adlif --epochs 100 --time-bins 30 --event-drop --save checkpoints/nmnist_n3_t30.pt --device cuda:0" \
        "dvs_gesture_sweep" \
        "python sweep.py --benchmark dvs_gesture --device cuda:0"

    # Wave 4: NTiDigits (GPU1=12GB, small model)
    run_single 1 "ntidigits_n3" \
        "python ntidigits/train.py --neuron adlif --hidden 512 --epochs 200 --dropout 0.3 --save checkpoints/ntidigits_n3_h512.pt --device cuda:0"

    log "===== TIER 1 COMPLETE ====="
}

# ── Tier 2: Feature Showcase ──────────────────────────────────────────────

tier2() {
    log "===== TIER 2: FEATURE SHOWCASE ====="

    # Wave 5: psMNIST (GPU0=20GB, long timesteps) + ECG arrhythmia (GPU1=12GB, tiny)
    run_paired \
        "psmnist_n3" \
        "python psmnist/train.py --neuron adlif --hidden 512 --epochs 200 --save checkpoints/psmnist_n3.pt --device cuda:0" \
        "ecg_n3" \
        "python ecg_arrhythmia/train.py --neuron adlif --hidden 256 --epochs 200 --save checkpoints/ecg_n3.pt --device cuda:0"

    # Wave 6: sCIFAR-10 (GPU0=20GB, 1024 timesteps) + WISDM HAR (GPU1=12GB, tiny)
    run_paired \
        "scifar10_n3" \
        "python scifar10/train.py --neuron adlif --hidden 512 --epochs 100 --save checkpoints/scifar10_n3.pt --device cuda:0" \
        "wisdm_n3" \
        "python wisdm_har/train.py --neuron adlif --hidden 128 --epochs 200 --save checkpoints/wisdm_n3.pt --device cuda:0"

    log "===== TIER 2 COMPLETE ====="
}

# ── Tier 3: Breadth Benchmarks ────────────────────────────────────────────

tier3() {
    log "===== TIER 3: BREADTH BENCHMARKS ====="

    # Wave 7: CIFAR10-DVS (GPU0=20GB) + S-MNIST (GPU1=12GB)
    run_paired \
        "cifar10dvs_n3" \
        "python cifar10_dvs/train.py --neuron adlif --hidden1 512 --hidden2 256 --epochs 300 --save checkpoints/cifar10dvs_n3.pt --device cuda:0" \
        "smnist_n3" \
        "python smnist/train.py --neuron adlif --hidden 256 --epochs 200 --save checkpoints/smnist_n3.pt --device cuda:0"

    # Wave 8: N-Caltech101 (GPU0=20GB) + ASL-DVS (GPU1=12GB)
    run_paired \
        "ncaltech_n3" \
        "python ncaltech101/train.py --neuron adlif --hidden1 512 --hidden2 256 --epochs 300 --save checkpoints/ncaltech_n3.pt --device cuda:0" \
        "asl_dvs_n3" \
        "python asl_dvs/train.py --neuron adlif --hidden1 512 --hidden2 256 --epochs 200 --save checkpoints/asl_dvs_n3.pt --device cuda:0"

    # Wave 9: N-Cars (GPU1=12GB, binary, tiny)
    run_single 1 "ncars_n3" \
        "python ncars/train.py --neuron adlif --hidden 256 --epochs 200 --save checkpoints/ncars_n3.pt --device cuda:0"

    log "===== TIER 3 COMPLETE ====="
}

# ── Tier 4: Ablation Studies ──────────────────────────────────────────────

tier4() {
    log "===== TIER 4: ABLATION STUDIES ====="
    log "Note: Tier 4 requires completed Tier 1 checkpoints."

    # These are quick deployment/evaluation runs, not training
    # Run deploy scripts on best checkpoints from Tier 1

    if [ -f checkpoints/shd_n3_sweep_results.json ]; then
        log "Running SHD precision sweep..."
        run_single 0 "shd_precision_sweep" \
            "python shd/deploy.py --checkpoint checkpoints/shd_n3-2layer-baseline.pt --precision-sweep"
    fi

    if [ -f checkpoints/ssc_n3_rec2_h768.pt ]; then
        log "Running SSC FACTOR compression sweep..."
        run_single 0 "ssc_factor_sweep" \
            "python ssc/deploy.py --checkpoint checkpoints/ssc_n3_rec2_h768.pt --factor-sweep"
    fi

    log "===== TIER 4 COMPLETE ====="
}

# ── Smoke Test ────────────────────────────────────────────────────────────

smoke() {
    log "===== SMOKE TEST (5 epochs each) ====="

    # Quick validation: SSC on big GPU, SHD on small GPU
    run_paired \
        "smoke_ssc" \
        "python ssc/train.py --hidden1 256 --hidden2 128 --recurrent2 --epochs 5 --amp --save checkpoints/smoke_ssc.pt --device cuda:0" \
        "smoke_shd" \
        "python shd/train.py --layers 2 --hidden 256 --hidden2 128 --epochs 5 --save checkpoints/smoke_shd.pt --device cuda:0"

    run_single 1 "smoke_nmnist" \
        "python nmnist/train.py --epochs 5 --save checkpoints/smoke_nmnist.pt --device cuda:0"

    log "===== SMOKE TEST COMPLETE ====="
}

# ── Main Dispatch ─────────────────────────────────────────────────────────

log "=========================================="
log " Catalyst N3 Benchmark Campaign"
log " Tier: $TIER"
log "=========================================="

case "$TIER" in
    tier1)  tier1 ;;
    tier2)  tier2 ;;
    tier3)  tier3 ;;
    tier4)  tier4 ;;
    smoke)  smoke ;;
    all)
        tier1
        tier2
        tier3
        tier4
        ;;
    *)
        echo "Usage: $0 {tier1|tier2|tier3|tier4|all|smoke|--dry-run}"
        exit 1
        ;;
esac

log "=========================================="
log " Campaign complete!"
log " Results: results.json"
log " Logs: logs/n3/"
log " Status: $STATUS_FILE"
log "=========================================="
