#!/bin/bash
# Master training script — runs all benchmarks in optimal GPU order.
#
# Dual GPU strategy:
#   GPU 0 (RTX 3080 Ti, 12GB): Larger models (SSC 1024, SHD sweep)
#   GPU 1 (RTX 3080, 10GB):    Smaller models (N-MNIST, DVS Gesture, GSC)
#
# Usage:
#   bash run_all.sh           # Run everything (both GPUs)
#   bash run_all.sh shd       # Just SHD sweep
#   bash run_all.sh quick     # One config per benchmark (smoke test)

set -e
cd "$(dirname "$0")"

mkdir -p checkpoints data

BENCHMARK="${1:-all}"

echo "=========================================="
echo " Catalyst Benchmark Training"
echo " $(date)"
echo "=========================================="

if [ "$BENCHMARK" = "quick" ]; then
    echo "Quick mode: one config per benchmark"
    echo ""

    # Smoke test: 5 epochs each
    echo "--- SHD (5 epochs, smoke test) ---"
    python shd/train.py --neuron adlif --epochs 5 --device cuda:0 --save checkpoints/shd_quick.pt

    echo "--- N-MNIST (5 epochs, smoke test) ---"
    python nmnist/train.py --epochs 5 --device cuda:0 --save checkpoints/nmnist_quick.pt

    echo "--- SSC (5 epochs, smoke test) ---"
    python ssc/train.py --epochs 5 --device cuda:0 --save checkpoints/ssc_quick.pt

    echo "--- DVS Gesture (5 epochs, smoke test) ---"
    python dvs_gesture/train.py --epochs 5 --device cuda:0 --save checkpoints/dvs_quick.pt

    echo "--- GSC KWS (5 epochs, smoke test) ---"
    python gsc_kws/train.py --epochs 5 --device cuda:0 --save checkpoints/gsc_quick.pt

    echo ""
    echo "Quick test complete. Check checkpoints/ for results."
    exit 0
fi

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "shd" ]; then
    echo ""
    echo "=== Phase 2: SHD Improvement (GPU 0) ==="
    echo "Target: 91%+ (achieved 90.7%)"
    echo ""
    CUDA_VISIBLE_DEVICES=0 python sweep.py --benchmark shd --device cuda:0 &
    SHD_PID=$!
fi

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "nmnist" ]; then
    echo ""
    echo "=== Phase 3: N-MNIST (GPU 1) ==="
    echo "Target: 98%+"
    echo ""
    CUDA_VISIBLE_DEVICES=1 python sweep.py --benchmark nmnist --device cuda:0 &
    NMNIST_PID=$!
fi

# Wait for first batch
if [ "$BENCHMARK" = "all" ]; then
    echo "Waiting for SHD + N-MNIST to finish..."
    wait $SHD_PID $NMNIST_PID 2>/dev/null || true
    echo "SHD + N-MNIST complete."
fi

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "ssc" ]; then
    echo ""
    echo "=== Phase 4: SSC (GPU 0, larger model) ==="
    echo "Target: 72%+ (beat Loihi 2's 69.8%)"
    echo ""
    CUDA_VISIBLE_DEVICES=0 python sweep.py --benchmark ssc --device cuda:0 &
    SSC_PID=$!
fi

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "dvs" ]; then
    echo ""
    echo "=== Phase 5: DVS Gesture (GPU 1) ==="
    echo "Target: 92%+"
    echo ""
    CUDA_VISIBLE_DEVICES=1 python sweep.py --benchmark dvs_gesture --device cuda:0 &
    DVS_PID=$!
fi

if [ "$BENCHMARK" = "all" ]; then
    echo "Waiting for SSC + DVS Gesture to finish..."
    wait $SSC_PID $DVS_PID 2>/dev/null || true
    echo "SSC + DVS Gesture complete."
fi

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "gsc" ]; then
    echo ""
    echo "=== Phase 6: GSC KWS (GPU 0) ==="
    echo "Target: 91%+"
    echo ""
    CUDA_VISIBLE_DEVICES=0 python sweep.py --benchmark gsc_kws --device cuda:0
fi

echo ""
echo "=========================================="
echo " All benchmarks complete!"
echo " Results: results.json"
echo " Checkpoints: checkpoints/"
echo " $(date)"
echo "=========================================="
