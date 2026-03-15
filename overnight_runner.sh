#!/bin/bash
# Overnight autonomous benchmark runner
# Monitors GPU processes and launches new jobs as slots free up
# Started: 2026-03-06 03:20 UTC

cd "C:/Users/mrwab/catalyst-benchmarks"
LOG="C:/Users/mrwab/Desktop/Catalyst Neuromorphic/07_Customers/MergePlot/MDA_RFI/OPERATIONS_LOG.md"

log() {
    echo "[$(date '+%H:%M')] $1" | tee -a "$LOG"
}

# Phase 1: Wait for KAN synapse to finish (cuda:0), then launch SHD attention
log "Phase 1: Waiting for KAN synapse (PID 220952) to finish..."
while kill -0 220952 2>/dev/null; do sleep 60; done
log "KAN synapse FINISHED. Launching SHD attention on cuda:0..."
python -u shd/train_attention.py --epochs 100 --hidden 512 --n-heads 8 --device cuda:0 --amp --save checkpoints/shd_n4_attention.pt 2>&1 &
SHD_ATT_PID=$!
log "SHD attention launched (PID=$SHD_ATT_PID)"

# Phase 2: Wait for Sleep consolidation to finish (cuda:1), then launch SSC-KAN
log "Phase 2: Waiting for Sleep consolidation (PID 160744) to finish..."
while kill -0 160744 2>/dev/null; do sleep 60; done
log "Sleep consolidation FINISHED. Launching SSC-KAN on cuda:1..."
python -u ssc/train_kan.py --epochs 150 --hidden1 1024 --hidden2 512 --device cuda:1 --amp --save checkpoints/ssc_n4_kan.pt 2>&1 &
SSC_KAN_PID=$!
log "SSC-KAN launched (PID=$SSC_KAN_PID)"

# Phase 3: Wait for SHD attention to finish, then launch SSC recurrent2
log "Phase 3: Waiting for SHD attention (PID=$SHD_ATT_PID) to finish..."
wait $SHD_ATT_PID
log "SHD attention FINISHED. Launching SSC recurrent2 on cuda:0..."
python -u ssc/train.py \
    --resume-weights n3_results/checkpoints/ssc_76.4.pt \
    --recurrent2 --hidden1 1024 --hidden2 768 \
    --epochs 150 --lr 5e-4 --warmup-epochs 15 \
    --device cuda:0 --amp \
    --save checkpoints/ssc_n4_rec2.pt 2>&1 &
SSC_REC2_PID=$!
log "SSC recurrent2 launched (PID=$SSC_REC2_PID)"

# Phase 4: Collect all results
log "Phase 4: Waiting for all remaining jobs..."
wait $SSC_KAN_PID 2>/dev/null
log "SSC-KAN done."
wait $SSC_REC2_PID 2>/dev/null
log "SSC recurrent2 done."

log "=== ALL OVERNIGHT BENCHMARKS COMPLETE ==="
log "Check results.json for full results summary"
