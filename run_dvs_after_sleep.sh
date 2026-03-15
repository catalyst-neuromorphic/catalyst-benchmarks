#!/bin/bash
# Wait for sleep consolidation to finish, then run DVS Gesture on cuda:1
cd "C:/Users/mrwab/catalyst-benchmarks"

echo "Waiting for sleep consolidation demo to finish..."
while pgrep -f "train_sleep.py" > /dev/null 2>&1; do
    sleep 30
done

echo "Sleep demo finished. Starting DVS128 Gesture benchmark on cuda:1..."
python -u dvs_gesture/train.py --conv --epochs 200 --device cuda:1 --amp --batch-size 32 2>&1
