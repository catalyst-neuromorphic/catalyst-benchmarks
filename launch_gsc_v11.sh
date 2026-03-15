#!/bin/bash
# GSC v11: 1024 hidden, less regularization (v10 was underfitting)
# Wait for v10 to finish, then launch v11

echo "=== Waiting for GSC v10 to finish ==="
while pgrep -f "gsc_s2s_v10" > /dev/null 2>&1; do
    sleep 60
done

echo "=== GSC v10 finished. Launching v11 ==="
echo "v11 config: 1024 hidden, dropout=0.2, activity_lambda=0.005, weight_decay=1e-4"

cd /c/Users/mrwab/catalyst-benchmarks
python gsc_kws/train.py \
    --hidden 1024 \
    --dropout 0.2 \
    --activity-lambda 0.005 \
    --weight-decay 1e-4 \
    --epochs 200 \
    --device cuda:1 \
    --amp \
    --save checkpoints/gsc_s2s_v11.pt \
    2>&1 | tee gsc_train11.log
