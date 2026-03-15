#!/bin/bash
# Upload code + checkpoints to DigitalOcean GPU droplet
# Usage: bash do_upload.sh <droplet-ip>

IP=$1
if [ -z "$IP" ]; then
    echo "Usage: bash do_upload.sh <droplet-ip>"
    exit 1
fi

echo "=== Uploading to root@$IP ==="

# Create remote dir
ssh root@$IP "mkdir -p /root/catalyst/catalyst-benchmarks/checkpoints /root/catalyst/catalyst-benchmarks/data"

# Upload training code (small files only, skip data/checkpoints)
rsync -avz --progress \
    --exclude 'checkpoints/' \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '*.log' \
    --exclude '.git/' \
    --exclude 'figures/' \
    --exclude '*.pdf' \
    --exclude '*.png' \
    -e ssh \
    "$(dirname "$0")/" root@$IP:/root/catalyst/catalyst-benchmarks/

# Upload key checkpoints to resume from
echo "=== Uploading checkpoints ==="
for ckpt in \
    checkpoints/ssc_v5c.pt \
    checkpoints/shd_n4v2_h1536_d20.pt \
    checkpoints/shd_n4v2_h1024_d20_mt4.pt \
    checkpoints/gsc_v4.pt \
    checkpoints/nmnist_n1_lif.pt \
    checkpoints/gsc_n1_lif.pt \
    checkpoints/mitbih_n3_256.pt; do
    if [ -f "$ckpt" ]; then
        echo "  Uploading $ckpt..."
        scp "$ckpt" root@$IP:/root/catalyst/catalyst-benchmarks/$ckpt
    fi
done

echo ""
echo "=== Done. SSH in and start training: ==="
echo "  ssh root@$IP"
echo "  cd /root/catalyst/catalyst-benchmarks"
echo "  source ../venv/bin/activate"
echo "  # Run setup first if not done: bash do_setup.sh"
