#!/bin/bash
# DigitalOcean GPU Droplet Setup — Catalyst Benchmarks
# Run this on the droplet after SSH in:
#   ssh root@<droplet-ip> 'bash -s' < do_setup.sh

set -e

echo "=== Catalyst Benchmarks — DO GPU Setup ==="

# System packages
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git unzip htop nvtop

# Verify GPU
nvidia-smi

# Create workspace
mkdir -p /root/catalyst
cd /root/catalyst

# Python venv
python3 -m venv venv
source venv/bin/activate

# PyTorch (CUDA 12.x)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SNN training dependencies
pip install --quiet numpy scipy h5py tonic snntorch matplotlib tensorboard

# Verify CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

echo ""
echo "=== Setup complete. Upload code and checkpoints: ==="
echo "  scp -r catalyst-benchmarks/ root@\$(hostname -I | awk '{print \$1}'):/root/catalyst/"
echo "  Then: cd /root/catalyst/catalyst-benchmarks && source ../venv/bin/activate"
echo ""
echo "=== Resume training examples: ==="
echo "  # SSC v5c (resume from checkpoint)"
echo "  python -u ssc/train.py --device cuda:0 --amp --epochs 200 --hidden1 1024 --hidden2 512 --resume checkpoints/ssc_v5c.pt --save checkpoints/ssc_v5c.pt 2>&1 | tee ssc_do.log &"
echo ""
echo "  # N4 SHD delays (resume)"
echo "  python -u shd/train_n4_delays_v2.py --hidden 1536 --max-delay 20 --neuron adlif --epochs 200 --device cuda:0 --amp --dropout 0.3 --label-smoothing 0.05 --delay-lr-scale 0.1 --resume checkpoints/shd_n4v2_h1536_d20.pt --save checkpoints/shd_n4v2_h1536_d20.pt 2>&1 | tee shd_n4_do.log &"
