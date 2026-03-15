# N3 Benchmark Results

## Final Results

| Benchmark | Accuracy | SOTA | vs SOTA | Params | Architecture | Status |
|-----------|----------|------|---------|--------|--------------|--------|
| SHD | **91.0%** | 96.41% (SpikCommander) | -5.4% | 3.47M | 700->1536(rec adLIF)->35 | DONE |
| SSC | **76.4%** | 85.98% (SpikCommander) | -9.6% | 2.31M | 700->1024(rec adLIF)->512(adLIF)->35 | DONE |
| N-MNIST | **99.2%** | 99.67% (PLIF) | -0.47% | 691K | Conv SNN (LIF) | DONE |
| GSC-35 | 43.9% | 95.69% (d-cAdLIF) | -51.8% | ~1.5M | Delay+rec adLIF+BN (v4) | IN PROGRESS |
| N-Caltech101 | 41.1% | 84.35% (EV-VGCNN) | -43.3% | 1.47M | FC rec adLIF | PAUSED (resumable) |

## Checkpoints

### Best Models (in `checkpoints/`)
- `shd_91.0.pt` — SHD best, epoch 120/200
- `ssc_76.4.pt` — SSC best, epoch 0 (warm-started from prior epoch 33)
- `nmnist_99.2.pt` — N-MNIST best, epoch 68/80
- `gsc_43.9_inprogress.pt` — GSC best so far, epoch 37
- `ncaltech101_41.1_inprogress.pt` — N-Caltech101 best, epoch 40

### Resume Checkpoints (in `catalyst-benchmarks/checkpoints/`)
- `ncaltech101_adlif.pt.last.pt` — Epoch 39, full optimizer+scheduler state, best_acc=40.7%
- `gsc_v4_rec.pt.last.pt` — Epoch 39, full optimizer+scheduler state, best_acc=43.9%

### How to Resume N-Caltech101
```bash
cd C:/Users/mrwab/catalyst-benchmarks
python -u ncaltech101/train.py --epochs 200 --device cuda:0 --event-drop \
  --label-smoothing 0.05 --save checkpoints/ncaltech101_adlif.pt
```
Auto-resumes from `checkpoints/ncaltech101_adlif.pt.last.pt` (epoch 39).
NOTE: Crashed from RAM OOM twice. Run alone without other training processes.

### How to Resume GSC-35
```bash
cd C:/Users/mrwab/catalyst-benchmarks
python -u gsc_kws/train_v2.py --epochs 300 --batch-size 128 \
  --hidden1 512 --hidden2 512 --max-delay 30 --threshold 0.3 \
  --dropout 0.35 --lr 3e-4 --weight-decay 5e-4 --n-mels 40 \
  --encoding s2s --device cuda:1 --amp \
  --save checkpoints/gsc_v4_rec.pt --warmup-epochs 10 \
  --activity-lambda 0.005
```
Auto-resumes from `checkpoints/gsc_v4_rec.pt.last.pt`.
Currently running on cuda:1 (PID 267224).

## Training Scripts

| Benchmark | Script | Key Args |
|-----------|--------|----------|
| SHD | `shd/train.py` | `--hidden 1536 --epochs 200 --lr 1e-3` |
| SSC | `ssc/train.py` | `--hidden1 1024 --hidden2 768 --recurrent2 --epochs 70` |
| N-MNIST | `nmnist/train.py` | `--epochs 80` (conv architecture) |
| GSC-35 | `gsc_kws/train_v2.py` | See resume command above |
| N-Caltech101 | `ncaltech101/train.py` | See resume command above |

## Logs
Training logs in `logs/` subdirectory and also at `catalyst-benchmarks/logs/n3/`.

## Key Architecture Notes
- **Neuron**: AdaptiveLIFNeuron with learnable per-neuron threshold (common/neurons.py)
- **Surrogate gradient**: fast sigmoid, scale=25.0
- **Delay**: DelayedLinear with sigmoid-bounded learnable delays, init delay_raw=-3.0
- **Training**: Cosine annealing LR, warmup, label smoothing, activity regularization
- **Augmentation**: event_drop, time_stretch, spec_augment (for speech tasks)

## Known Issues
- GSC-35 test accuracy is very volatile (swings 20% between epochs)
- N-Caltech101 OOMs on RAM when running alongside other training — run alone
- N-Caltech101 loader had a critical bug (sequential split on class-sorted data) — FIXED in loader.py
- DVS-Gesture and CIFAR10-DVS data downloads are corrupted (0-byte files)
