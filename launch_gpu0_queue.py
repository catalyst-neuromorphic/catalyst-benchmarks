"""Sequential benchmark queue on GPU0 (RTX 3080 20GB).

Runs SSC first (resumed from 72.1%), then the smaller N3 benchmarks
sequentially. All on GPU0 via CUDA_VISIBLE_DEVICES=0.

Usage:
    python launch_gpu0_queue.py
    python launch_gpu0_queue.py --skip-ssc          # Skip SSC, start from DVS
    python launch_gpu0_queue.py --start-from 2      # Start from job index 2
"""

import os
import sys
import time
import subprocess
import argparse

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# nvidia-smi GPU 0 = RTX 3080 20GB. NEVER use GPU 1.
GPU_ENV = {"CUDA_VISIBLE_DEVICES": "0"}

JOBS = [
    {
        "name": "SSC fine-tune (from 72.1%)",
        "script": "ssc/train.py",
        "args": [
            "--epochs", "70", "--batch-size", "64", "--lr", "5e-4",
            "--hidden1", "1024", "--hidden2", "512", "--dropout", "0.3",
            "--amp", "--activity-lambda", "0", "--warmup-epochs", "0",
            "--label-smoothing", "0.0",
            "--resume-weights", "checkpoints/ssc_adlif_amp2.pt",
            "--save", "checkpoints/ssc_n3_v3.pt",
            "--device", "cuda:0",
        ],
        "log": "logs/n3/ssc_n3_v3.log",
    },
    {
        "name": "DVS Gesture WTA",
        "script": "dvs_gesture/train.py",
        "args": [
            "--wta", "--epochs", "200", "--batch-size", "32",
            "--lr", "1e-3", "--fc-hidden", "256",
            "--wta-groups", "8", "--wta-k", "2",
            "--dropout", "0.3", "--amp", "--event-drop",
            "--save", "checkpoints/dvs_n3_wta.pt",
            "--device", "cuda:0",
        ],
        "log": "logs/n3/dvs_n3_wta.log",
    },
    {
        "name": "N-MNIST Deep Conv",
        "script": "nmnist/train.py",
        "args": [
            "--n3-deep", "--epochs", "50", "--batch-size", "128",
            "--lr", "1e-3", "--fc-hidden", "256",
            "--dropout", "0.2", "--amp", "--event-drop",
            "--save", "checkpoints/nmnist_n3_deep.pt",
            "--device", "cuda:0",
        ],
        "log": "logs/n3/nmnist_n3_deep.log",
    },
    {
        "name": "psMNIST TDM",
        "script": "psmnist/train.py",
        "args": [
            "--tdm", "--tdm-banks", "4", "--epochs", "50",
            "--batch-size", "128", "--lr", "1e-3", "--hidden", "256",
            "--dropout", "0.2",
            "--save", "checkpoints/psmnist_n3_tdm.pt",
            "--device", "cuda:0",
        ],
        "log": "logs/n3/psmnist_n3_tdm.log",
    },
    {
        "name": "ECG Gated",
        "script": "ecg_arrhythmia/train.py",
        "args": [
            "--gated", "--epochs", "100", "--batch-size", "64",
            "--lr", "1e-3", "--hidden", "128", "--dropout", "0.2",
            "--save", "checkpoints/ecg_n3_gated.pt",
            "--device", "cuda:0",
        ],
        "log": "logs/n3/ecg_n3_gated.log",
    },
]


def run_job(job):
    """Run a training job, wait for completion."""
    log_path = os.path.join(BENCHMARKS_DIR, job["log"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [PYTHON, "-u", os.path.join(BENCHMARKS_DIR, job["script"])] + job["args"]
    env = {**os.environ, **GPU_ENV, "PYTHONUNBUFFERED": "1"}

    print(f"\n{'='*60}")
    print(f"LAUNCHING: {job['name']}")
    print(f"Log: {log_path}")
    print(f"{'='*60}\n", flush=True)

    log_file = open(log_path, 'w')
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        env=env, cwd=BENCHMARKS_DIR
    )

    while proc.poll() is None:
        time.sleep(60)
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if line and ("Epoch" in line or "acc=" in line):
                        print(f"  [{job['name']}] {line}", flush=True)
                        break
        except Exception:
            pass

    log_file.close()
    rc = proc.returncode
    status = "DONE" if rc == 0 else f"FAIL (exit {rc})"
    print(f"\n  [{status}] {job['name']}", flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ssc", action="store_true")
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.join(BENCHMARKS_DIR, "logs/n3"), exist_ok=True)

    start = args.start_from
    if args.skip_ssc and start == 0:
        start = 1

    jobs = JOBS[start:]
    print(f"GPU0 Queue: {len(jobs)} jobs")
    for i, job in enumerate(jobs):
        print(f"  {i+start}. {job['name']}")
    print(flush=True)

    for job in jobs:
        run_job(job)

    print(f"\n{'='*60}")
    print("ALL GPU0 JOBS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
