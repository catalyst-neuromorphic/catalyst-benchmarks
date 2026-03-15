"""Auto-queue N3 benchmark training on GPU1 after SHD finishes.

Monitors the SHD training log. When SHD completes (or GPU1 becomes free),
launches benchmarks sequentially: GSC N3 -> DVS WTA -> N-MNIST Deep -> psMNIST TDM -> ECG Gated.

Each job runs as a DETACHED_PROCESS so it survives shell exit.

Usage:
    python launch_n3_queue.py                 # Wait for SHD, then run queue
    python launch_n3_queue.py --skip-wait     # Start immediately (if GPU1 is free)
    python launch_n3_queue.py --start-from 2  # Skip GSC, start from DVS WTA
"""

import os
import sys
import time
import subprocess
import argparse

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# nvidia-smi GPU 1 = RTX 3080 Ti 12GB.
# CUDA_VISIBLE_DEVICES=1 makes it appear as cuda:0 inside PyTorch.
GPU_ENV = {"CUDA_VISIBLE_DEVICES": "1"}

JOBS = [
    {
        "name": "GSC N3 Hybrid (INT8 MAC)",
        "script": "gsc_kws/train.py",
        "args": [
            "--n3-hybrid", "--epochs", "200", "--batch-size", "128",
            "--lr", "1e-3", "--hidden", "512", "--n3-proj", "256",
            "--dropout", "0.3", "--amp", "--event-drop", "--time-stretch",
            "--save", "checkpoints/gsc_n3_hybrid.pt",
            "--device", "cuda:0",
        ],
        "log": "logs/n3/gsc_n3_hybrid.log",
    },
    # DVS, N-MNIST, psMNIST, ECG moved to launch_gpu0_queue.py (GPU0)
]


def gpu1_is_free():
    """Check if GPU1 (nvidia-smi index 1) has low memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True, text=True, timeout=10
        )
        mem_used = int(result.stdout.strip())
        return mem_used < 1000  # Less than 1GB = free
    except Exception:
        return False


def shd_is_done():
    """Check if SHD training log indicates completion."""
    log_path = os.path.join(BENCHMARKS_DIR, "logs/n3/shd_n3_tier1.log")
    if not os.path.exists(log_path):
        return True  # No log = never started or already done
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        # Check for completion markers
        if "Best test accuracy" in content or "Training complete" in content:
            return True
        # Check if last epoch matches total
        lines = content.strip().split('\n')
        for line in reversed(lines):
            if "Epoch" in line and "/" in line:
                parts = line.split("Epoch")[1].strip().split("/")
                current = int(parts[0].strip())
                total = int(parts[1].split("|")[0].strip())
                if current >= total:
                    return True
                break
    except Exception:
        pass
    return False


def run_job(job):
    """Launch a training job as a detached process. Wait for completion."""
    log_path = os.path.join(BENCHMARKS_DIR, job["log"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [PYTHON, os.path.join(BENCHMARKS_DIR, job["script"])] + job["args"]
    env = {**os.environ, **GPU_ENV}

    print(f"\n{'='*60}")
    print(f"LAUNCHING: {job['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    print(f"{'='*60}\n")

    log_file = open(log_path, 'w')
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        env=env, cwd=BENCHMARKS_DIR
    )

    # Poll until process finishes
    while proc.poll() is None:
        time.sleep(30)
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].strip()
                    if last:
                        print(f"  [{job['name']}] {last}")
        except Exception:
            pass

    log_file.close()
    rc = proc.returncode
    if rc == 0:
        print(f"\n  [DONE] {job['name']} completed successfully.")
    else:
        print(f"\n  [FAIL] {job['name']} exited with code {rc}.")
    return rc


def main():
    parser = argparse.ArgumentParser(description="N3 benchmark training queue")
    parser.add_argument("--skip-wait", action="store_true",
                        help="Don't wait for SHD to finish")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Job index to start from (0=GSC, 1=DVS, 2=NMNIST, 3=psMNIST, 4=ECG)")
    args = parser.parse_args()

    os.makedirs(os.path.join(BENCHMARKS_DIR, "logs/n3"), exist_ok=True)

    if not args.skip_wait:
        print("Waiting for SHD training to finish on GPU1...")
        while not shd_is_done() and not gpu1_is_free():
            time.sleep(60)
            # Check progress
            try:
                with open(os.path.join(BENCHMARKS_DIR, "logs/n3/shd_n3_tier1.log"), 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        if "Epoch" in line and "/" in line:
                            print(f"  SHD: {line.strip()}")
                            break
            except Exception:
                pass
        print("GPU1 is free! Starting N3 benchmark queue.\n")

    jobs = JOBS[args.start_from:]
    print(f"Queue: {len(jobs)} jobs to run")
    for i, job in enumerate(jobs):
        print(f"  {i+args.start_from}. {job['name']}")

    for i, job in enumerate(jobs):
        rc = run_job(job)
        if rc != 0:
            print(f"\nJob {job['name']} failed. Continuing with next job...")

    print(f"\n{'='*60}")
    print("ALL N3 BENCHMARK JOBS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
