"""Quick status check for running training jobs."""
import subprocess, glob, os

# Check GPU status
r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                     '--format=csv,noheader'], capture_output=True, text=True)
print("=== GPUs ===")
print(r.stdout.strip())

# Check training logs
print("\n=== Training Logs ===")
for log in sorted(glob.glob('logs/n3/*.log')):
    size = os.path.getsize(log)
    if size == 0:
        continue
    with open(log) as f:
        lines = f.readlines()
    # Find last epoch summary or batch progress
    last_epoch = last_batch = ""
    for line in lines:
        if "Epoch" in line and "/" in line:
            last_epoch = line.strip()
        if "[" in line and "/" in line and "loss=" in line:
            last_batch = line.strip()
    name = os.path.basename(log)
    print(f"\n{name}:")
    if last_epoch:
        print(f"  Last epoch: {last_epoch}")
    if last_batch:
        print(f"  Last batch: {last_batch}")
    if not last_epoch and not last_batch:
        print(f"  {lines[-1].strip()}" if lines else "  (empty)")

# Check python processes
r2 = subprocess.run(['powershell', '-Command',
    'Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, @{N="RAM_MB";E={[math]::Round($_.WorkingSet64/1MB)}}, CPU | Format-Table -AutoSize'],
    capture_output=True, text=True)
print("\n=== Python Processes ===")
print(r2.stdout.strip())
