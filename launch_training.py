"""Launch training as a fully detached Windows process.

Usage:
    python launch_training.py --script ssc/train.py --gpu 0 --log logs/n3/ssc.log -- [train args...]
"""
import subprocess
import sys
import os

def main():
    # Split args on '--'
    try:
        sep = sys.argv.index('--')
        launcher_args = sys.argv[1:sep]
        train_args = sys.argv[sep+1:]
    except ValueError:
        print("Usage: python launch_training.py --script X --gpu N --log Y -- [train args]")
        sys.exit(1)

    # Parse launcher args
    script = gpu = log = None
    i = 0
    while i < len(launcher_args):
        if launcher_args[i] == '--script':
            script = launcher_args[i+1]; i += 2
        elif launcher_args[i] == '--gpu':
            gpu = launcher_args[i+1]; i += 2
        elif launcher_args[i] == '--log':
            log = launcher_args[i+1]; i += 2
        else:
            i += 1

    if not script:
        print("--script required")
        sys.exit(1)

    os.makedirs(os.path.dirname(log) if log else '.', exist_ok=True)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    cmd = [sys.executable, '-u', script] + train_args

    log_fh = open(log, 'w') if log else None

    # DETACHED_PROCESS on Windows = survives parent exit
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_fh or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    print(f"Launched PID {proc.pid}: {' '.join(cmd)}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")
    if log:
        print(f"Log: {log}")

if __name__ == '__main__':
    main()
