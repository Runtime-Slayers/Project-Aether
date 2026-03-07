"""Submit multiple seeded runs to Kaggle by duplicating the kernel and pushing.

Usage:
    python tools/submit_kaggle_runs.py --seeds 0 1 2 --push

By default this only prepares per-seed kernel folders under `kaggle_kernel_runs/`.
If `--push` is passed, the script will attempt to call `kaggle kernels push -p <dir>` for each run.

Note: You must have the Kaggle CLI installed and `~/.kaggle/kaggle.json` configured.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_KERNEL = ROOT / 'kaggle_kernel_full'
OUT_DIR = ROOT / 'kaggle_kernel_runs'

RUNNER_NAME = 'run_on_kaggle.py'
META_NAME = 'kernel-metadata.json'

TEMPLATE_RUNNER = BASE_KERNEL / RUNNER_NAME
TEMPLATE_META = BASE_KERNEL / META_NAME


def prepare_run(seed: int, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    # copy entire base kernel
    shutil.copytree(BASE_KERNEL, run_dir, dirs_exist_ok=True)

    # modify runner to set seed
    runner_path = run_dir / RUNNER_NAME
    text = runner_path.read_text()
    # naive replace: look for cfg['seed'] assignment and replace if present
    if "cfg['seed']" in text:
        text = text.replace("cfg['seed'] = 42", f"cfg['seed'] = {seed}")
    else:
        # append a seed line after create_default_config() usage
        text = text.replace("cfg = create_default_config()", "cfg = create_default_config()\ncfg['seed'] = %d" % seed)
    runner_path.write_text(text)

    # update metadata title/id to include seed and timestamp to avoid collisions
    meta_path = run_dir / META_NAME
    meta = meta_path.read_text()
    stamp = int(time.time())
    meta = meta.replace('gnesai-full-training', f'gnesai-full-training-seed{seed}-{stamp}')
    meta = meta.replace('G-NeSAI Full Training Pipeline', f'G-NeSAI Full Training (seed={seed})')
    meta_path.write_text(meta)

    return run_dir


def push_run(run_dir: Path):
    print(f"Pushing kernel in {run_dir} ...")
    subprocess.run(['kaggle', 'kernels', 'push', '-p', str(run_dir)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--push', action='store_true', help='If set, push kernels to Kaggle')
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for s in args.seeds:
        run_dir = OUT_DIR / f'seed_{s}'
        prepare_run(s, run_dir)
        created.append(run_dir)
        print('Prepared', run_dir)

    print('\nPrepared all kernel runs in', OUT_DIR)
    if args.push:
        for rd in created:
            push_run(rd)


if __name__ == '__main__':
    main()
