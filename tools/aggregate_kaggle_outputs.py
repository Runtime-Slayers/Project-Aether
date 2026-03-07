"""Download kernel outputs from Kaggle for multiple kernel slugs and aggregate final metrics.

Usage:
    python tools/aggregate_kaggle_outputs.py --kernels user/kernel-slug1 user/kernel-slug2 --out summary.json

This script requires the `kaggle` CLI to be installed and configured.
It will call `kaggle kernels output <kernel>` for each kernel and look for
`summary.json` or `summary.csv` inside the downloaded outputs to extract
`val_acc` or best validation accuracy.
"""
import argparse
import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / 'kaggle_downloads'
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_kernel_output(kernel_slug: str, target_dir: Path):
    # calls: kaggle kernels output <kernel> -p <target_dir> --unzip
    cmd = ['kaggle', 'kernels', 'output', kernel_slug, '-p', str(target_dir), '--unzip']
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def find_metric_in_dir(d: Path):
    # Look for summary.json or summary.csv or outputs/summary.csv
    candidates = list(d.rglob('summary.json')) + list(d.rglob('summary.csv')) + list(d.rglob('outputs/summary.csv'))
    for c in candidates:
        try:
            if c.suffix == '.json':
                j = json.loads(c.read_text())
                # try to get last val_acc from history
                if isinstance(j, dict) and ('val_acc' in j):
                    return float(j['val_acc'])
                # if it's a metrics file with list
                if isinstance(j, dict) and ('final_metrics' in j):
                    fm = j['final_metrics']
                    if isinstance(fm, dict) and ('val_acc' in fm):
                        return float(fm['val_acc'])
            elif c.suffix == '.csv':
                import csv
                with open(c, newline='') as f:
                    rdr = csv.DictReader(f)
                    rows = list(rdr)
                    if len(rows) > 0 and 'val_acc' in rows[-1]:
                        return float(rows[-1]['val_acc'])
        except Exception:
            continue
    return None


def aggregate(kernels, out_path: Path):
    records = []
    for k in kernels:
        kd = DOWNLOAD_DIR / k.replace('/', '_')
        if kd.exists():
            shutil.rmtree(kd)
        kd.mkdir(parents=True)
        try:
            download_kernel_output(k, kd)
            val = find_metric_in_dir(kd)
            records.append({'kernel': k, 'val_acc': val})
        except subprocess.CalledProcessError as e:
            print('Failed to download outputs for', k, e)
            records.append({'kernel': k, 'val_acc': None})
    # aggregate
    vals = [r['val_acc'] for r in records if r['val_acc'] is not None]
    summary = {
        'n_runs': len(records),
        'n_successful': len(vals),
        'mean_val_acc': float(np.mean(vals)) if len(vals) > 0 else None,
        'std_val_acc': float(np.std(vals)) if len(vals) > 0 else None,
        'records': records
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print('Saved summary to', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kernels', nargs='+', required=True)
    p.add_argument('--out', type=str, default='kaggle_aggregate_summary.json')
    args = p.parse_args()
    aggregate(args.kernels, Path(args.out))
