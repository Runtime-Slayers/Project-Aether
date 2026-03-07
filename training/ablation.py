"""
Run simple ablation comparing `multiclass_train.py` (spectral),
`multiclass_train_raw.py` (raw IQ) and a lightweight baseline.

This script runs short experiments and writes `outputs/ablation_summary.csv`.
"""
import os
import csv
import time
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in __import__('sys').path:
    __import__('sys').path.insert(0, ROOT)

from training.multiclass_train import train as train_spectral
from training.multiclass_train_raw import train as train_raw

OUTPUT = 'outputs'
os.makedirs(OUTPUT, exist_ok=True)

def run_short(func, epochs=3, samples=200, batch=128):
    class Args:
        pass
    a = Args()
    a.epochs = epochs
    a.samples_per_class = samples
    a.batch_size = batch
    a.lr = 1e-3
    best_acc, best_path = func(a)
    return best_acc, best_path

def main():
    rows = []
    print('Running spectral short...')
    acc_s, p_s = run_short(train_spectral)
    rows.append(['spectral', acc_s, p_s])
    print('Running raw short...')
    acc_r, p_r = run_short(train_raw)
    rows.append(['raw', acc_r, p_r])

    out_csv = os.path.join(OUTPUT, 'ablation_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'best_val_acc', 'best_checkpoint'])
        for r in rows:
            w.writerow(r)

    print('Ablation complete. Summary:', out_csv)

if __name__ == '__main__':
    main()
