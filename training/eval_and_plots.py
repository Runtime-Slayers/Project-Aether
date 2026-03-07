"""
Evaluate saved checkpoints and produce publication-ready figures.

Usage:
    python training/eval_and_plots.py --checkpoint outputs/multiclass_raw_best_*.pt

If no checkpoint provided, the script will pick the latest matching `multiclass_raw_best_*.pt` or `multiclass_best_*.pt`.
"""
import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in __import__('sys').path:
    __import__('sys').path.insert(0, ROOT)

from training.multiclass_train_raw import RawCNN, make_dataset

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_latest_checkpoint():
    patterns = [os.path.join(OUTPUT_DIR, 'multiclass_raw_best_*.pt'), os.path.join(OUTPUT_DIR, 'multiclass_best_*.pt')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_model(checkpoint_path, device='cpu'):
    # Create model stub and load
    model = RawCNN(n_classes=3)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(model, device='cpu'):
    X_test, y_test = make_dataset(num_per_class=1000, seed=42, snr_db=10.0)
    X_test = (X_test - X_test.mean(axis=(0,2), keepdims=True)) / (X_test.std(axis=(0,2), keepdims=True) + 1e-9)
    import torch.nn.functional as F
    xb = torch.tensor(X_test).to(device)
    with torch.no_grad():
        logits = model(xb)
        preds = torch.exp(logits).argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return cm, report


def plot_confusion(cm, labels, outpath):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_report(report, outpath):
    import json
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ckpt = args.checkpoint if args.checkpoint else find_latest_checkpoint()
    if ckpt is None:
        print('No checkpoint found in outputs/. Train a model first.')
        return

    print('Loading checkpoint:', ckpt)
    model = load_model(ckpt, device=args.device)
    cm, report = evaluate(model, device=args.device)

    labels = ['BPSK', 'QPSK', 'QAM16']
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_eval.png')
    rpt_path = os.path.join(OUTPUT_DIR, 'classification_report_eval.json')
    plot_confusion(cm, labels, cm_path)
    save_report(report, rpt_path)
    print('Saved:', cm_path, rpt_path)


if __name__ == '__main__':
    main()
