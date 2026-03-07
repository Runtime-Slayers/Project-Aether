"""
Multiclass training on raw I/Q channels using a simple 1D CNN.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in __import__('sys').path:
    __import__('sys').path.insert(0, ROOT)

from data_factory.generator import SignalGenerator, SignalConfig, ModulationType

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODS = [ModulationType.BPSK, ModulationType.QPSK, ModulationType.QAM16]

def make_dataset(num_per_class=500, seed=0, snr_db=20.0, augment=False):
    gen = SignalGenerator(seed=seed)
    X, y = [], []
    for i, mod in enumerate(MODS):
        for _ in range(num_per_class):
            cfg = SignalConfig(num_samples=256, sample_rate=1e6, snr_db=snr_db)
            s, _ = gen.generate(mod, config=cfg)
            if np.iscomplexobj(s):
                xi = np.stack([s.real, s.imag], axis=0).astype(np.float32)
            else:
                xi = np.stack([s.astype(np.float32), np.zeros_like(s, dtype=np.float32)], axis=0)
            X.append(xi)
            y.append(i)
    X = np.stack(X)
    y = np.array(y)
    perm = np.random.default_rng(seed+1).permutation(len(y))
    return X[perm], y[perm]


class RawCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256 * 8, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # seeds for reproducibility
    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    X_train, y_train = make_dataset(num_per_class=args.samples_per_class, seed=1, snr_db=20.0)
    X_val, y_val = make_dataset(num_per_class=int(args.samples_per_class * 0.2), seed=2, snr_db=15.0)

    mean = X_train.mean(axis=(0,2), keepdims=True)
    std = X_train.std(axis=(0,2), keepdims=True) + 1e-9
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val = torch.Generator()
    g_val.manual_seed(seed + 1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g_train)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, generator=g_val)

    model = RawCNN(n_classes=len(MODS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    best_path = None

    for epoch in range(args.epochs):
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(args, 'clip_grad_norm', 5.0))
            optimizer.step()
            tloss += loss.item() * xb.size(0)
        tloss /= len(train_loader.dataset)

        model.eval()
        all_preds, all_true = [], []
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds = torch.exp(logits).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_true.append(yb.cpu().numpy())
        vloss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        acc = accuracy_score(all_true, all_preds)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={tloss:.4f} val_loss={vloss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(OUTPUT_DIR, f'multiclass_raw_best_{int(time.time())}.pt')
            torch.save(model.state_dict(), best_path)

    print('Best val acc:', best_acc)
    return best_acc, best_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip-grad-norm', dest='clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--samples-per-class', type=int, default=800)
    args = parser.parse_args()
    train(args)
