"""
Kaggle kernel: Ablation sweeps comparing `multiclass_train.py` (spectral)
and `multiclass_train_raw.py` (raw IQ). Runs short/medium experiments across
SNR values and saves a CSV summary and plots to `outputs/`.
"""
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in __import__('sys').path:
    __import__('sys').path.insert(0, ROOT)

OUTPUT = 'outputs'
os.makedirs(OUTPUT, exist_ok=True)

# --- Embedded trainers (self-contained for Kaggle) ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def make_dataset_local(num_per_class=500, seed=0, snr_db=20.0):
    from kaggle_kernel_ablation_local_utils import SignalGeneratorLocal, SignalConfigLocal
    gen = SignalGeneratorLocal(seed=seed)
    MODS = ['bpsk', 'qpsk', '16qam']
    X, y = [], []
    for i, m in enumerate(MODS):
        for _ in range(num_per_class):
            cfg = SignalConfigLocal(num_samples=256, snr_db=snr_db)
            s = gen.generate(m, cfg)
            xi = np.stack([s.real, s.imag], axis=0).astype(np.float32)
            X.append(xi)
            y.append(i)
    X = np.stack(X)
    y = np.array(y)
    perm = np.random.default_rng(seed+1).permutation(len(y))
    return X[perm], y[perm]


class SimpleRawModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 512)
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


def train_local(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=256, lr=5e-4, device='cuda', seed=42, clip_grad_norm=5.0):
    device = device if torch.cuda.is_available() else 'cpu'
    # seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.NLLLoss()
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val = torch.Generator()
    g_val.manual_seed(seed + 1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_train)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=g_val)

    best_acc = 0.0
    best_path = None
    for epoch in range(epochs):
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
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

        print(f'Epoch {epoch+1}/{epochs} | train_loss={tloss:.4f} val_loss={vloss:.4f} val_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_path = f'best_{int(time.time())}.pt'
            torch.save(model.state_dict(), best_path)
        if best_acc >= 0.95:
            print('Reached target, early stopping')
            break

    return best_acc, best_path


def main():
    # import local signal utils implemented inline to avoid module path issues
    # Create a tiny local utils module file if not present
    utils_path = os.path.join(os.path.dirname(__file__), 'kaggle_kernel_ablation_local_utils.py')
    if not os.path.exists(utils_path):
        with open(utils_path, 'w') as f:
            f.write("""
import numpy as np

class SignalConfigLocal:
    def __init__(self, num_samples=256, sample_rate=1e6, snr_db=20.0, symbol_rate=50e3):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.symbol_rate = symbol_rate

class SignalGeneratorLocal:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate(self, mod_type, config):
        L = config.num_samples
        if mod_type == 'bpsk':
            const = np.array([1, -1])
        elif mod_type == 'qpsk':
            const = np.exp(1j * np.pi * np.array([1/4, 3/4, 5/4, 7/4]))
        else:
            real = np.array([-3, -1, 1, 3])
            imag = np.array([-3, -1, 1, 3])
            const = np.array([r + 1j*i for r in real for i in imag])
            const = const / np.sqrt(np.mean(np.abs(const)**2))
        samples_per_symbol = int(config.sample_rate / config.symbol_rate)
        num_symbols = L // samples_per_symbol + 1
        symbol_indices = self.rng.integers(0, len(const), num_symbols)
        symbols = const[symbol_indices]
        signal = np.repeat(symbols, samples_per_symbol)[:L]
        # add AWGN
        sig_pow = np.mean(np.abs(signal)**2)
        snr_lin = 10**(config.snr_db/10.0)
        noise_pow = sig_pow / snr_lin
        noise = (np.sqrt(noise_pow/2) * (self.rng.normal(size=L) + 1j*self.rng.normal(size=L)))
        signal = signal + noise
        return signal.astype(np.complex64)
""")

    results = []
    snr_values = [30, 20, 10]
    for model_name in ['raw']:
        for snr in snr_values:
            print(f'Running {model_name} @ SNR={snr}')
            X_train, y_train = make_dataset_local(num_per_class=800, seed=1, snr_db=snr)
            X_val, y_val = make_dataset_local(num_per_class=160, seed=2, snr_db=max(5, snr-5))
            model = SimpleRawModel(n_classes=3)
            best_acc, ckpt = train_local(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=256)
            results.append({'model': model_name, 'snr_db': snr, 'best_val_acc': float(best_acc), 'checkpoint': ckpt})
            with open(os.path.join(OUTPUT, 'ablation_progress.json'), 'w') as f:
                json.dump(results, f, indent=2)

    # Save summary CSV and plot
    import csv
    csv_path = os.path.join(OUTPUT, 'ablation_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'snr_db', 'best_val_acc', 'checkpoint'])
        for r in results:
            w.writerow([r['model'], r['snr_db'], r['best_val_acc'], r['checkpoint']])

    import pandas as pd
    df = pd.DataFrame(results)
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x='snr_db', y='best_val_acc', hue='model', marker='o')
    plt.gca().invert_xaxis()
    plt.ylim(0.0, 1.0)
    plt.title('Ablation: Best Val Acc vs SNR')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, 'ablation_valacc_vs_snr.png'), dpi=200)
    print('Ablation complete. Artifacts saved to outputs/')


if __name__ == '__main__':
    main()
