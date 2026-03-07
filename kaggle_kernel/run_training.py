"""
Kaggle kernel entrypoint: runs multiclass raw-IQ training on GPU.

This script is self-contained to avoid external imports beyond standard libs
and PyTorch. It generates synthetic data, trains a 1D CNN, saves best model
and evaluation artifacts into the working directory (which Kaggle retains).
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random


class ModulationType:
    BPSK = 'bpsk'
    QPSK = 'qpsk'
    QAM16 = '16qam'


class SignalConfig:
    def __init__(self, num_samples=256, sample_rate=1e6, snr_db=20.0, symbol_rate=50e3):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.symbol_rate = symbol_rate


class SignalGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate(self, mod_type, config: SignalConfig):
        L = config.num_samples
        if mod_type == ModulationType.BPSK:
            const = np.array([1, -1])
        elif mod_type == ModulationType.QPSK:
            const = np.exp(1j * np.pi * np.array([1/4, 3/4, 5/4, 7/4]))
        else:
            # 16-QAM simple grid
            real = np.array([-3, -1, 1, 3])
            imag = np.array([-3, -1, 1, 3])
            const = np.array([r + 1j*i for r in real for i in imag])
            const = const / np.sqrt(np.mean(np.abs(const)**2))

        samples_per_symbol = int(config.sample_rate / config.symbol_rate)
        num_symbols = L // samples_per_symbol + 1
        symbol_indices = self.rng.integers(0, len(const), num_symbols)
        symbols = const[symbol_indices]
        signal = np.repeat(symbols, samples_per_symbol)[:L]
        # Add AWGN
        sig_pow = np.mean(np.abs(signal)**2)
        snr_lin = 10**(config.snr_db/10.0)
        noise_pow = sig_pow / snr_lin
        noise = (np.sqrt(noise_pow/2) * (self.rng.normal(size=L) + 1j*self.rng.normal(size=L)))
        signal = signal + noise
        return signal.astype(np.complex64)


def make_dataset(num_per_class=800, seed=0, snr_db=20.0):
    gen = SignalGenerator(seed=seed)
    MODS = [ModulationType.BPSK, ModulationType.QPSK, ModulationType.QAM16]
    X, y = [], []
    for i, m in enumerate(MODS):
        for _ in range(num_per_class):
            cfg = SignalConfig(num_samples=256, snr_db=snr_db)
            s = gen.generate(m, cfg)
            xi = np.stack([s.real, s.imag], axis=0).astype(np.float32)
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


def train_and_eval(epochs=40, samples_per_class=1000, batch_size=256, lr=5e-4, seed=42, clip_grad_norm=5.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    X_train, y_train = make_dataset(num_per_class=samples_per_class, seed=1, snr_db=20.0)
    X_val, y_val = make_dataset(num_per_class=int(samples_per_class*0.2), seed=2, snr_db=15.0)

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
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_train)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=g_val)

    model = RawCNN(n_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.NLLLoss()

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
            # gradient clipping
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

        print(f"Epoch {epoch+1}/{epochs} | train_loss={tloss:.4f} val_loss={vloss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_path = f'multiclass_raw_best_{int(time.time())}.pt'
            torch.save(model.state_dict(), best_path)

        # stop early if target reached
        if best_acc >= 0.95:
            print('Target achieved. Stopping early.')
            break

    # final evaluation on test
    X_test, y_test = make_dataset(num_per_class=1000, seed=42, snr_db=10.0)
    X_test = (X_test - mean) / std
    xb = torch.tensor(X_test).to(device)
    with torch.no_grad():
        logits = model(xb)
        preds = torch.exp(logits).argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, preds)
    rpt = classification_report(y_test, preds, output_dict=True)

    # save artifacts
    os.makedirs('outputs', exist_ok=True)
    import json
    with open(os.path.join('outputs', 'classification_report.json'), 'w') as f:
        json.dump(rpt, f, indent=2)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BPSK','QPSK','QAM16'], yticklabels=['BPSK','QPSK','QAM16'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'confusion_matrix.png'), dpi=200)

    print('Best val acc:', best_acc)
    print('Saved checkpoint:', best_path)
    print('Saved outputs in outputs/')


if __name__ == '__main__':
    # read env vars for configuration
    EPOCHS = int(os.environ.get('EPOCHS', '40'))
    SAMPLES = int(os.environ.get('SAMPLES_PER_CLASS', '1000'))
    BATCH = int(os.environ.get('BATCH_SIZE', '256'))
    train_and_eval(epochs=EPOCHS, samples_per_class=SAMPLES, batch_size=BATCH)
