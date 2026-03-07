import os
import sys
import time

# Ensure project root is on sys.path for local imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn.functional as F

# Local imports from the project
from data_factory.generator import SignalGenerator, SignalConfig, ModulationType
from perception_layer.complex_net import SimpleComplexCNN
import importlib.util

# Load lightweight ActiveInferenceAgent directly to avoid importing heavy package __init__
spec_ai = importlib.util.spec_from_file_location(
    "cognitive_core_active_inference",
    os.path.join(ROOT, 'cognitive_core', 'active_inference.py')
)
ai_mod = importlib.util.module_from_spec(spec_ai)
spec_ai.loader.exec_module(ai_mod)
ActiveInferenceAgent = ai_mod.ActiveInferenceAgent

# Load DoctrineRules similarly
spec_ns = importlib.util.spec_from_file_location(
    "neuro_symbolic_logic",
    os.path.join(ROOT, 'neuro_symbolic', 'logic.py')
)
ns_mod = importlib.util.module_from_spec(spec_ns)
spec_ns.loader.exec_module(ns_mod)
DoctrineRules = ns_mod.DoctrineRules

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Small experiment configuration for a quick end-to-end run
CFG = {
    'n_classes': 2,
    'samples_per_class': 3000,
    'signal_length': 256,
    'batch_size': 128,
    'epochs': 50,
    'lr': 5e-4,
    'seed': 42,
    'clip_grad_norm': 5.0,
}


# Choose a small set of modulations available in the generator
MODS = [ModulationType.BPSK, ModulationType.QPSK, ModulationType.QAM16, ModulationType.PSK8]
CLASS_NAMES = [m.name for m in MODS]

def make_train_val_dataset(train_seed=1, val_seed=2, val_fraction=0.2):
    """Generate separate train and validation datasets using different RNG seeds.

    This prevents leakage between sets (no shared samples) and ensures
    normalization statistics are computed on the training set only.
    """
    def gen_split(seed, samples_per_class):
        gen = SignalGenerator(seed=seed)
        X_list = []
        y_list = []
        for i, mod in enumerate(MODS[:CFG['n_classes']]):
            for _ in range(samples_per_class):
                cfg = SignalConfig(
                    num_samples=CFG['signal_length'],
                    sample_rate=1e6,
                    snr_db=25.0,
                    phase_noise_std=0.0,
                    frequency_offset=0.0,
                    fading_type='none'
                )
                s, _ = gen.generate(mod, config=cfg)
                if np.iscomplexobj(s):
                    xi = np.stack([s.real, s.imag], axis=0).astype(np.float32)
                else:
                    xi = np.stack([s.astype(np.float32), np.zeros_like(s, dtype=np.float32)], axis=0)
                X_list.append(xi)
                y_list.append(i)
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        # shuffle
        idx = np.random.default_rng(seed).permutation(len(y))
        return X[idx], y[idx]

    samples = CFG['samples_per_class']
    # split samples per class between train and val
    val_per_class = int(np.ceil(samples * val_fraction))
    train_per_class = samples - val_per_class

    X_train, y_train = gen_split(train_seed, train_per_class)
    X_val, y_val = gen_split(val_seed, val_per_class)

    # Compute normalization on training set (global per-channel mean/std) and apply to both
    mean = X_train.mean(axis=(0,2), keepdims=True)
    std = X_train.std(axis=(0,2), keepdims=True) + 1e-9
    # mean/std shapes: (1,2,1) -> broadcastable to (N,2,L)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, y_train, X_val, y_val


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reproducibility: fix random seeds for numpy, python random and torch
    seed = CFG.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic where possible for reproducible runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate disjoint train/val datasets
    X_train, y_train, X_val, y_val = make_train_val_dataset(train_seed=100, val_seed=200, val_fraction=0.2)

    # Convert complex I/Q to magnitude spectrum (log-power) for robust features
    def to_spectrum(X):
        N, C, L = X.shape
        specs = np.fft.fft(X[:,0] + 1j*X[:,1], axis=1)
        mag = np.abs(specs)[:, :L//2]
        mag = np.log1p(mag).astype(np.float32)
        return mag[:, None, :]

    X_train = to_spectrum(X_train)
    X_val = to_spectrum(X_val)

    # torch datasets
    train_tensor = torch.tensor(X_train)
    val_tensor = torch.tensor(X_val)
    y_train_t = torch.tensor(y_train)
    y_val_t = torch.tensor(y_val)

    train_ds = torch.utils.data.TensorDataset(train_tensor, y_train_t)
    val_ds = torch.utils.data.TensorDataset(val_tensor, y_val_t)
    # Use a fixed generator for DataLoader shuffling to make epoch shuffles reproducible
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val = torch.Generator()
    g_val.manual_seed(seed + 1)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CFG['batch_size'], shuffle=True, generator=g_train
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CFG['batch_size'], shuffle=False, generator=g_val
    )

    # Define a compact spectral CNN for magnitude-spectrum inputs
    class SpectralCNN(nn.Module):
        def __init__(self, num_classes, input_length):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.fc1 = nn.Linear(256 * 8, 512)
            self.fc2 = nn.Linear(512, num_classes)
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

    model = SpectralCNN(num_classes=CFG['n_classes'], input_length=X_train.shape[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    ai = ActiveInferenceAgent(n_classes=CFG['n_classes'], device=device)
    doctrine = DoctrineRules(forbidden_modulations=['8PSK'])
    allowed_mask = torch.tensor(doctrine.allowed_mask(CLASS_NAMES[:CFG['n_classes']]), dtype=torch.bool)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'free_energy': []}
    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(CFG['epochs']):
        model.train()
        t_loss = 0.0
        fe_sum = 0.0
        fe_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            # model may expect [B, C, L]
            logits = model(xb)
            loss = criterion(logits, yb.to(device))
            probs = torch.exp(logits)
            fe = ai.free_energy(probs, yb.to(device))
            total_loss = loss + 0.1 * fe
            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping to stabilize training and prevent large updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.get('clip_grad_norm', 5.0))
            optimizer.step()
            t_loss += total_loss.item() * xb.size(0)
            # accumulate free energy for reporting (mean over training samples)
            try:
                fe_sum += (fe.item() if isinstance(fe, torch.Tensor) else float(fe)) * xb.size(0)
            except Exception:
                pass
            fe_count += xb.size(0)
        t_loss /= len(train_loader.dataset)
        history['train_loss'].append(t_loss)

        # validation
        model.eval()
        v_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb.to(device))
                probs = torch.exp(logits)
                # apply doctrine mask to actions
                probs_filtered = ai.preferred_action_filter(probs, allowed_mask)
                v_loss += loss.item() * xb.size(0)
                preds = probs_filtered.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.numpy())
        v_loss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc = accuracy_score(all_targets, all_preds)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(acc)
        # append mean free energy for the epoch if available
        if fe_count > 0:
            history['free_energy'].append(fe_sum / fe_count)
        else:
            history['free_energy'].append(float('nan'))

        print(f"Epoch {epoch+1}/{CFG['epochs']}: train_loss={t_loss:.4f}, val_loss={v_loss:.4f}, val_acc={acc:.3f}")
        # step scheduler on validation accuracy
        scheduler.step(acc)

        # checkpoint best model
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_path = os.path.join(OUTPUT_DIR, f'best_model_{int(time.time())}.pt')
            torch.save(model.state_dict(), best_model_path)

    # Save model and results
    ts = int(time.time())
    model_path = os.path.join(OUTPUT_DIR, f'model_{ts}.pt')
    torch.save(model.state_dict(), model_path)

    # Plots
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.savefig(os.path.join(OUTPUT_DIR, 'val_acc.png'))
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(range(CFG['n_classes']), CLASS_NAMES[:CFG['n_classes']], rotation=45)
    plt.yticks(range(CFG['n_classes']), CLASS_NAMES[:CFG['n_classes']])
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    # Save a small CSV summary
    import csv
    with open(os.path.join(OUTPUT_DIR, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'free_energy'])
        for i in range(len(history['train_loss'])):
            w.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['val_acc'][i], history['free_energy'][i]])

    print('Training complete. Outputs saved to', OUTPUT_DIR)

if __name__ == '__main__':
    train()
