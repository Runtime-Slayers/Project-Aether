import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from perception_layer.complex_net import SimpleComplexCNN
from data_factory.generator import SignalGenerator, SignalConfig, ModulationType

MODEL_PATH = 'outputs/model_1768812194.pt'
CFG = {
    'n_classes': 4,
    'samples_per_class': 1200,
    'signal_length': 256,
}
MODS = [ModulationType.BPSK, ModulationType.QPSK, ModulationType.QAM16, ModulationType.PSK8]

def make_dataset():
    gen = SignalGenerator(seed=123)
    X_list = []
    y_list = []
    for i, mod in enumerate(MODS[:CFG['n_classes']]):
        for _ in range(CFG['samples_per_class']):
            cfg = SignalConfig(num_samples=CFG['signal_length'], sample_rate=1e6, snr_db=25.0, phase_noise_std=0.0, frequency_offset=0.0, fading_type='none')
            s, _ = gen.generate(mod, config=cfg)
            if np.iscomplexobj(s):
                xi = np.stack([s.real, s.imag], axis=0).astype(np.float32)
            else:
                xi = np.stack([s.astype(np.float32), np.zeros_like(s, dtype=np.float32)], axis=0)
            X_list.append(xi)
            y_list.append(i)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    X = X - X.mean(axis=2, keepdims=True)
    X = X / (X.std(axis=2, keepdims=True) + 1e-9)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleComplexCNN(num_classes=CFG['n_classes'], input_length=CFG['signal_length']).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X, y = make_dataset()
    split = int(0.8 * len(y))
    X_val, y_val = X[split:], y[split:]

    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix

    with torch.no_grad():
        xb = torch.tensor(X_val).to(device)
        logits = model(xb)
        probs = torch.exp(logits)
        preds = probs.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_val, preds)
    print('Eval accuracy (pure model):', acc)
    print('Confusion matrix:')
    print(confusion_matrix(y_val, preds))

if __name__ == '__main__':
    main()
