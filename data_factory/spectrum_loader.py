"""
Project AETHER - Spectrum Data Loader
=====================================

PyTorch Dataset and DataLoader utilities for loading complex-valued
I/Q signal data for training G-NeSAI cognitive agents.

Features:
- Efficient loading of complex-valued data
- On-the-fly augmentation (noise, fading, frequency offset)
- Support for real SDR captures and synthetic data
- Batch processing optimized for GPU training
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Callable, Union
from pathlib import Path
import h5py


class SpectrumDataset(Dataset):
    """
    PyTorch Dataset for complex-valued RF spectrum data.
    
    Handles both synthetic generated data and loaded data files,
    providing normalized complex I/Q samples suitable for CVNN processing.
    
    Args:
        signals: Complex numpy array of shape (N, num_samples)
        labels: Integer labels of shape (N,)
        transform: Optional transform to apply to each sample
        normalize: Whether to normalize signals to unit power
    """
    
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        self.signals = signals.astype(np.complex64)
        self.labels = labels.astype(np.int64)
        self.transform = transform
        self.normalize = normalize
        
        if len(self.signals) != len(self.labels):
            raise ValueError("Signals and labels must have same length")
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        signal = self.signals[idx].copy()
        label = self.labels[idx]
        
        # Normalize to unit power
        if self.normalize:
            power = np.mean(np.abs(signal) ** 2)
            if power > 0:
                signal = signal / np.sqrt(power)
        
        # Apply transform if provided
        if self.transform is not None:
            signal = self.transform(signal)
        
        # Convert to tensor with real/imag channels
        # Shape: (2, num_samples) - channel 0 is real, channel 1 is imag
        signal_tensor = torch.stack([
            torch.from_numpy(signal.real.astype(np.float32)),
            torch.from_numpy(signal.imag.astype(np.float32))
        ])
        
        return signal_tensor, label
    
    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        transform: Optional[Callable] = None
    ) -> 'SpectrumDataset':
        """
        Load dataset from HDF5 file.
        
        Expected format:
            - 'signals': Complex array (N, num_samples)
            - 'labels': Integer array (N,)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.h5' or filepath.suffix == '.hdf5':
            with h5py.File(filepath, 'r') as f:
                signals = f['signals'][:]
                labels = f['labels'][:]
        elif filepath.suffix == '.npz':
            data = np.load(filepath)
            signals = data['signals']
            labels = data['labels']
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(signals, labels, transform)


class SignalAugmentation:
    """
    Data augmentation transforms for RF signals.
    
    Applies random transformations to improve model generalization:
    - Noise injection
    - Frequency offset
    - Phase rotation
    - Time reversal
    """
    
    def __init__(
        self,
        noise_std_range: Tuple[float, float] = (0, 0.1),
        freq_offset_range: Tuple[float, float] = (-0.01, 0.01),
        phase_rotation: bool = True,
        time_reversal_prob: float = 0.0
    ):
        self.noise_std_range = noise_std_range
        self.freq_offset_range = freq_offset_range
        self.phase_rotation = phase_rotation
        self.time_reversal_prob = time_reversal_prob
    
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        # Random noise
        noise_std = np.random.uniform(*self.noise_std_range)
        if noise_std > 0:
            noise = noise_std * (np.random.randn(len(signal)) + 
                                 1j * np.random.randn(len(signal))) / np.sqrt(2)
            signal = signal + noise
        
        # Random frequency offset
        freq_offset = np.random.uniform(*self.freq_offset_range)
        if freq_offset != 0:
            t = np.arange(len(signal))
            signal = signal * np.exp(1j * 2 * np.pi * freq_offset * t)
        
        # Random phase rotation
        if self.phase_rotation:
            phase = np.random.uniform(0, 2 * np.pi)
            signal = signal * np.exp(1j * phase)
        
        # Time reversal
        if np.random.random() < self.time_reversal_prob:
            signal = signal[::-1].copy()
        
        return signal


def load_iq_data(
    filepath: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = True
) -> DataLoader:
    """
    Load I/Q data and return a DataLoader.
    
    Args:
        filepath: Path to data file (.h5, .hdf5, or .npz)
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of parallel workers
        augment: Whether to apply data augmentation
        
    Returns:
        PyTorch DataLoader ready for training
    """
    transform = SignalAugmentation() if augment else None
    dataset = SpectrumDataset.from_file(filepath, transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def create_train_val_loaders(
    signals: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.2,
    batch_size: int = 32,
    augment_train: bool = True,
    seed: int = 42,
    loader_seed: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from arrays.
    
    Args:
        signals: Complex signal array
        labels: Label array
        val_split: Fraction for validation
        batch_size: Batch size
        augment_train: Whether to augment training data
        seed: Random seed for splitting
        
    Returns:
        (train_loader, val_loader)
    """
    rng = np.random.default_rng(seed)
    
    # Split indices
    n = len(signals)
    indices = rng.permutation(n)
    val_size = int(n * val_split)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    # Create datasets
    train_transform = SignalAugmentation() if augment_train else None
    
    train_dataset = SpectrumDataset(
        signals[train_idx], 
        labels[train_idx],
        transform=train_transform
    )
    val_dataset = SpectrumDataset(
        signals[val_idx],
        labels[val_idx],
        transform=None  # No augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=(torch.Generator().manual_seed(loader_seed if loader_seed is not None else seed))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        generator=(torch.Generator().manual_seed((loader_seed + 1) if loader_seed is not None else (seed + 1)))
    )
    
    return train_loader, val_loader


def save_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    filepath: Union[str, Path],
    label_names: Optional[List[str]] = None
):
    """
    Save dataset to HDF5 file.
    
    Args:
        signals: Complex signal array
        labels: Label array  
        filepath: Output path
        label_names: Optional list of class names
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('signals', data=signals, compression='gzip')
        f.create_dataset('labels', data=labels)
        
        if label_names is not None:
            # Store string list as variable length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('label_names', data=label_names, dtype=dt)
        
        # Store metadata
        f.attrs['num_samples'] = signals.shape[1] if signals.ndim > 1 else 0
        f.attrs['num_classes'] = len(np.unique(labels))
        f.attrs['dataset_size'] = len(signals)


if __name__ == "__main__":
    # Demo usage
    print("Project AETHER - Spectrum Loader Demo")
    print("=" * 50)
    
    # Import generator
    from generator import generate_training_data
    
    # Generate sample data
    signals, labels, label_names = generate_training_data(
        num_samples=500,
        seed=42
    )
    
    print(f"Generated {len(signals)} signals")
    print(f"Signal shape: {signals.shape}")
    print(f"Classes: {label_names}")
    
    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(
        signals, labels,
        val_split=0.2,
        batch_size=32
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    batch_signals, batch_labels = next(iter(train_loader))
    print(f"\nBatch signal shape: {batch_signals.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Signal dtype: {batch_signals.dtype}")
    
    print("\nDone!")
