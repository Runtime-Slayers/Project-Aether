# Project AETHER - Data Factory Module
# Synthetic RF Signal Generation using TorchSig

"""
Data Factory: Generates synthetic electromagnetic warfare datasets
for training G-NeSAI (Generative Neuro-Symbolic Active Inference) agents.

This module provides:
- TorchSig-based signal generation with 50+ modulation classes
- RF impairments (phase noise, fading, amplifier compression)
- Adversarial signal generation using GANs
- Complex-valued I/Q data pipelines
"""

from .generator import SignalGenerator, generate_training_data
from .spectrum_loader import SpectrumDataset, load_iq_data

__all__ = [
    'SignalGenerator',
    'generate_training_data', 
    'SpectrumDataset',
    'load_iq_data'
]
