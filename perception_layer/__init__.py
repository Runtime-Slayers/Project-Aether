# Project AETHER - Perception Layer Module
# Complex-Valued Neural Networks for RF Signal Processing

"""
Perception Layer: The "Eyes" of the G-NeSAI architecture.

Implements Complex-Valued Neural Networks (CVNNs) that preserve 
phase and amplitude information inherent in electromagnetic signals.

Features:
- Complex-valued convolutions maintaining phase relationships
- Holographic phase-space memory for anomaly detection
- Latent space compression via complex autoencoders
- Zero-shot anomaly detection for LPI radar signals
"""

from .complex_net import ComplexValuedCNN, ComplexConv1d, ComplexLinear
from .latent_space import ComplexAutoencoder, StateEncoder

__all__ = [
    'ComplexValuedCNN',
    'ComplexConv1d', 
    'ComplexLinear',
    'ComplexAutoencoder',
    'StateEncoder'
]
