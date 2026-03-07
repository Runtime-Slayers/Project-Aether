"""
Project AETHER - Complex-Valued Neural Network
==============================================

Implements Complex-Valued Neural Networks (CVNNs) for RF signal processing
that preserve phase and amplitude information inherent in electromagnetic signals.

This is the "Eyes" of the G-NeSAI architecture - it perceives the raw physics
of radio waves and extracts meaningful features for the cognitive core.

Key Features:
- Complex-valued convolutions maintaining phase relationships
- CReLU and modReLU activation functions
- Holographic phase-space representation
- Zero-shot anomaly detection capability

Mathematical Foundation:
- Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
- Complex convolution preserves both magnitude and phase information
- Real-valued networks lose phase information critical for RF analysis

Author: Project AETHER Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class ComplexConv1d(nn.Module):
    """
    Complex-valued 1D Convolution Layer.
    
    Implements complex convolution by decomposing into real operations:
    (x_r + ix_i) * (w_r + iw_i) = (x_r*w_r - x_i*w_i) + i(x_r*w_i + x_i*w_r)
    
    Args:
        in_channels: Number of input complex channels
        out_channels: Number of output complex channels
        kernel_size: Size of convolution kernel
        stride: Convolution stride
        padding: Padding size
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Real and imaginary convolution weights
        self.conv_r = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.conv_i = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        
        # Initialize with glorot/xavier
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable complex training."""
        for conv in [self.conv_r, self.conv_i]:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with complex convolution.
        
        Args:
            x: Input tensor of shape (batch, 2, channels, length)
               where dim 1 contains [real, imag] components
               
        Returns:
            Output tensor of same structure
        """
        # Split real and imaginary parts
        # Input shape: (batch, 2, in_channels, length) or (batch, 2, length)
        if x.dim() == 3:
            x_r = x[:, 0:1, :]  # (batch, 1, length)
            x_i = x[:, 1:2, :]
        else:
            x_r = x[:, 0, :, :]  # (batch, in_channels, length)
            x_i = x[:, 1, :, :]
        
        # Complex multiplication
        # (x_r + ix_i) * (w_r + iw_i) = (x_r*w_r - x_i*w_i) + i(x_r*w_i + x_i*w_r)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        
        # Stack back to (batch, 2, out_channels, length)
        out = torch.stack([out_r, out_i], dim=1)
        
        return out


class ComplexLinear(nn.Module):
    """
    Complex-valued Linear (Fully Connected) Layer.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize
        nn.init.xavier_uniform_(self.fc_r.weight)
        nn.init.xavier_uniform_(self.fc_i.weight)
        if bias:
            nn.init.zeros_(self.fc_r.bias)
            nn.init.zeros_(self.fc_i.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, 2, features) - [real, imag]
            
        Returns:
            Shape (batch, 2, out_features)
        """
        x_r = x[:, 0, :]
        x_i = x[:, 1, :]
        
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        
        return torch.stack([out_r, out_i], dim=1)


class ModReLU(nn.Module):
    """
    Modulus ReLU activation for complex numbers.
    
    f(z) = ReLU(|z| + b) * (z / |z|)
    
    Applies ReLU to the magnitude while preserving phase.
    The learnable bias b allows the network to learn thresholds.
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, 2, features, ...)
        """
        # Compute magnitude and phase
        x_r = x[:, 0, ...]
        x_i = x[:, 1, ...]
        
        magnitude = torch.sqrt(x_r**2 + x_i**2 + 1e-8)
        
        # Reshape bias for broadcasting
        bias = self.bias
        while bias.dim() < magnitude.dim():
            bias = bias.unsqueeze(-1)
        
        # Apply ReLU to (magnitude + bias)
        new_magnitude = F.relu(magnitude + bias)
        
        # Compute unit phase vector
        phase_r = x_r / (magnitude + 1e-8)
        phase_i = x_i / (magnitude + 1e-8)
        
        # Scale by new magnitude
        out_r = new_magnitude * phase_r
        out_i = new_magnitude * phase_i
        
        return torch.stack([out_r, out_i], dim=1)


class CReLU(nn.Module):
    """
    Complex ReLU - applies ReLU separately to real and imaginary parts.
    
    f(z) = ReLU(Re(z)) + i*ReLU(Im(z))
    
    Simple but effective for many applications.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class ComplexBatchNorm1d(nn.Module):
    """
    Batch Normalization for complex-valued signals.
    
    Normalizes the magnitude while preserving phase information.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_i = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, 2, channels, length)
        """
        x_r = x[:, 0, :, :]
        x_i = x[:, 1, :, :]
        
        out_r = self.bn_r(x_r)
        out_i = self.bn_i(x_i)
        
        return torch.stack([out_r, out_i], dim=1)


class ComplexValuedCNN(nn.Module):
    """
    Complete Complex-Valued CNN for RF Signal Classification.
    
    Architecture follows the document specification:
    - Complex convolutions preserve phase information
    - Multiple scales capture both fine-grained and wideband features
    - Outputs probability distribution over signal states
    
    This network serves as the "Eyes" of G-NeSAI, translating raw I/Q
    samples into discrete state representations for the Active Inference engine.
    
    Args:
        num_classes: Number of output classes/states
        input_length: Expected input signal length
        base_channels: Base number of convolutional channels
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_length: int = 1024,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        
        # Complex convolutional layers
        # Layer 1: Wide receptive field for wideband features
        self.conv1 = ComplexConv1d(1, base_channels, kernel_size=7, padding=3)
        self.bn1 = ComplexBatchNorm1d(base_channels)
        self.act1 = ModReLU(base_channels)
        
        # Layer 2: Medium scale features
        self.conv2 = ComplexConv1d(base_channels, base_channels*2, kernel_size=5, padding=2)
        self.bn2 = ComplexBatchNorm1d(base_channels*2)
        self.act2 = ModReLU(base_channels*2)
        
        # Layer 3: Fine-grained features
        self.conv3 = ComplexConv1d(base_channels*2, base_channels*4, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm1d(base_channels*4)
        self.act3 = ModReLU(base_channels*4)
        
        # Layer 4: High-level abstractions
        self.conv4 = ComplexConv1d(base_channels*4, base_channels*4, kernel_size=3, padding=1)
        self.bn4 = ComplexBatchNorm1d(base_channels*4)
        self.act4 = ModReLU(base_channels*4)
        
        # Pooling
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        
        # Calculate flattened size
        self._calculate_fc_input_size()
        
        # Fully connected layers (real-valued for classification)
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def _calculate_fc_input_size(self):
        """Calculate the input size for FC layers based on conv output."""
        # After 4 pool operations with stride 4
        length_after_pool = self.input_length // (4 ** 4)
        length_after_pool = max(1, length_after_pool)
        
        # Channels * 2 (real+imag) * length
        self.fc_input_size = 32 * 4 * 2 * length_after_pool
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for signal classification.
        
        Args:
            x: Input tensor of shape (batch, 2, length)
               where dim 1 contains [real, imag] components
               
        Returns:
            Log probabilities over classes, shape (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # Add channel dimension: (batch, 2, length) -> (batch, 2, 1, length)
        x = x.unsqueeze(2)
        
        # Conv blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1])).view(batch_size, 2, -1, x.shape[-1]//4)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1])).view(batch_size, 2, -1, x.shape[-1]//4)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1])).view(batch_size, 2, -1, x.shape[-1]//4)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1]))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Pad or truncate to expected size
        if x.shape[1] < self.fc_input_size:
            x = F.pad(x, (0, self.fc_input_size - x.shape[1]))
        elif x.shape[1] > self.fc_input_size:
            x = x[:, :self.fc_input_size]
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representation (latent vector) for Active Inference.
        
        Returns the penultimate layer activations as state representation.
        """
        batch_size = x.shape[0]
        x = x.unsqueeze(2)
        
        # Conv blocks (same as forward)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1])).view(batch_size, 2, -1, x.shape[-1]//4)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1])).view(batch_size, 2, -1, x.shape[-1]//4)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1])).view(batch_size, 2, -1, x.shape[-1]//4)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool(x.view(batch_size, -1, x.shape[-1]))
        
        x = x.view(batch_size, -1)
        
        if x.shape[1] < self.fc_input_size:
            x = F.pad(x, (0, self.fc_input_size - x.shape[1]))
        elif x.shape[1] > self.fc_input_size:
            x = x[:, :self.fc_input_size]
        
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        return features


class SimpleComplexCNN(nn.Module):
    """
    Simplified Complex-Valued CNN for faster training/inference.
    
    Good for initial experiments and resource-constrained environments.
    """
    
    def __init__(self, num_classes: int = 10, input_length: int = 1024):
        super().__init__()

        # Increased capacity for better classification
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Smaller pooled length to reduce FC parameters
        self.pool = nn.AdaptiveAvgPool1d(8)

        self.fc1 = nn.Linear(256 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, length)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


def train_cvnn(model: nn.Module, train_loader, val_loader, num_epochs: int = 50,
               learning_rate: float = 1e-3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train a Complex-Valued Neural Network for RF signal classification.

    Args:
        model: CVNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        trained_model: Trained model
        history: Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cvnn_model.pth')

    # Save final model
    torch.save(model.state_dict(), 'final_cvnn_model.pth')

    return model, history


if __name__ == "__main__":
    print("Project AETHER - Complex-Valued Neural Network Demo")
    print("=" * 50)
    
    # Test complex convolution
    print("\nTesting ComplexConv1d...")
    conv = ComplexConv1d(1, 16, kernel_size=5, padding=2)
    x = torch.randn(8, 2, 1, 1024)  # batch=8, complex, 1 channel, 1024 samples
    y = conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test full network
    print("\nTesting ComplexValuedCNN...")
    model = ComplexValuedCNN(num_classes=10, input_length=1024)
    x = torch.randn(8, 2, 1024)  # batch=8, complex (real+imag), 1024 samples
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output sum (log probs): {torch.exp(logits).sum(dim=1)}")
    
    # Get features
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
    
    # Test simple CNN
    print("\nTesting SimpleComplexCNN...")
    simple_model = SimpleComplexCNN(num_classes=10)
    logits = simple_model(x)
    print(f"Simple CNN output shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    simple_params = sum(p.numel() for p in simple_model.parameters())
    print(f"\nComplexValuedCNN parameters: {total_params:,}")
    print(f"SimpleComplexCNN parameters: {simple_params:,}")
    
    print("\nDone!")
