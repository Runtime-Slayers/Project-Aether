"""
Project AETHER - Latent Space Compression
=========================================

Implements Complex-Valued Autoencoders for compressing raw I/Q signals
into compact latent representations suitable for Active Inference.

The latent space serves as the "state space" for the cognitive core,
mapping continuous RF signals to discrete states for decision-making.

Key Features:
- Complex-valued autoencoder preserving phase information
- Variational formulation for uncertainty quantification
- State quantization via K-means clustering
- Anomaly detection through reconstruction error

Author: Project AETHER Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from sklearn.cluster import KMeans


class ComplexAutoencoder(nn.Module):
    """
    Complex-Valued Autoencoder for RF Signal Compression.
    
    Compresses raw I/Q samples into a compact latent representation
    while preserving phase information critical for RF analysis.
    
    The encoder maps signals to latent vectors, and the decoder
    reconstructs the original signal. The latent space provides
    a compressed state representation for Active Inference.
    
    Args:
        input_length: Length of input I/Q signal
        latent_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        input_length: int = 1024,
        latent_dim: int = 64,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch, 2, input_length)
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Calculate size after convolutions
        self._calc_conv_output_size()
        
        # Latent projection
        self.fc_encode = nn.Linear(self.conv_output_size, latent_dim)
        
        # Decoder projection
        self.fc_decode = nn.Linear(latent_dim, self.conv_output_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 2, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
    
    def _calc_conv_output_size(self):
        """Calculate output size after encoder convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, 2, self.input_length)
            x = self.encoder(x)
            self.conv_output_shape = x.shape[1:]  # (channels, length)
            self.conv_output_size = x.numel()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input signal to latent representation.
        
        Args:
            x: Input tensor (batch, 2, input_length)
            
        Returns:
            Latent vector (batch, latent_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to signal.
        
        Args:
            z: Latent vector (batch, latent_dim)
            
        Returns:
            Reconstructed signal (batch, 2, input_length)
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.conv_output_shape)
        x_recon = self.decoder(h)
        
        # Ensure output matches input length
        if x_recon.size(2) != self.input_length:
            x_recon = F.interpolate(x_recon, size=self.input_length, mode='linear', align_corners=True)
        
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode.
        
        Args:
            x: Input signal (batch, 2, input_length)
            
        Returns:
            x_recon: Reconstructed signal
            z: Latent representation
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly detection.
        
        High reconstruction error indicates anomalous/unknown signals.
        
        Args:
            x: Input signal
            
        Returns:
            Per-sample reconstruction error
        """
        x_recon, _ = self.forward(x)
        error = torch.mean((x - x_recon) ** 2, dim=(1, 2))
        return error


class VariationalComplexAutoencoder(nn.Module):
    """
    Variational Autoencoder for Complex-Valued Signals.
    
    Learns a probabilistic latent space with explicit uncertainty,
    useful for Active Inference where belief uncertainty is important.
    
    The latent space is a Gaussian: z ~ N(mu, sigma^2)
    """
    
    def __init__(
        self,
        input_length: int = 1024,
        latent_dim: int = 64,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        
        self.fc_mu = nn.Linear(128 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 16)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 2, kernel_size=4, stride=4),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to mean and log-variance."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to signal."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 16)
        x_recon = self.decoder(h)
        
        if x_recon.size(2) != self.input_length:
            x_recon = F.interpolate(x_recon, size=self.input_length, mode='linear', align_corners=True)
        
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization.
        
        Returns:
            x_recon: Reconstructed signal
            mu: Latent mean
            logvar: Latent log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction + beta * KL divergence.
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class StateEncoder(nn.Module):
    """
    Encoder that maps continuous latent vectors to discrete states.
    
    Uses K-means clustering to quantize the latent space into
    discrete states suitable for Active Inference.
    
    This bridges the continuous CVNN output to the discrete state
    representation required by the pymdp Active Inference engine.
    
    Args:
        autoencoder: Trained autoencoder model
        num_states: Number of discrete states
        device: Torch device
    """
    
    def __init__(
        self,
        autoencoder: nn.Module,
        num_states: int = 16,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.autoencoder = autoencoder
        self.num_states = num_states
        self.device = device
        
        self.kmeans = None
        self.centroids = None
    
    def fit_clusters(self, data_loader, max_samples: int = 10000):
        """
        Fit K-means clusters on latent representations.
        
        Args:
            data_loader: DataLoader with training signals
            max_samples: Maximum samples to use for fitting
        """
        self.autoencoder.eval()
        latent_vectors = []
        
        with torch.no_grad():
            for batch_signals, _ in data_loader:
                batch_signals = batch_signals.to(self.device)
                
                if hasattr(self.autoencoder, 'encode'):
                    z = self.autoencoder.encode(batch_signals)
                    if isinstance(z, tuple):
                        z = z[0]  # For VAE, take mean
                else:
                    z = self.autoencoder.get_features(batch_signals)
                
                latent_vectors.append(z.cpu().numpy())
                
                if sum(len(v) for v in latent_vectors) >= max_samples:
                    break
        
        latent_vectors = np.concatenate(latent_vectors, axis=0)[:max_samples]
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.num_states, random_state=42, n_init=10)
        self.kmeans.fit(latent_vectors)
        
        self.centroids = torch.tensor(
            self.kmeans.cluster_centers_,
            dtype=torch.float32,
            device=self.device
        )
        
        print(f"Fitted {self.num_states} state clusters on {len(latent_vectors)} samples")
    
    def encode_to_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode signal to discrete state.
        
        Args:
            x: Input signal (batch, 2, length)
            
        Returns:
            state_idx: Discrete state index (batch,)
            state_probs: Probability over states (batch, num_states)
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            if hasattr(self.autoencoder, 'encode'):
                z = self.autoencoder.encode(x.to(self.device))
                if isinstance(z, tuple):
                    z = z[0]
            else:
                z = self.autoencoder.get_features(x.to(self.device))
        
        # Compute distances to centroids
        distances = torch.cdist(z, self.centroids)  # (batch, num_states)
        
        # Convert to probabilities (softmax of negative distances)
        state_probs = F.softmax(-distances, dim=1)
        
        # Hard assignment
        state_idx = torch.argmin(distances, dim=1)
        
        return state_idx, state_probs
    
    def get_state_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over states.
        
        This is the A-matrix likelihood for Active Inference:
        P(observation | state)
        """
        _, state_probs = self.encode_to_state(x)
        return state_probs


if __name__ == "__main__":
    print("Project AETHER - Latent Space Demo")
    print("=" * 50)
    
    # Test autoencoder
    print("\nTesting ComplexAutoencoder...")
    ae = ComplexAutoencoder(input_length=1024, latent_dim=64)
    x = torch.randn(8, 2, 1024)
    
    x_recon, z = ae(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    
    error = ae.reconstruction_error(x)
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean error: {error.mean().item():.4f}")
    
    # Test VAE
    print("\nTesting VariationalComplexAutoencoder...")
    vae = VariationalComplexAutoencoder(input_length=1024, latent_dim=64)
    x_recon, mu, logvar = vae(x)
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    losses = vae.loss_function(x, x_recon, mu, logvar)
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"Recon loss: {losses['recon_loss'].item():.4f}")
    print(f"KL loss: {losses['kl_loss'].item():.4f}")
    
    # Count parameters
    ae_params = sum(p.numel() for p in ae.parameters())
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"\nAutoencoder parameters: {ae_params:,}")
    print(f"VAE parameters: {vae_params:,}")
    
    print("\nDone!")
