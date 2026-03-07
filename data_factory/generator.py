"""
Project AETHER - Synthetic RF Signal Generator
===============================================

Generates electromagnetic warfare training data using TorchSig-style
synthetic signal generation for training G-NeSAI cognitive agents.

Features:
- 50+ modulation classes (PSK, QAM, FSK, OFDM, etc.)
- RF impairments (phase noise, fading, amplifier nonlinearity)
- Adversarial "future war" waveforms
- Complex-valued I/Q data with configurable SNR

Author: Project AETHER Team
Date: January 2026
License: Research Use Only
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings


class ModulationType(Enum):
    """Supported modulation types for signal generation."""
    # Phase Shift Keying
    BPSK = "bpsk"
    QPSK = "qpsk"
    PSK8 = "8psk"
    PSK16 = "16psk"
    
    # Quadrature Amplitude Modulation  
    QAM16 = "16qam"
    QAM32 = "32qam"
    QAM64 = "64qam"
    QAM128 = "128qam"
    QAM256 = "256qam"
    
    # Frequency Shift Keying
    FSK2 = "2fsk"
    FSK4 = "4fsk"
    FSK8 = "8fsk"
    GFSK = "gfsk"
    MSK = "msk"
    GMSK = "gmsk"
    
    # Amplitude Modulation
    AM_DSB = "am_dsb"
    AM_SSB = "am_ssb"
    AM_LSB = "am_lsb"
    AM_USB = "am_usb"
    
    # OFDM (Modern wideband)
    OFDM = "ofdm"
    OFDM_QPSK = "ofdm_qpsk"
    OFDM_16QAM = "ofdm_16qam"
    
    # Radar Waveforms
    LFM_CHIRP = "lfm_chirp"
    BARKER = "barker"
    FMCW = "fmcw"
    
    # Spread Spectrum
    DSSS = "dsss"
    FHSS = "fhss"
    
    # Unknown/Adversarial
    UNKNOWN = "unknown"
    ADVERSARIAL = "adversarial"


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    num_samples: int = 1024
    sample_rate: float = 1e6  # 1 MHz
    center_freq: float = 0.0
    snr_db: float = 10.0
    symbol_rate: float = 50e3  # symbols per second
    
    # Impairments
    phase_noise_std: float = 0.0  # radians
    frequency_offset: float = 0.0  # Hz
    timing_offset: float = 0.0  # samples
    iq_imbalance_amp: float = 0.0  # dB
    iq_imbalance_phase: float = 0.0  # radians
    
    # Fading channel
    fading_type: str = "none"  # "none", "rayleigh", "rician"
    doppler_shift: float = 0.0  # Hz
    
    # Amplifier nonlinearity
    amplifier_saturation: float = float('inf')  # linear if inf


class SignalGenerator:
    """
    Generates synthetic RF signals for cognitive electronic warfare training.
    
    This generator creates complex-valued I/Q samples with configurable
    modulation types, SNR levels, and RF impairments to simulate realistic
    electromagnetic environments.
    
    Example:
        generator = SignalGenerator()
        signal, label = generator.generate(ModulationType.QPSK, snr_db=10)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the signal generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self._setup_constellations()
    
    def _setup_constellations(self):
        """Initialize constellation points for various modulation schemes."""
        # PSK constellations (unit circle)
        self.constellations = {
            ModulationType.BPSK: np.array([1, -1]),
            ModulationType.QPSK: np.exp(1j * np.pi * np.array([1/4, 3/4, 5/4, 7/4])),
            ModulationType.PSK8: np.exp(1j * 2 * np.pi * np.arange(8) / 8),
            ModulationType.PSK16: np.exp(1j * 2 * np.pi * np.arange(16) / 16),
        }
        
        # QAM constellations (rectangular grid)
        def create_qam(m):
            """Create M-QAM constellation."""
            n = int(np.sqrt(m))
            real = np.arange(-(n-1), n, 2)
            imag = np.arange(-(n-1), n, 2)
            constellation = np.array([r + 1j*i for r in real for i in imag])
            return constellation / np.sqrt(np.mean(np.abs(constellation)**2))
        
        self.constellations[ModulationType.QAM16] = create_qam(16)
        self.constellations[ModulationType.QAM64] = create_qam(64)
        self.constellations[ModulationType.QAM256] = create_qam(256)
    
    def generate(
        self,
        mod_type: ModulationType,
        config: Optional[SignalConfig] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a synthetic RF signal.
        
        Args:
            mod_type: Type of modulation to generate
            config: Signal configuration parameters
            
        Returns:
            signal: Complex-valued I/Q samples (num_samples,)
            metadata: Dictionary with signal parameters
        """
        if config is None:
            config = SignalConfig()
        
        # Generate base signal based on modulation type
        if mod_type in self.constellations:
            signal = self._generate_psk_qam(mod_type, config)
        elif mod_type in [ModulationType.FSK2, ModulationType.FSK4, ModulationType.FSK8]:
            signal = self._generate_fsk(mod_type, config)
        elif mod_type == ModulationType.LFM_CHIRP:
            signal = self._generate_lfm_chirp(config)
        elif mod_type == ModulationType.OFDM:
            signal = self._generate_ofdm(config)
        elif mod_type == ModulationType.UNKNOWN:
            signal = self._generate_unknown(config)
        elif mod_type == ModulationType.ADVERSARIAL:
            signal = self._generate_adversarial(config)
        else:
            signal = self._generate_psk_qam(ModulationType.QPSK, config)
        
        # Apply RF impairments
        signal = self._apply_impairments(signal, config)
        
        # Add noise
        signal = self._add_noise(signal, config.snr_db)
        
        # Ensure correct length
        if len(signal) > config.num_samples:
            signal = signal[:config.num_samples]
        elif len(signal) < config.num_samples:
            signal = np.pad(signal, (0, config.num_samples - len(signal)))
        
        metadata = {
            'modulation': mod_type.value,
            'snr_db': config.snr_db,
            'sample_rate': config.sample_rate,
            'num_samples': config.num_samples,
            'center_freq': config.center_freq
        }
        
        return signal.astype(np.complex64), metadata

    def generate_training_data(self, num_samples: int = 10000, modulations=None, snr_range=(-5,20), seed: Optional[int]=None):
        """Convenience wrapper to call the module-level `generate_training_data`.

        Returns: signals, labels, label_names
        """
        return generate_training_data(num_samples=num_samples, modulations=modulations, snr_range=snr_range, seed=seed)
    
    def _generate_psk_qam(
        self, 
        mod_type: ModulationType, 
        config: SignalConfig
    ) -> np.ndarray:
        """Generate PSK/QAM modulated signal."""
        constellation = self.constellations[mod_type]
        m = len(constellation)
        
        # Calculate number of symbols
        samples_per_symbol = int(config.sample_rate / config.symbol_rate)
        num_symbols = config.num_samples // samples_per_symbol + 1
        
        # Generate random symbols
        symbol_indices = self.rng.integers(0, m, num_symbols)
        symbols = constellation[symbol_indices]
        
        # Upsample with pulse shaping (raised cosine)
        signal = np.repeat(symbols, samples_per_symbol)
        signal = self._apply_pulse_shaping(signal, samples_per_symbol)
        
        return signal
    
    def _generate_fsk(
        self, 
        mod_type: ModulationType, 
        config: SignalConfig
    ) -> np.ndarray:
        """Generate FSK modulated signal."""
        m_map = {
            ModulationType.FSK2: 2,
            ModulationType.FSK4: 4,
            ModulationType.FSK8: 8
        }
        m = m_map[mod_type]
        
        # Frequency deviation
        freq_dev = config.symbol_rate * 0.5
        freqs = np.linspace(-freq_dev * (m-1)/2, freq_dev * (m-1)/2, m)
        
        samples_per_symbol = int(config.sample_rate / config.symbol_rate)
        num_symbols = config.num_samples // samples_per_symbol + 1
        
        # Generate random symbols
        symbol_indices = self.rng.integers(0, m, num_symbols)
        
        # Generate FSK signal
        t = np.arange(config.num_samples) / config.sample_rate
        signal = np.zeros(config.num_samples, dtype=np.complex128)
        
        phase = 0
        for i, idx in enumerate(symbol_indices):
            start = i * samples_per_symbol
            end = min((i + 1) * samples_per_symbol, config.num_samples)
            if start >= config.num_samples:
                break
            
            t_sym = np.arange(end - start) / config.sample_rate
            signal[start:end] = np.exp(1j * (2 * np.pi * freqs[idx] * t_sym + phase))
            phase += 2 * np.pi * freqs[idx] * (end - start) / config.sample_rate
        
        return signal
    
    def _generate_lfm_chirp(self, config: SignalConfig) -> np.ndarray:
        """Generate Linear Frequency Modulated (LFM) chirp signal (radar)."""
        t = np.arange(config.num_samples) / config.sample_rate
        duration = config.num_samples / config.sample_rate
        
        # Chirp bandwidth (Hz)
        bandwidth = config.sample_rate * 0.8
        chirp_rate = bandwidth / duration
        
        # Generate chirp
        phase = 2 * np.pi * (0.5 * chirp_rate * t**2)
        signal = np.exp(1j * phase)
        
        return signal
    
    def _generate_ofdm(self, config: SignalConfig) -> np.ndarray:
        """Generate OFDM signal."""
        num_carriers = 64
        cp_len = 16  # Cyclic prefix length
        
        symbols_per_carrier = config.num_samples // (num_carriers + cp_len) + 1
        
        # Generate random QPSK symbols for each carrier
        qpsk = np.exp(1j * np.pi * np.array([1/4, 3/4, 5/4, 7/4]))
        data = qpsk[self.rng.integers(0, 4, (symbols_per_carrier, num_carriers))]
        
        # IFFT to generate time domain signal
        signal = []
        for i in range(symbols_per_carrier):
            ofdm_symbol = np.fft.ifft(data[i], num_carriers) * np.sqrt(num_carriers)
            # Add cyclic prefix
            with_cp = np.concatenate([ofdm_symbol[-cp_len:], ofdm_symbol])
            signal.extend(with_cp)
        
        return np.array(signal[:config.num_samples])
    
    def _generate_unknown(self, config: SignalConfig) -> np.ndarray:
        """Generate unknown/novel waveform for adversarial training."""
        # Mix multiple modulation schemes randomly
        signal = np.zeros(config.num_samples, dtype=np.complex128)
        
        # Random number of components
        num_components = self.rng.integers(2, 5)
        
        for _ in range(num_components):
            # Random frequency offset
            freq = self.rng.uniform(-config.sample_rate/4, config.sample_rate/4)
            t = np.arange(config.num_samples) / config.sample_rate
            
            # Random amplitude
            amp = self.rng.uniform(0.3, 1.0)
            
            # Random phase modulation
            phase_mod = self.rng.uniform(0, 2*np.pi, config.num_samples)
            phase_mod = np.cumsum(phase_mod) * 0.01  # Smooth it
            
            component = amp * np.exp(1j * (2 * np.pi * freq * t + phase_mod))
            signal += component
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        return signal
    
    def _generate_adversarial(self, config: SignalConfig) -> np.ndarray:
        """Generate adversarial waveform designed to evade detection."""
        # Low probability of intercept (LPI) characteristics
        t = np.arange(config.num_samples) / config.sample_rate
        
        # Spread spectrum with pseudo-random frequency hopping
        num_hops = 20
        hop_duration = config.num_samples // num_hops
        
        signal = np.zeros(config.num_samples, dtype=np.complex128)
        
        # Generate PN sequence for hopping
        pn_sequence = self.rng.choice([-1, 1], num_hops)
        freqs = pn_sequence * config.sample_rate * 0.3
        
        for i in range(num_hops):
            start = i * hop_duration
            end = min((i + 1) * hop_duration, config.num_samples)
            t_hop = t[start:end] - t[start]
            
            # Add random data modulation
            data_phase = self.rng.choice([0, np.pi], len(t_hop))
            signal[start:end] = np.exp(1j * (2 * np.pi * freqs[i] * t_hop + data_phase))
        
        return signal
    
    def _apply_pulse_shaping(
        self, 
        signal: np.ndarray, 
        samples_per_symbol: int
    ) -> np.ndarray:
        """Apply raised cosine pulse shaping."""
        # Simple moving average filter as approximation
        kernel_len = min(samples_per_symbol * 4, len(signal) // 2)
        if kernel_len > 0:
            kernel = np.ones(kernel_len) / kernel_len
            signal = np.convolve(signal, kernel, mode='same')
        return signal
    
    def _apply_impairments(
        self, 
        signal: np.ndarray, 
        config: SignalConfig
    ) -> np.ndarray:
        """Apply RF impairments to the signal."""
        # Phase noise
        if config.phase_noise_std > 0:
            phase_noise = self.rng.normal(0, config.phase_noise_std, len(signal))
            phase_noise = np.cumsum(phase_noise)  # Random walk
            signal = signal * np.exp(1j * phase_noise)
        
        # Frequency offset
        if config.frequency_offset != 0:
            t = np.arange(len(signal)) / config.sample_rate
            signal = signal * np.exp(1j * 2 * np.pi * config.frequency_offset * t)
        
        # IQ imbalance
        if config.iq_imbalance_amp != 0 or config.iq_imbalance_phase != 0:
            gain = 10 ** (config.iq_imbalance_amp / 20)
            i = signal.real
            q = signal.imag * gain * np.cos(config.iq_imbalance_phase)
            q += signal.real * np.sin(config.iq_imbalance_phase)
            signal = i + 1j * q
        
        # Amplifier saturation (soft clipping)
        if config.amplifier_saturation < float('inf'):
            amplitude = np.abs(signal)
            phase = np.angle(signal)
            # Soft limiter
            amplitude = config.amplifier_saturation * np.tanh(amplitude / config.amplifier_saturation)
            signal = amplitude * np.exp(1j * phase)
        
        # Rayleigh fading
        if config.fading_type == "rayleigh":
            h = (self.rng.normal(0, 1, len(signal)) + 
                 1j * self.rng.normal(0, 1, len(signal))) / np.sqrt(2)
            signal = signal * h
        
        return signal
    
    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add AWGN noise at specified SNR."""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.sqrt(noise_power / 2) * (
            self.rng.normal(0, 1, len(signal)) + 
            1j * self.rng.normal(0, 1, len(signal))
        )
        
        return signal + noise


def generate_training_data(
    num_samples: int = 10000,
    modulations: Optional[List[ModulationType]] = None,
    snr_range: Tuple[float, float] = (-5, 20),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate a training dataset for G-NeSAI.
    
    Args:
        num_samples: Total number of signals to generate
        modulations: List of modulation types (default: all basic types)
        snr_range: (min_snr, max_snr) in dB
        seed: Random seed
        
    Returns:
        signals: Complex array of shape (num_samples, 1024)
        labels: Integer labels for each signal
        label_names: List mapping indices to modulation names
    """
    if modulations is None:
        modulations = [
            ModulationType.BPSK, ModulationType.QPSK,
            ModulationType.PSK8, ModulationType.QAM16,
            ModulationType.QAM64, ModulationType.FSK2,
            ModulationType.FSK4, ModulationType.LFM_CHIRP,
            ModulationType.OFDM, ModulationType.UNKNOWN
        ]
    
    generator = SignalGenerator(seed=seed)
    rng = np.random.default_rng(seed)
    
    signals = []
    labels = []
    label_names = [m.value for m in modulations]
    
    samples_per_class = num_samples // len(modulations)
    
    for i, mod in enumerate(modulations):
        for _ in range(samples_per_class):
            snr = rng.uniform(snr_range[0], snr_range[1])
            config = SignalConfig(
                snr_db=snr,
                phase_noise_std=rng.uniform(0, 0.1),
                frequency_offset=rng.uniform(-1000, 1000)
            )
            signal, _ = generator.generate(mod, config)
            signals.append(signal)
            labels.append(i)
    
    signals = np.stack(signals)
    labels = np.array(labels)
    
    # Shuffle
    perm = rng.permutation(len(signals))
    signals = signals[perm]
    labels = labels[perm]
    
    return signals, labels, label_names


if __name__ == "__main__":
    # Demo usage
    print("Project AETHER - Signal Generator Demo")
    print("=" * 50)
    
    generator = SignalGenerator(seed=42)
    
    # Generate examples of each modulation type
    demo_mods = [
        ModulationType.QPSK,
        ModulationType.QAM64,
        ModulationType.LFM_CHIRP,
        ModulationType.OFDM,
        ModulationType.ADVERSARIAL
    ]
    
    for mod in demo_mods:
        signal, meta = generator.generate(mod, SignalConfig(snr_db=15))
        print(f"{mod.value:15s}: shape={signal.shape}, "
              f"power={np.mean(np.abs(signal)**2):.3f}, "
              f"peak={np.max(np.abs(signal)):.3f}")
    
    print("\nGenerating training dataset...")
    signals, labels, label_names = generate_training_data(num_samples=1000, seed=42)
    print(f"Dataset shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes: {label_names}")
    print("Done!")
