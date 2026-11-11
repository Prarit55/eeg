"""
EEG window segmentation, normalization, and synthetic signal generation.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

SFREQ: int = 256
N_CHANNELS: int = 23
WINDOW_SEC: int = 4
WINDOW_SIZE: int = SFREQ * WINDOW_SEC  # 1024 samples
STRIDE: int = WINDOW_SIZE // 2         # 50 % overlap -> 512 samples


def segment_windows(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """Slice a continuous recording into overlapping windows.

    data   : (n_channels, n_samples)
    return : (n_windows, n_channels, window_size)
    """
    n_ch, n_samples = data.shape
    starts = range(0, n_samples - window_size + 1, stride)
    if len(starts) == 0:
        return np.empty((0, n_ch, window_size), dtype=data.dtype)
    return np.stack([data[:, s:s + window_size] for s in starts], axis=0)


def normalize_windows(windows: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-window, per-channel z-score. shape stays (n_windows, n_channels, window_size)."""
    mean = windows.mean(axis=-1, keepdims=True)  # (W, C, 1)
    std = windows.std(axis=-1, keepdims=True)    # (W, C, 1)
    return (windows - mean) / (std + eps)


def make_dummy_eeg(
    duration_sec: int = 60,
    sfreq: int = SFREQ,
    n_channels: int = N_CHANNELS,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic EEG: noise + 10 Hz alpha + a 3 Hz rhythmic spiking burst (25 %-60 % of the file).

    return : (n_channels, duration_sec * sfreq)
    """
    rng = np.random.default_rng(seed)
    n = duration_sec * sfreq
    t = np.arange(n) / sfreq
    data = rng.normal(0.0, 10.0, size=(n_channels, n))   # background noise (uV)
    data += 15.0 * np.sin(2 * np.pi * 10.0 * t)          # alpha rhythm

    kt = np.arange(-int(0.05 * sfreq), int(0.05 * sfreq) + 1) / sfreq
    spike = 150.0 * np.exp(-(kt / 0.015) ** 2)           # sharp spike template
    for s in np.arange(int(0.25 * n), int(0.60 * n), sfreq // 3):  # ~3 Hz spike train
        lo = s - len(spike) // 2
        if lo >= 0 and lo + len(spike) <= n:
            data[:, lo:lo + len(spike)] += spike
    return data.astype(np.float32)
