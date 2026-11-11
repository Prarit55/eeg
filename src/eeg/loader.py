"""
EEG signal loading and filtering utilities (CHB-MIT Scalp EEG, .edf format).
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

SFREQ: int = 256
N_CHANNELS: int = 23


def load_and_filter_edf(
    edf_path: str,
    l_freq: float = 0.5,
    h_freq: float = 60.0,
    notch_freq: float = 60.0,
    target_sfreq: int = SFREQ,
    n_channels: int = N_CHANNELS,
) -> Tuple[np.ndarray, float, List[str]]:
    """Load an .edf file, band-pass (0.5-60 Hz) and notch (60 Hz) filter it.

    Returns
    -------
    data     : float32 array, shape (n_channels, n_samples), in microvolts
    sfreq    : sampling frequency after (optional) resampling
    ch_names : list of the kept channel names
    """
    import mne  # local import so module works even if mne is deferred

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # CHB-MIT quirks: remove non-EEG channels like "-", "ECG", "VNS"
    drop = [c for c in raw.ch_names if c in ("-", "ECG", "VNS")]
    if len(raw.ch_names) - len(drop) > n_channels:
        drop += [c for c in raw.ch_names if c.endswith("-1")]
    raw.drop_channels([c for c in drop if c in raw.ch_names])
    raw.rename_channels({c: c[:-2] for c in raw.ch_names if c.endswith("-0")})

    kept_channels = list(raw.ch_names[:n_channels])
    raw.pick(kept_channels)

    if int(raw.info["sfreq"]) != target_sfreq:
        raw.resample(target_sfreq, verbose="ERROR")

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")   # band-pass 0.5-60 Hz
    raw.notch_filter(freqs=notch_freq, verbose="ERROR")          # 60 Hz mains

    data = raw.get_data().astype(np.float32) * 1e6               # V -> uV
    if data.shape[0] < n_channels:
        pad_size = n_channels - data.shape[0]
        padding = np.zeros((pad_size, data.shape[1]), dtype=np.float32)
        data = np.vstack([data, padding])
        for p in range(pad_size):
            kept_channels.append(f"PAD-{p+1}")

    return data, float(raw.info["sfreq"]), kept_channels        # data shape: (23, n_samples)
