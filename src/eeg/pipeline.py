"""
High-level EEG processing pipeline wrapper.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.eeg.loader import N_CHANNELS, SFREQ, load_and_filter_edf
from src.eeg.models import EMBED_DIM, EEGEncoder
from src.eeg.preprocessor import (
    STRIDE,
    WINDOW_SEC,
    WINDOW_SIZE,
    make_dummy_eeg,
    normalize_windows,
    segment_windows,
)

DUMMY_EDF_PATH: str = "data/raw/eeg/chb01/chb01_03.edf"


class EEGPipeline:
    """edf path -> windows -> (seizure probabilities, z_eeg per window, z_eeg per patient)."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.model = EEGEncoder().to(self.device)
        if weights_path and Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            warnings.warn("EEGPipeline: no trained weights found - using a randomly initialised encoder.")
        self.model.eval()

    def load_windows(self, edf_path: Optional[str] = None) -> torch.Tensor:
        """Return normalised windows, shape (n_windows, 23, 1024). Falls back to dummy data."""
        if edf_path and Path(edf_path).exists():
            data, _, _ = load_and_filter_edf(edf_path)
        else:
            warnings.warn(f"EEG file '{edf_path}' not found - using synthetic dummy EEG.")
            data = make_dummy_eeg()
        windows = normalize_windows(segment_windows(data))
        return torch.from_numpy(windows).float()

    @torch.no_grad()
    def analyze(self, edf_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        windows = self.load_windows(edf_path)                       # (W, 23, 1024)
        logits, zs = [], []
        for i in range(0, len(windows), self.batch_size):
            batch = windows[i:i + self.batch_size].to(self.device)  # (B, 23, 1024)
            logit, z = self.model(batch)                            # (B,), (B, 128)
            logits.append(logit.cpu())
            zs.append(z.cpu())
        logits_t = torch.cat(logits)                                # (W,)
        z_t = torch.cat(zs)                                         # (W, 128)
        return {
            "n_windows": torch.tensor(len(windows)),
            "logits": logits_t,                                     # (W,)
            "seizure_prob": torch.sigmoid(logits_t),                # (W,)
            "z_eeg": z_t,                                           # (W, 128)
            "z_patient": z_t.mean(dim=0, keepdim=True),             # (1, 128) recording-level embedding
        }
