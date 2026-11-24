"""
High-level EEG processing pipeline wrapper.
Supports dynamic multi-subject discovery from the complete PhysioNet CHB-MIT dataset.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.eeg.chb_registry import get_chb_registry
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

DEFAULT_WEIGHTS_PATH: str = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "eeg_encoder.pt")
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
        self.registry = get_chb_registry()

        target_weights = weights_path or DEFAULT_WEIGHTS_PATH
        if target_weights and Path(target_weights).exists():
            self.model.load_state_dict(torch.load(target_weights, map_location=self.device))
        else:
            warnings.warn("EEGPipeline: no trained weights found - using a randomly initialised encoder.")
        self.model.eval()

    def resolve_recording(self, edf_path: Optional[str] = None) -> Tuple[Optional[Path], Optional[str], List[Tuple[int, int]]]:
        """Dynamically resolve any EDF recording identifier from the complete CHB dataset, with HF on-demand fallback."""
        local_p = self.registry.find_local_recording(edf_path)
        if not local_p and edf_path:
            meta = self.registry.get_recording_meta(edf_path)
            if meta:
                # Seamless on-demand fetch from Hugging Face hub if missing locally
                try:
                    from huggingface_hub import hf_hub_download
                    dl_file = hf_hub_download(
                        repo_id="parthsaini333/chbmit-eeg",
                        filename=meta.relative_path,
                        repo_type="dataset",
                        local_dir=str(self.registry.data_dir),
                    )
                    local_p = Path(dl_file)
                    self.registry.refresh_local_files()
                except Exception:
                    pass

        if local_p:
            meta = self.registry.get_recording_meta(local_p.name)
            subj_id = meta.subject_id if meta else local_p.parent.name
            intervals = meta.seizure_intervals if meta else []
            return local_p, subj_id, intervals
        return None, None, []

    def load_windows(self, edf_path: Optional[str] = None) -> Tuple[torch.Tensor, Optional[Path], Optional[str], List[Tuple[int, int]]]:
        """Return normalised windows and recording metadata. Falls back to synthetic dummy EEG if no file found."""
        resolved_p, subj_id, ground_truth_intervals = self.resolve_recording(edf_path)
        if resolved_p and resolved_p.exists():
            data, _, _ = load_and_filter_edf(str(resolved_p))
        else:
            warnings.warn(f"EEG file '{edf_path}' not found - using synthetic dummy EEG.")
            data = make_dummy_eeg()
            subj_id = "synthetic"
            ground_truth_intervals = []

        windows = normalize_windows(segment_windows(data))
        return torch.from_numpy(windows).float(), resolved_p, subj_id, ground_truth_intervals

    @torch.no_grad()
    def analyze(self, edf_path: Optional[str] = None) -> Dict[str, Any]:
        windows, local_path, subj_id, ground_truth_intervals = self.load_windows(edf_path)
        logits, zs = [], []
        for i in range(0, len(windows), self.batch_size):
            batch = windows[i:i + self.batch_size].to(self.device)
            logit, z = self.model(batch)
            logits.append(logit.cpu())
            zs.append(z.cpu())

        logits_t = torch.cat(logits) if logits else torch.empty(0)
        z_t = torch.cat(zs) if zs else torch.empty(0, EMBED_DIM)
        probs = torch.sigmoid(logits_t)

        # Calculate detected temporal event intervals (stride = 2 sec)
        seizure_indices = (probs > 0.5).nonzero(as_tuple=True)[0]
        detected_intervals: List[Tuple[int, int]] = []
        if len(seizure_indices) > 0:
            start_idx = seizure_indices[0].item()
            prev_idx = start_idx
            for idx in seizure_indices[1:]:
                curr_idx = idx.item()
                if curr_idx - prev_idx > 2:  # gap > 4 seconds
                    detected_intervals.append((start_idx * 2, prev_idx * 2 + 4))
                    start_idx = curr_idx
                prev_idx = curr_idx
            detected_intervals.append((start_idx * 2, prev_idx * 2 + 4))

        return {
            "n_windows": torch.tensor(len(windows)),
            "logits": logits_t,
            "seizure_prob": probs,
            "z_eeg": z_t,
            "z_patient": z_t.mean(dim=0, keepdim=True) if len(z_t) > 0 else torch.zeros(1, EMBED_DIM),
            "detected_intervals_sec": detected_intervals,
            "ground_truth_intervals_sec": ground_truth_intervals,
            "file_name": local_path.name if local_path else "synthetic",
            "subject_id": subj_id or "unknown",
        }
