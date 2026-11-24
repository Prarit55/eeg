"""
EEG Model Training & Multi-Subject Window Caching Script.
Processes PhysioNet CHB-MIT recordings across subjects,
automatically extracts window-level ground-truth annotations from the CHB registry,
trains the 1D-CNN + BiLSTM EEGEncoder, evaluates multi-subject seizure detection metrics,
and saves trained weights to `data/processed/eeg_encoder.pt`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eeg.chb_registry import get_chb_registry
from src.eeg.loader import load_and_filter_edf
from src.eeg.models import EEGEncoder
from src.eeg.preprocessor import normalize_windows, segment_windows

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "eeg_windows"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "processed" / "eeg_encoder.pt"


def prepare_recording_windows(
    edf_path: Path | str,
    intervals: Optional[List[Tuple[int, int]]] = None,
    window_sec: int = 4,
    stride_sec: int = 2,
    sfreq: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice recording into windows and label each window based on seizure intervals."""
    edf_p = Path(edf_path)
    if not edf_p.exists():
        raise FileNotFoundError(f"EDF file not found at {edf_p}.")

    reg = get_chb_registry()
    if intervals is None:
        intervals = reg.get_seizure_intervals(edf_p.name)

    print(f"Loading and filtering EDF recording: {edf_p.name} (seizure intervals: {intervals})...")
    data, actual_sfreq, ch_names = load_and_filter_edf(str(edf_p), target_sfreq=sfreq)

    window_size = int(sfreq * window_sec)   # 1024
    stride = int(sfreq * stride_sec)        # 512

    raw_windows = segment_windows(data, window_size=window_size, stride=stride)
    norm_windows = normalize_windows(raw_windows)

    n_windows = len(norm_windows)
    labels = np.zeros(n_windows, dtype=np.float32)
    timestamps = np.zeros((n_windows, 2), dtype=np.float32)

    for i in range(n_windows):
        t_start = (i * stride) / sfreq
        t_end = t_start + window_sec
        timestamps[i] = [t_start, t_end]

        for s_start, s_end in intervals:
            if max(t_start, s_start) < min(t_end, s_end):
                labels[i] = 1.0
                break

    x_tensor = torch.from_numpy(norm_windows).float()
    y_tensor = torch.from_numpy(labels).float()
    t_tensor = torch.from_numpy(timestamps).float()

    n_seiz = int(labels.sum())
    print(f"  Processed {n_windows} windows: {n_windows - n_seiz} background, {n_seiz} seizure windows.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PROCESSED_DIR / f"{edf_p.stem}_processed.pt"
    torch.save({
        "windows": x_tensor,
        "labels": y_tensor,
        "timestamps": t_tensor,
        "sfreq": sfreq,
        "ch_names": ch_names,
    }, cache_file)

    return x_tensor, y_tensor, t_tensor


def prepare_multi_recording_dataset(
    target_files: List[Path],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine windows across multiple EDF recordings from one or more CHB subjects."""
    all_x, all_y, all_t = [], [], []
    for edf in target_files:
        x, y, t = prepare_recording_windows(edf)
        all_x.append(x)
        all_y.append(y)
        all_t.append(t)

    combined_x = torch.cat(all_x, dim=0)
    combined_y = torch.cat(all_y, dim=0)
    combined_t = torch.cat(all_t, dim=0)

    print(f"\nMulti-recording dataset combined:")
    print(f"  Total recordings: {len(target_files)}")
    print(f"  Total windows:    {len(combined_x)}")
    print(f"  Total seizure windows: {int(combined_y.sum())}")
    return combined_x, combined_y, combined_t


def train_eeg_encoder(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: Path | str = DEFAULT_WEIGHTS,
) -> EEGEncoder:
    """Train 1D-CNN + BiLSTM EEGEncoder with balanced class weighting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining EEGEncoder on {device} ({len(x)} windows)...")

    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = EEGEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        if ep % 3 == 0 or ep == epochs:
            print(f"  Epoch [{ep:2d}/{epochs:2d}] - Loss: {avg_loss:.4f}")

    # Evaluate detection performance
    model.eval()
    with torch.no_grad():
        all_logits = []
        for i in range(0, len(x), 64):
            bx = x[i:i + 64].to(device)
            logit, _ = model(bx)
            all_logits.append(logit.cpu())
        preds = torch.sigmoid(torch.cat(all_logits)).numpy()
        targets = y.numpy()

    binary_preds = (preds >= 0.5).astype(int)
    true_pos = int(((binary_preds == 1) & (targets == 1)).sum())
    false_neg = int(((binary_preds == 0) & (targets == 1)).sum())
    true_neg = int(((binary_preds == 0) & (targets == 0)).sum())
    false_pos = int(((binary_preds == 1) & (targets == 0)).sum())

    sensitivity = true_pos / max(1, true_pos + false_neg)
    specificity = true_neg / max(1, true_neg + false_pos)

    print("\n--- Model Evaluation on Combined Dataset ---")
    print(f"Sensitivity (Recall): {sensitivity * 100:.1f}%")
    print(f"Specificity:          {specificity * 100:.1f}%")
    print(f"True Positives: {true_pos} | False Negatives: {false_neg}")
    print(f"True Negatives: {true_neg} | False Positives: {false_pos}")

    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_p)
    print(f"\nSaved trained model weights to: {save_p}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Multi-Subject EEG Processor & Model Trainer")
    parser.add_argument("--all", action="store_true", help="Train on all available local EDF files across subjects")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to include, e.g. --subjects chb01 chb02")
    parser.add_argument("--edf", type=str, default=None, help="Train on specific EDF recording path")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    reg = get_chb_registry()
    local_recordings = reg.list_local_recordings()

    if not local_recordings:
        print("No local EDF recordings found. Please run download_chbmit.py first.")
        return

    if args.edf:
        target_files = [Path(args.edf)]
    elif args.subjects:
        subjs = [s.lower() for s in args.subjects]
        target_files = [m.local_path for m in local_recordings if m.subject_id.lower() in subjs and m.local_path]
    elif args.all:
        target_files = [m.local_path for m in local_recordings if m.local_path]
    else:
        # Default to all locally available recordings
        target_files = [m.local_path for m in local_recordings if m.local_path]

    print("=" * 80)
    print("CHB-MIT MULTI-SUBJECT EEG DATASET TRAINER")
    print(f"Target recordings ({len(target_files)}): {[f.name for f in target_files]}")
    print("=" * 80)

    x, y, t = prepare_multi_recording_dataset(target_files)
    train_eeg_encoder(x, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    print("=" * 80)
    print("Multi-Subject EEG Training Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
