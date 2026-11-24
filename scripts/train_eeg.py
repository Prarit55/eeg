"""
EEG Model Training & Window Caching Script.
Processes real CHB-MIT recordings, generates window-level annotations,
trains the 1D-CNN + BiLSTM EEGEncoder, evaluates seizure detection metrics,
and saves trained weights to `data/processed/eeg_encoder.pt`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eeg.loader import load_and_filter_edf
from src.eeg.models import EEGEncoder
from src.eeg.preprocessor import normalize_windows, segment_windows

DEFAULT_EDF = Path(__file__).resolve().parent.parent / "data" / "raw" / "eeg" / "chb01" / "chb01_03.edf"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "eeg_windows"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "processed" / "eeg_encoder.pt"

# Known annotations for chb01_03 from chb01-summary.txt
SEIZURE_INTERVALS = [(2996, 3036)]  # seconds


def prepare_labeled_windows(
    edf_path: Path | str = DEFAULT_EDF,
    intervals: list = SEIZURE_INTERVALS,
    window_sec: int = 4,
    stride_sec: int = 2,
    sfreq: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice recording into windows and label each window based on seizure intervals."""
    edf_p = Path(edf_path)
    if not edf_p.exists():
        raise FileNotFoundError(f"EDF file not found at {edf_p}. Download it first via download_chbmit.py.")

    print(f"Loading and filtering EDF recording: {edf_p.name}...")
    data, actual_sfreq, ch_names = load_and_filter_edf(str(edf_p), target_sfreq=sfreq)
    
    window_size = int(sfreq * window_sec)   # 1024
    stride = int(sfreq * stride_sec)        # 512
    
    print("Segmenting continuous signal into overlapping 4s windows (50% stride)...")
    raw_windows = segment_windows(data, window_size=window_size, stride=stride)
    norm_windows = normalize_windows(raw_windows)
    
    n_windows = len(norm_windows)
    labels = np.zeros(n_windows, dtype=np.float32)
    timestamps = np.zeros((n_windows, 2), dtype=np.float32)
    
    for i in range(n_windows):
        t_start = (i * stride) / sfreq
        t_end = t_start + window_sec
        timestamps[i] = [t_start, t_end]
        
        # Check overlap with any seizure interval
        for s_start, s_end in intervals:
            if max(t_start, s_start) < min(t_end, s_end):
                labels[i] = 1.0
                break

    x_tensor = torch.from_numpy(norm_windows).float()
    y_tensor = torch.from_numpy(labels).float()
    t_tensor = torch.from_numpy(timestamps).float()
    
    n_seiz = int(labels.sum())
    print(f"Total windows: {n_windows} | Seizure windows: {n_seiz} ({n_seiz/n_windows*100:.2f}%)")
    
    # Cache processed dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PROCESSED_DIR / f"{edf_p.stem}_processed.pt"
    torch.save({
        "windows": x_tensor,
        "labels": y_tensor,
        "timestamps": t_tensor,
        "sfreq": sfreq,
        "ch_names": ch_names,
    }, cache_file)
    print(f"Saved processed windows cache to: {cache_file}")
    
    return x_tensor, y_tensor, t_tensor


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
    print(f"\nTraining EEGEncoder on {device}...")

    # Class weighting to handle extreme imbalance (majority background vs minority seizure)
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

    print("\n--- Model Evaluation on Training Recording ---")
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
    parser = argparse.ArgumentParser(description="EEG Window Processor & Model Trainer")
    parser.add_argument("--edf", default=str(DEFAULT_EDF), help="Path to EDF recording")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("EEG DATABASE PROCESSOR & MODEL TRAINING")
    print("=" * 80)
    x, y, t = prepare_labeled_windows(args.edf)
    train_eeg_encoder(x, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print("=" * 80)
    print("EEG Database window caching and model training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
