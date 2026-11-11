"""
CHB-MIT Scalp EEG Dataset Loader & Seizure Annotation Parser.
Parses `chbXX-summary.txt` files and indexes all subject recordings.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

DEFAULT_EEG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "eeg"


def parse_subject_summary(summary_path: str | Path) -> Dict[str, Any]:
    """Parse a single chbXX-summary.txt file.

    Extracts:
    - Subject info (channels, sample rate)
    - Per-file metadata (file name, duration, seizure count, seizure start/end times in seconds)
    """
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Extract sample rate
    sfreq_match = re.search(r"Sampling Rate:\s*(\d+)\s*Hz", content, re.I)
    sfreq = int(sfreq_match.group(1)) if sfreq_match else 256

    # Extract channels
    channels = []
    for line in content.splitlines():
        ch_match = re.search(r"Channel \d+:\s*([A-Za-z0-9\-]+)", line)
        if ch_match:
            channels.append(ch_match.group(1))

    # Split per file block
    file_blocks = re.split(r"File Name:\s*", content)[1:]
    files_info = []

    for block in file_blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        filename = lines[0].strip()
        
        # Start/end time
        start_time_match = re.search(r"File Start Time:\s*([0-9:]+)", block)
        end_time_match = re.search(r"File End Time:\s*([0-9:]+)", block)
        
        # Number of seizures
        num_seizures_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        num_seizures = int(num_seizures_match.group(1)) if num_seizures_match else 0

        # Seizure start & end timestamps in seconds
        seizure_windows: List[Tuple[int, int]] = []
        starts = re.findall(r"Seizure\s*(?:\d+)?\s*Start Time:\s*(\d+)\s*seconds", block, re.I)
        ends = re.findall(r"Seizure\s*(?:\d+)?\s*End Time:\s*(\d+)\s*seconds", block, re.I)
        for s, e in zip(starts, ends):
            seizure_windows.append((int(s), int(e)))

        files_info.append({
            "filename": filename,
            "subject": filename.split("_")[0] if "_" in filename else path.parent.name,
            "has_seizure": num_seizures > 0,
            "num_seizures": num_seizures,
            "seizure_intervals_sec": seizure_windows,
            "start_time_str": start_time_match.group(1) if start_time_match else "",
            "end_time_str": end_time_match.group(1) if end_time_match else "",
        })

    return {
        "subject": path.stem.replace("-summary", ""),
        "sampling_rate": sfreq,
        "channels": channels,
        "files": files_info,
    }


class CHBMITDataset:
    """Manages local CHB-MIT recordings and seizure annotations."""

    def __init__(self, eeg_dir: Optional[str | Path] = None) -> None:
        self.eeg_dir = Path(eeg_dir or DEFAULT_EEG_DIR)

    def available_subjects(self) -> List[str]:
        """List subject subdirectories currently available on disk."""
        if not self.eeg_dir.exists():
            return []
        subjects = [
            d.name for d in self.eeg_dir.iterdir()
            if d.is_dir() and d.name.startswith("chb")
        ]
        return sorted(subjects)

    def get_seizure_manifest(self) -> pd.DataFrame:
        """Scan all available subjects and build a DataFrame of all recordings with seizure annotations."""
        records = []
        for subj in self.available_subjects():
            summary_path = self.eeg_dir / subj / f"{subj}-summary.txt"
            if summary_path.exists():
                info = parse_subject_summary(summary_path)
                for f in info["files"]:
                    edf_path = self.eeg_dir / subj / f["filename"]
                    records.append({
                        "subject": f["subject"],
                        "filename": f["filename"],
                        "file_exists": edf_path.exists(),
                        "file_size_mb": round(edf_path.stat().st_size / (1024 * 1024), 2) if edf_path.exists() else 0.0,
                        "has_seizure": f["has_seizure"],
                        "num_seizures": f["num_seizures"],
                        "seizure_intervals_sec": f["seizure_intervals_sec"],
                        "full_path": str(edf_path),
                    })
        return pd.DataFrame(records)
