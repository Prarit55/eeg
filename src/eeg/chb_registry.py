"""
Universal CHB-MIT Scalp EEG Dataset Registry and Metadata Parser.
Provides full programmatic access across all 24 subjects (chb01 - chb24),
parsing all subject summaries, seizure timestamps, channel montages,
and patient demographics from PhysioNet CHB-MIT.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RAW_EEG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "eeg"


@dataclass
class SeizureEvent:
    start_sec: int
    end_sec: int
    duration_sec: int


@dataclass
class RecordingMeta:
    file_name: str
    subject_id: str
    relative_path: str
    n_seizures: int
    seizure_intervals: List[Tuple[int, int]] = field(default_factory=list)
    local_path: Optional[Path] = None
    is_available_locally: bool = False


@dataclass
class SubjectDemographics:
    subject_id: str
    gender: str
    age_years: float
    total_recordings: int = 0
    total_seizures: int = 0


class CHBRegistry:
    """In-memory index of the entire PhysioNet CHB-MIT Scalp EEG Database (24 subjects)."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else RAW_EEG_DIR
        self.subjects: Dict[str, SubjectDemographics] = {}
        self.recordings: Dict[str, RecordingMeta] = {}
        self.records_with_seizures: set[str] = set()
        self._initialized: bool = False
        self._load_registry()

    def _load_registry(self) -> None:
        if self._initialized:
            return

        # 1. Parse SUBJECT-INFO if available
        subj_info_path = self.data_dir / "SUBJECT-INFO"
        if subj_info_path.exists():
            for line in subj_info_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("Case") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    cid = parts[0].lower()
                    gender = parts[1].upper()
                    try:
                        age = float(parts[2])
                    except ValueError:
                        age = 0.0
                    self.subjects[cid] = SubjectDemographics(
                        subject_id=cid,
                        gender=gender,
                        age_years=age,
                    )

        # 2. Parse RECORDS-WITH-SEIZURES if available
        rws_path = self.data_dir / "RECORDS-WITH-SEIZURES"
        if rws_path.exists():
            for line in rws_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line:
                    self.records_with_seizures.add(line.lower())

        # 3. Parse all subject summary files (chb01 through chb24)
        for i in range(1, 25):
            subj_id = f"chb{i:02d}"
            summary_path = self.data_dir / subj_id / f"{subj_id}-summary.txt"
            if not summary_path.exists():
                summary_path = self.data_dir / f"{subj_id}-summary.txt"

            if summary_path.exists():
                self._parse_summary_file(subj_id, summary_path)

        # 4. Scan local directory to mark local availability
        self.refresh_local_files()
        self._initialized = True

    def _parse_summary_file(self, subject_id: str, summary_file: Path) -> None:
        content = summary_file.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        current_file: Optional[str] = None
        current_n_seiz: int = 0
        current_intervals: List[Tuple[int, int]] = []
        pending_start: Optional[int] = None

        def commit_current():
            nonlocal current_file, current_n_seiz, current_intervals
            if current_file:
                # Normalize file name
                fname = Path(current_file).name
                rel_path = f"{subject_id}/{fname}"
                meta = RecordingMeta(
                    file_name=fname,
                    subject_id=subject_id,
                    relative_path=rel_path,
                    n_seizures=max(current_n_seiz, len(current_intervals)),
                    seizure_intervals=list(current_intervals),
                )
                self.recordings[fname.lower()] = meta
                self.recordings[rel_path.lower()] = meta

                if subject_id in self.subjects:
                    self.subjects[subject_id].total_recordings += 1
                    self.subjects[subject_id].total_seizures += meta.n_seizures

            current_file = None
            current_n_seiz = 0
            current_intervals = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect File Name
            m_fn = re.search(r"File Name:\s*([^\s]+\.edf)", line, re.IGNORECASE)
            if m_fn:
                commit_current()
                current_file = m_fn.group(1).strip()
                continue

            # Detect Number of Seizures
            m_ns = re.search(r"Number of Seizures in File:\s*(\d+)", line, re.IGNORECASE)
            if m_ns:
                current_n_seiz = int(m_ns.group(1))
                continue

            # Detect Seizure Start Time
            m_st = re.search(r"Seizure(?:\s*\d+)?\s*Start Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            if m_st:
                pending_start = int(m_st.group(1))
                continue

            # Detect Seizure End Time
            m_et = re.search(r"Seizure(?:\s*\d+)?\s*End Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            if m_et and pending_start is not None:
                end_t = int(m_et.group(1))
                current_intervals.append((pending_start, end_t))
                pending_start = None

        commit_current()

    def refresh_local_files(self) -> None:
        """Scan the raw directory and link all downloaded .edf files."""
        for edf_path in self.data_dir.rglob("*.edf"):
            fname = edf_path.name.lower()
            if fname in self.recordings:
                self.recordings[fname].local_path = edf_path
                self.recordings[fname].is_available_locally = True
            else:
                # File not in summaries (e.g. custom or unindexed)
                subj_id = edf_path.parent.name.lower()
                self.recordings[fname] = RecordingMeta(
                    file_name=edf_path.name,
                    subject_id=subj_id,
                    relative_path=f"{subj_id}/{edf_path.name}",
                    n_seizures=0,
                    seizure_intervals=[],
                    local_path=edf_path,
                    is_available_locally=True,
                )

    def get_recording_meta(self, identifier: str) -> Optional[RecordingMeta]:
        """Look up recording by name (e.g. 'chb01_03.edf', 'chb01_03', or 'chb01/chb01_03.edf')."""
        key = Path(identifier).name.lower()
        if not key.endswith(".edf"):
            key += ".edf"
        return self.recordings.get(key)

    def get_seizure_intervals(self, identifier: str) -> List[Tuple[int, int]]:
        """Return list of (start_sec, end_sec) for any recording across all 24 subjects."""
        meta = self.get_recording_meta(identifier)
        if meta:
            return meta.seizure_intervals
        return []

    def find_local_recording(self, identifier: Optional[str] = None) -> Optional[Path]:
        """Resolve a local path for an EDF recording."""
        self.refresh_local_files()

        # If specific identifier requested
        if identifier:
            meta = self.get_recording_meta(identifier)
            if meta and meta.is_available_locally and meta.local_path:
                return meta.local_path

            # Direct file path check
            p = Path(identifier)
            if p.exists() and p.suffix.lower() == ".edf":
                return p
            p_rel = self.data_dir / identifier
            if p_rel.exists() and p_rel.suffix.lower() == ".edf":
                return p_rel

        # Fallback to any locally available recording with seizures
        available_seizures = [
            m.local_path for m in self.recordings.values()
            if m.is_available_locally and m.n_seizures > 0 and m.local_path
        ]
        if available_seizures:
            return available_seizures[0]

        # Fallback to any locally available recording
        all_available = [m.local_path for m in self.recordings.values() if m.is_available_locally and m.local_path]
        return all_available[0] if all_available else None

    def list_local_recordings(self) -> List[RecordingMeta]:
        """Return all EDF recordings currently present on disk."""
        self.refresh_local_files()
        seen = set()
        res = []
        for m in self.recordings.values():
            if m.is_available_locally and m.file_name not in seen:
                seen.add(m.file_name)
                res.append(m)
        return res

    def list_all_seizure_recordings(self) -> List[RecordingMeta]:
        """Return metadata for all 141+ recordings with documented seizures in the dataset."""
        seen = set()
        res = []
        for m in self.recordings.values():
            if m.n_seizures > 0 and m.file_name not in seen:
                seen.add(m.file_name)
                res.append(m)
        return res


# Global singleton
_REGISTRY: Optional[CHBRegistry] = None


def get_chb_registry(data_dir: Optional[Path] = None) -> CHBRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = CHBRegistry(data_dir=data_dir)
    return _REGISTRY
