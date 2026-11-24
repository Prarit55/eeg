"""
High-Speed AWS S3 Downloader for PhysioNet CHB-MIT Scalp EEG Database.
Uses AWS Open Data (s3://physionet-open/chbmit/1.0.0/) with public unsigned access.
Supports parallel multi-part streaming, automatic file verification, resume, and progress tracking.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.client import Config

BUCKET_NAME = "physionet-open"
PREFIX = "chbmit/1.0.0/"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "eeg"


def get_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_s3_files(
    prefix: str = PREFIX,
    subjects: Optional[List[str]] = None,
    seizures_only: bool = False,
    seizure_records: Optional[set] = None,
) -> List[Dict[str, any]]:
    """List all objects in the S3 bucket matching filters."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    files = []

    subjs_filter = [s.lower() for s in subjects] if subjects else None

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            size = item["Size"]
            rel_path = key[len(prefix):]

            if not rel_path or rel_path.endswith("/"):
                continue

            parts = rel_path.split("/")
            subj = parts[0].lower() if len(parts) > 1 else ""

            # Filter by subject if specified
            if subjs_filter and subj and subj not in subjs_filter:
                continue

            # Filter by seizures only if specified
            if seizures_only and seizure_records and rel_path.endswith(".edf"):
                if rel_path not in seizure_records and rel_path.lower() not in seizure_records:
                    continue

            files.append({
                "key": key,
                "rel_path": rel_path,
                "size": size,
            })

    return files


def download_single_file(item: Dict[str, any], dest_dir: Path) -> Tuple[str, bool, str]:
    """Download a single file from S3 with resume and verification."""
    key = item["key"]
    rel_path = item["rel_path"]
    expected_size = item["size"]
    local_path = dest_dir / rel_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Verification of existing file
    if local_path.exists():
        if local_path.stat().st_size == expected_size:
            return rel_path, True, "Already downloaded (verified size)"

    temp_path = local_path.with_suffix(local_path.suffix + ".part")
    s3 = get_s3_client()

    for attempt in range(1, 4):
        try:
            start_t = time.time()
            s3.download_file(BUCKET_NAME, key, str(temp_path))
            elapsed = max(0.01, time.time() - start_t)

            if temp_path.exists() and temp_path.stat().st_size == expected_size:
                if local_path.exists():
                    local_path.unlink()
                temp_path.rename(local_path)
                rate = (expected_size / (1024 * 1024)) / elapsed
                return rel_path, True, f"{expected_size / (1024*1024):.1f} MB in {elapsed:.1f}s ({rate:.2f} MB/s)"
        except Exception as e:
            if attempt == 3:
                return rel_path, False, f"Failed after 3 attempts: {e}"
            time.sleep(2)

    return rel_path, False, "Unknown failure"


def main():
    parser = argparse.ArgumentParser(description="AWS S3 High-Speed CHB-MIT Database Downloader")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST), help="Destination folder")
    parser.add_argument("--workers", type=int, default=6, help="Parallel download threads (default: 6)")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects (e.g. chb01 chb02)")
    parser.add_argument("--seizures-only", action="store_true", help="Download only recordings with seizures")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AWS S3 PHYSIOMET CHB-MIT EEG DATABASE DOWNLOADER")
    print(f"S3 Source:      s3://{BUCKET_NAME}/{PREFIX}")
    print(f"Local Dest:     {dest_dir}")
    print(f"Parallelism:    {args.workers} concurrent workers")
    print("=" * 80)

    # Load RECORDS-WITH-SEIZURES if available
    seizure_records = set()
    rws_path = dest_dir / "RECORDS-WITH-SEIZURES"
    if rws_path.exists():
        for line in rws_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                seizure_records.add(line.strip().lower())

    print("\nScanning AWS S3 bucket manifest...")
    all_files = list_s3_files(
        prefix=PREFIX,
        subjects=args.subjects,
        seizures_only=args.seizures_only,
        seizure_records=seizure_records,
    )

    total_bytes = sum(f["size"] for f in all_files)
    print(f"Target files: {len(all_files)} total objects ({total_bytes / (1024**3):.2f} GB)")

    # Execute downloads with ThreadPoolExecutor
    print(f"\nStarting parallel downloads with {args.workers} workers...\n")
    start_time = time.time()
    completed_count = 0
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_single_file, f, dest_dir): f for f in all_files}

        for future in as_completed(futures):
            completed_count += 1
            rel_path, ok, msg = future.result()
            if ok:
                success_count += 1
            print(f"[{completed_count:3d}/{len(all_files)}] {rel_path}: {msg}", flush=True)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Download complete: {success_count}/{len(all_files)} files ready ({total_elapsed:.1f}s total).")
    print("=" * 80)


if __name__ == "__main__":
    main()
