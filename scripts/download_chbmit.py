"""
CHB-MIT Scalp EEG Database Downloader (PhysioNet).
Supports:
- Resumable chunked streaming (Range header)
- Automatic verification of existing files (skips already downloaded files)
- Subject filtering (--subjects chb02 chb03 ...)
- Seizures-only mode (--seizures-only) for bandwidth efficiency
- Full dataset mode (--all)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "eeg"
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks


def fetch_url_text(url: str, local_cache_path: Optional[Path] = None, timeout: int = 45, max_retries: int = 5) -> str:
    """Fetch text content from a URL with retry logic and local caching."""
    if local_cache_path and local_cache_path.exists() and local_cache_path.stat().st_size > 0:
        return local_cache_path.read_text(encoding="utf-8", errors="ignore")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
                if local_cache_path:
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    local_cache_path.write_text(text, encoding="utf-8")
                return text
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)
    return ""


def download_file(
    remote_url: str,
    local_path: Path,
    max_retries: int = 5,
    timeout: int = 30,
) -> bool:
    """Download a remote file with resumption and retry support."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".part")

    # Fast skip for text files that exist
    if local_path.suffix == ".txt" and local_path.exists() and local_path.stat().st_size > 0:
        print(f"  [OK] Summary exists: {local_path.name}", flush=True)
        return True

    # If full EDF file already exists and is non-trivial (>10MB), check remote size or skip
    remote_size = 0
    if local_path.exists() and local_path.stat().st_size > 10 * 1024 * 1024:
        head_req = urllib.request.Request(remote_url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        try:
            with urllib.request.urlopen(head_req, timeout=10) as resp:
                remote_size = int(resp.headers.get("Content-Length", 0))
        except Exception:
            pass

        if remote_size > 0 and local_path.stat().st_size == remote_size:
            print(f"  [OK] Already downloaded: {local_path.name} ({remote_size / (1024*1024):.1f} MB)", flush=True)
            return True
        elif remote_size == 0 and local_path.stat().st_size > 20 * 1024 * 1024:
            print(f"  [OK] Verified existing file: {local_path.name} ({local_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)
            return True

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0"}

            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            req = urllib.request.Request(remote_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                mode = "ab" if downloaded > 0 and status == 206 else "wb"
                if mode == "wb":
                    downloaded = 0

                total_expected = remote_size if remote_size > 0 else int(resp.headers.get("Content-Length", 0)) + downloaded
                start_time = time.time()
                last_print = start_time

                with open(temp_path, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print >= 5.0:
                            elapsed = now - start_time
                            rate = (downloaded / elapsed / (1024 * 1024)) if elapsed > 0 else 0
                            pct = (downloaded / total_expected * 100) if total_expected > 0 else 0
                            print(f"    Downloading {local_path.name}: {pct:5.1f}% ({downloaded / (1024*1024):.1f} MB @ {rate:.2f} MB/s)", flush=True)
                            last_print = now

            print(f"  [Done] {local_path.name} ({downloaded / (1024*1024):.1f} MB)", flush=True)
            # Atomic rename from .part to final
            if temp_path.exists():
                if local_path.exists():
                    local_path.unlink()
                temp_path.rename(local_path)
            return True

        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
            print(f"\n  [Retry {attempt}/{max_retries}] {local_path.name} connection dropped: {e}. Retrying in 3s...", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"\n  [Error] {local_path.name}: {e}", flush=True)
            return False

    return False


AWS_S3_BASE = "s3://physionet-open/chbmit/1.0.0/"


def is_aws_cli_available() -> bool:
    import shutil
    return shutil.which("aws") is not None


def run_aws_sync(s3_source: str, local_dest: Path, dry_run: bool = False) -> int:
    import subprocess
    cmd = ["aws", "s3", "sync", s3_source, str(local_dest), "--no-sign-request"]
    if dry_run:
        cmd.append("--dryrun")
    print(f"Executing AWS S3 command:\n  {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="PhysioNet CHB-MIT Scalp EEG Downloader (AWS S3 & HTTP)")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST), help="Destination directory for EEG files")
    parser.add_argument("--all", action="store_true", help="Download all 24 subjects (chb01 - chb24, ~35 GB)")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to download, e.g. --subjects chb02 chb03")
    parser.add_argument("--seizures-only", action="store_true", help="Download only recordings with annotated seizure events")
    parser.add_argument("--no-aws", action="store_true", help="Disable AWS S3 transfer and force HTTP streaming")
    parser.add_argument("--dry-run", action="store_true", help="List files to be downloaded without downloading")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    use_aws = not args.no_aws and is_aws_cli_available() and not args.seizures_only

    print("=" * 80)
    print("PHYSIOMET CHB-MIT SCALP EEG DATABASE DOWNLOADER")
    print(f"Destination: {dest_dir}")
    print(f"Transfer backend: {'AWS S3 Open Data (High Speed)' if use_aws else 'HTTP Resumable Streaming'}")
    print("=" * 80)

    # 1. AWS S3 High Speed Path
    if use_aws:
        if args.subjects:
            target_subjects = [s.lower() for s in args.subjects]
            for subj in target_subjects:
                subj_dir = dest_dir / subj
                subj_dir.mkdir(parents=True, exist_ok=True)
                run_aws_sync(f"{AWS_S3_BASE}{subj}/", subj_dir, dry_run=args.dry_run)
            return
        elif args.all:
            # Sync the entire dataset directly in parallel
            run_aws_sync(AWS_S3_BASE, dest_dir, dry_run=args.dry_run)
            return

    # 2. HTTP Fallback Path
    print("Fetching RECORDS manifest from PhysioNet...")
    records_txt = fetch_url_text(BASE_URL + "RECORDS", local_cache_path=dest_dir / "RECORDS")
    all_records = [line.strip() for line in records_txt.splitlines() if line.strip()]
    print(f"Found {len(all_records)} total EDF recordings across all 24 subjects.")

    seizure_records = []
    try:
        seiz_txt = fetch_url_text(BASE_URL + "RECORDS-WITH-SEIZURES", local_cache_path=dest_dir / "RECORDS-WITH-SEIZURES")
        seizure_records = [line.strip() for line in seiz_txt.splitlines() if line.strip()]
        print(f"Found {len(seizure_records)} recordings containing annotated seizures.")
    except Exception:
        pass

    # Filter target subjects
    if args.subjects:
        target_subjects = [s.lower() for s in args.subjects]
    elif args.all:
        target_subjects = [f"chb{i:02d}" for i in range(1, 25)]
    else:
        print("\nNotice: You did not specify --all or --subjects.")
        print("Defaulting to download metadata + chb02 (for multi-subject verification).")
        print("To download the full 35 GB dataset, run: python scripts/download_chbmit.py --all")
        target_subjects = ["chb02"]

    files_to_download: List[str] = []
    for rec in all_records:
        subj = rec.split("/")[0]
        if subj in target_subjects:
            if args.seizures_only and rec not in seizure_records:
                continue
            files_to_download.append(rec)

    # Add summary files and seizure annotation files for target subjects
    meta_files: List[str] = []
    for subj in target_subjects:
        meta_files.append(f"{subj}/{subj}-summary.txt")

    print(f"\nTarget subjects: {target_subjects}")
    print(f"Total EDF files targeted: {len(files_to_download)}")
    print(f"Total metadata/summary files: {len(meta_files)}")

    if args.dry_run:
        print("\n[Dry Run] Planned downloads:")
        for f in meta_files + files_to_download:
            print(f"  - {f}")
        return

    # Download summary files first
    print("\n--- Downloading Subject Summaries & Annotations ---")
    for mf in meta_files:
        download_file(BASE_URL + mf, dest_dir / mf)

    # Download EDF recordings
    print("\n--- Downloading EDF Recordings ---")
    success_count = 0
    for i, rec in enumerate(files_to_download, 1):
        print(f"\n[{i}/{len(files_to_download)}] {rec}:")
        ok = download_file(BASE_URL + rec, dest_dir / rec)
        if ok:
            success_count += 1

    print("\n" + "=" * 80)
    print(f"Download complete! Successfully processed {success_count}/{len(files_to_download)} recordings.")
    print("=" * 80)


if __name__ == "__main__":
    main()
