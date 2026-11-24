"""
High-Speed Staged Uploader for CHB-MIT Scalp EEG Dataset to Hugging Face Hub.
Uploads root metadata and dataset card first, followed by subject-by-subject commits
(chb01 through chb24) for rock-solid reliability, progress tracking, and resumability.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi, create_repo, get_token, whoami

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "eeg"


def upload_with_retry(api: HfApi, **kwargs) -> bool:
    for attempt in range(1, 4):
        try:
            api.upload_folder(**kwargs)
            return True
        except Exception as e:
            print(f"    [Retry {attempt}/3] Error: {e}. Retrying in 5s...", flush=True)
            time.sleep(5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Upload EEG dataset to Hugging Face Datasets")
    parser.add_argument("--repo-id", type=str, default="parthsaini333/chbmit-eeg", help="HF dataset repo ID")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Local directory containing EEG files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--public", action="store_true", default=True, help="Make dataset public")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to upload, e.g. chb01 chb02")
    args = parser.parse_args()

    token = args.token or get_token()
    if not token:
        print("\n[Error] Not authenticated with Hugging Face.")
        print("Please pass '--token <your_token>' or run 'huggingface-cli login'.\n")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = whoami(token=token)
    except Exception as e:
        print(f"[Auth Error]: Failed to authenticate with Hugging Face: {e}")
        sys.exit(1)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"[Error] Source directory '{data_path}' does not exist.")
        sys.exit(1)

    print("=" * 80)
    print("HUGGING FACE DATASET STAGED UPLOADER")
    print(f"Authenticated as: {user_info.get('name', 'Unknown')} ({user_info.get('fullname', '')})")
    print(f"Target Repo     : {args.repo_id}")
    print(f"Source Directory: {data_path}")
    print(f"Visibility      : {'PUBLIC' if args.public else 'PRIVATE'}")
    print("=" * 80)

    # 1. Ensure repository exists
    is_private = not args.public
    print(f"\nVerifying dataset repository '{args.repo_id}' on Hugging Face...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=is_private,
        token=token,
        exist_ok=True,
    )
    print("Repository confirmed: https://huggingface.co/datasets/" + args.repo_id)

    # 2. Upload root metadata files (README.md dataset card, RECORDS, SUBJECT-INFO, etc.)
    print("\n--- [Stage 1/25] Uploading Root Metadata & Dataset Card ---")
    root_files = [f for f in data_path.iterdir() if f.is_file() and not f.name.endswith(".part")]
    print(f"Uploading {len(root_files)} root files: {[f.name for f in root_files]}")

    for rf in root_files:
        try:
            api.upload_file(
                path_or_fileobj=str(rf),
                path_in_repo=rf.name,
                repo_id=args.repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"docs: upload dataset card and {rf.name}",
            )
            print(f"  [OK] Uploaded {rf.name}", flush=True)
        except Exception as e:
            print(f"  [Warning] {rf.name}: {e}", flush=True)

    # 3. Discover and upload subject directories (chb01 through chb24)
    if args.subjects:
        target_subjs = [s.lower() for s in args.subjects]
    else:
        target_subjs = [f"chb{i:02d}" for i in range(1, 25)]

    subject_dirs = [data_path / s for s in target_subjs if (data_path / s).is_dir()]
    print(f"\nFound {len(subject_dirs)} subject directories to upload to Hugging Face.\n")

    for i, sdir in enumerate(subject_dirs, 1):
        subj_name = sdir.name
        edf_count = len(list(sdir.glob("*.edf")))
        total_count = len(list(sdir.iterdir()))
        print(f"--- [Stage {i+1}/{len(subject_dirs)+1}] Uploading {subj_name} ({edf_count} EDFs, {total_count} files) ---", flush=True)

        start_t = time.time()
        ok = upload_with_retry(
            api,
            folder_path=str(sdir),
            path_in_repo=subj_name,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"data: add {subj_name} recordings and annotations",
            ignore_patterns=["*.part", "*.tmp"],
        )
        elapsed = time.time() - start_t
        if ok:
            print(f"  [Done] {subj_name} uploaded successfully in {elapsed:.1f}s.\n", flush=True)
        else:
            print(f"  [Failed] {subj_name} upload encountered issues.\n", flush=True)

    print("=" * 80)
    print("ALL STAGES COMPLETED!")
    print(f"Dataset is live on Hugging Face: https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
