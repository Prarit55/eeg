"""
Download the EEG dataset from a private (or public) Hugging Face dataset repo.

Usage:
  python scripts/pull_from_hf.py --repo-id <username>/chbmit-eeg
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import get_token, snapshot_download

DEFAULT_DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "eeg"


def main():
    parser = argparse.ArgumentParser(description="Download EEG dataset from Hugging Face")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo ID, e.g. 'username/chbmit-eeg'")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST_DIR), help="Destination folder on disk")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (required if private repo)")
    args = parser.parse_args()

    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    token = args.token or get_token()

    print("=" * 80)
    print("HUGGING FACE DATASET DOWNLOADER")
    print(f"Dataset Repository: {args.repo_id}")
    print(f"Destination       : {dest_path}")
    print("=" * 80)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(dest_path),
            token=token,
            max_workers=8,
        )
        print("\n" + "=" * 80)
        print(f"Successfully downloaded dataset to {dest_path}!")
        print("=" * 80)
    except Exception as e:
        print(f"\n[Download Failed]: {e}")
        if not token:
            print("Note: If the dataset is PRIVATE, please pass '--token <hf_token>' or run 'huggingface-cli login'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
