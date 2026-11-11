"""
Upload the CHB-MIT EEG dataset to Hugging Face as a private (or public) dataset.

Usage:
  1. Authenticate with Hugging Face (needs a write token from https://huggingface.co/settings/tokens):
       huggingface-cli login
     or pass --token hf_...

  2. Run upload:
       python scripts/push_to_hf.py --repo-id <username>/chbmit-eeg --private
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi, create_repo, get_token, whoami

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "eeg"


def main():
    parser = argparse.ArgumentParser(description="Upload EEG dataset to Hugging Face Datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo ID, e.g. 'username/chbmit-eeg'")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Local directory containing EEG files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (optional if logged in via CLI)")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default is PRIVATE for team privacy)")
    args = parser.parse_args()

    token = args.token or get_token()
    if not token:
        print("\n[Error] Not authenticated with Hugging Face.")
        print("Please obtain a token with WRITE permissions from: https://huggingface.co/settings/tokens")
        print("Then either run: 'huggingface-cli login' or pass '--token <your_token>'.\n")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = whoami(token=token)
    print("=" * 80)
    print("HUGGING FACE DATASET UPLOADER")
    print(f"Logged in as: {user_info.get('name', 'Unknown')} ({user_info.get('fullname', '')})")
    print(f"Target Repo : {args.repo_id}")
    print(f"Visibility  : {'PUBLIC' if args.public else 'PRIVATE (Only you and invited collaborators)'}")
    print(f"Source Dir  : {args.data_dir}")
    print("=" * 80)

    # 1. Create dataset repository (default is private)
    is_private = not args.public
    print(f"\nEnsuring repository '{args.repo_id}' exists on Hugging Face (private={is_private})...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=is_private,
        token=token,
        exist_ok=True,
    )
    print("Repository confirmed.")

    # 2. Upload folder
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"[Error] Source directory '{data_path}' does not exist.")
        sys.exit(1)

    print(f"\nUploading files from {data_path} to Hugging Face...")
    print("Large files will be uploaded concurrently with progress reporting.\n")

    api.upload_folder(
        folder_path=str(data_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        ignore_patterns=["*.part", "*.tmp", "*.F*"],
    )

    print("\n" + "=" * 80)
    print("Upload complete!")
    print(f"View dataset on Hugging Face: https://huggingface.co/datasets/{args.repo_id}")
    if is_private:
        print("\nTo share with your collaborator:")
        print(f"1. Open https://huggingface.co/datasets/{args.repo_id}/settings")
        print("2. Scroll to 'Collaborators' and add your collaborator's Hugging Face username.")
        print("3. Your collaborator can download the dataset locally via:")
        print(f"   python scripts/pull_from_hf.py --repo-id {args.repo_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
