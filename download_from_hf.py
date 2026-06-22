#!/usr/bin/env python3
"""Download MacaBrainNet model checkpoints from HuggingFace Hub via mirror."""

import os
import argparse

# 走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download


def download_models(repo_id="yhwei/MacaBrainNet",
                    local_dir=None,
                    allow_patterns=None):
    """
    Download model checkpoints from HF Hub.

    Args:
        repo_id: HF repo ID
        local_dir: local directory to save models (default: ./swinunetr_models)
        allow_patterns: glob patterns to filter (default: best checkpoints only)
    """
    if local_dir is None:
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "swinunetr_models")

    if allow_patterns is None:
        allow_patterns = [
            "*/best_3d_swinunetr_model.pth",
            "*/best_model_per_class_dice.pth",
        ]

    print(f"Downloading models from {repo_id}...")
    print(f"  Local dir: {local_dir}")
    print(f"  Files: {allow_patterns}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        repo_type="model",
        resume_download=True,
    )

    # Verify
    from pathlib import Path
    pth_files = sorted(Path(local_dir).rglob("best_3d_swinunetr_model.pth"))
    if not pth_files:
        print("\n[WARN] No model files found. Trying full download...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="model",
            resume_download=True,
        )
        pth_files = sorted(Path(local_dir).rglob("best_3d_swinunetr_model.pth"))

    print(f"\nDownloaded {len(pth_files)} model checkpoints:")
    for f in pth_files:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f} ({size_mb:.0f} MB)")

    print(f"\nModels saved to: {local_dir}")
    print("Ready for pipeline inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MacaBrainNet models from HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default="yhwei/MacaBrainNet",
                        help="HF repo ID")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Local directory to save models")
    args = parser.parse_args()

    download_models(repo_id=args.repo_id, local_dir=args.local_dir)
