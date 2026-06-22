#!/usr/bin/env python3
"""Upload MacaBrainNet model checkpoints to HuggingFace Hub via mirror."""

import os

# 走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import login, create_repo, upload_folder

# 登录 — token 从环境变量读取
token = os.environ.get("HF_TOKEN", "")
if not token:
    raise RuntimeError("Please set HF_TOKEN environment variable, e.g.: HF_TOKEN=hf_xxx python upload_to_hf.py")
login(token=token)
print("Login OK")

# 创建 repo
repo_id = "yhwei/MacaBrainNet"
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Repo {repo_id} ready")

# 模型目录
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swinunetr_models")

# 统计（只上传 best_3d_swinunetr_model.pth + best_model_per_class_dice.pth）
from pathlib import Path
pth_files = sorted(Path(model_dir).rglob("best_3d_swinunetr_model.pth"))
dice_files = sorted(Path(model_dir).rglob("best_model_per_class_dice.pth"))
all_files = pth_files + dice_files
total_gb = sum(f.stat().st_size for f in all_files) / 1e9
print(f"Files: {len(pth_files)} best models + {len(dice_files)} dice dicts = {total_gb:.2f} GB")

print("\nUploading (resumable, interrupt and re-run to continue)...")
upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type="model",
    path_in_repo=".",
    commit_message="Upload MacaBrainNet SwinUNETR checkpoints (skull_stripping + tissue_segmentation, 5-fold each)",
    ignore_patterns=[
        "**/latest_checkpoint.pth",
        "**/curr_3d_swinunetr_model.pth",
    ],
)
print(f"\nDone! https://hf-mirror.com/{repo_id}")
