"""Upload only pure backend files to Hugging Face Spaces."""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"

from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
REPO_ID = "shubham918748/voice-rag"

api = HfApi(token=TOKEN)

print(f"Connecting to Hugging Face Space: {REPO_ID}...")

# 1. Upload Core Backend Config Files
for filename in ["app.py", "requirements.txt", "README.md"]:
    p = Path(filename)
    if p.exists():
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="space",
        )

# 2. Upload Backend & Pipeline Folders (skipping all node_modules, .venv, etc.)
for folder in ["pipeline", "backend"]:
    p = Path(folder)
    if p.exists():
        print(f"Uploading {folder}/ folder...")
        api.upload_folder(
            folder_path=str(p),
            path_in_repo=folder,
            repo_id=REPO_ID,
            repo_type="space",
            ignore_patterns=["__pycache__/**", "*.pyc"]
        )

print("\nSUCCESS! Pure backend files successfully uploaded to Hugging Face Space!")
