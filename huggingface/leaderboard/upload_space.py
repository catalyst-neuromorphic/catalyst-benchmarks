"""Upload leaderboard Space to HuggingFace."""

from huggingface_hub import HfApi, create_repo
import os

TOKEN = os.environ.get("HF_TOKEN", "")
SPACE_ID = "Catalyst-Neuromorphic/snn-benchmark-leaderboard"

api = HfApi(token=TOKEN)

# Create the Space (Gradio type)
try:
    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        token=TOKEN,
        exist_ok=True,
    )
    print(f"Space created/exists: {SPACE_ID}")
except Exception as e:
    print(f"Create repo: {e}")

# Upload all files
space_dir = os.path.dirname(os.path.abspath(__file__))

for fname in ["app.py", "requirements.txt", "README.md"]:
    fpath = os.path.join(space_dir, fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=SPACE_ID,
            repo_type="space",
            token=TOKEN,
        )
        print(f"Uploaded: {fname}")

print(f"\nSpace live at: https://huggingface.co/spaces/{SPACE_ID}")
