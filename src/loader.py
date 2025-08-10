import os, shutil
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set. Please export your Hugging Face token before running.")
MODEL_REPO = "ibm-granite/granite-3b-code-instruct-2k"
OUTPUT_DIR = "./granite"

CACHE_DIR = "./hf_cache"
local_model_path = os.path.join(CACHE_DIR, MODEL_REPO.replace("/", "_"))
clean_dir = OUTPUT_DIR

if not os.path.exists(clean_dir):
    print(f"Downloading model {MODEL_REPO}...")
    local_model_path = snapshot_download(
        repo_id=MODEL_REPO,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN
    )
    shutil.move(local_model_path, clean_dir)
    print(f"Copied model files to {clean_dir}")
else:
    print(f"Using cached model at {clean_dir}")
