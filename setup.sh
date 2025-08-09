#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <HF_TOKEN>"
  exit 1
fi

HF_TOKEN="$1"
export HF_TOKEN

echo "Using Hugging Face token from argument."

# Upgrade/install dependencies
python3 -m pip install --upgrade -r requirements.txt

# Run your quantization/loader script
python3 ./src/loader.py

echo "Cleaning up temporary cache directories..."

# Remove HuggingFace cache to free disk (adjust if you want to keep cache)
rm -rf ./hf_cache

# Remove offload folder used during quantization
rm -rf ./offload

echo "Cleanup complete."
