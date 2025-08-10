#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <HF_TOKEN>"
  exit 1
fi

HF_TOKEN="$1"
export HF_TOKEN

echo "Using Hugging Face token from argument."

python3 -m pip install --upgrade -r requirements.txt

if python3 -c "import torchvision" &> /dev/null; then
    echo "torchvision detected — uninstalling to avoid NMS errors..."
    python3 -m pip uninstall -y torchvision
else
    echo "torchvision not installed — skipping uninstall."
fi

python3 ./src/loader.py

echo "Cleaning up temporary cache directories..."
rm -rf ./hf_cache
echo "Cleanup complete."
