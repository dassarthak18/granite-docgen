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
python3 ./src/loader.py

echo "Cleaning up temporary cache directories..."
rm -rf ./hf_cache
echo "Cleanup complete."
