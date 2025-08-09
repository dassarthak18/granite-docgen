#!/bin/bash
set -e

# Check for token argument before anything else
if [ -z "$1" ]; then
  echo "Usage: $0 <HF_TOKEN>"
  exit 1
fi

HF_TOKEN="$1"
export HF_TOKEN

echo "Using Hugging Face token from argument."

# Upgrade pip packages from requirements.txt
python3 -m pip install --upgrade -r requirements.txt

# Run the model loader
python3 ./src/loader.py
