#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <REPO_URL> <BRANCH_OR_COMMIT> <GITHUB_TOKEN or 'none'> [USE_GPU]"
  exit 1
fi

REPO_URL="$1"
BRANCH_OR_COMMIT="$2"
GITHUB_TOKEN="$3"
USE_GPU="${4:-0}"  # default to 0 if not provided

export REPO_URL
export BRANCH_OR_COMMIT
if [ "$GITHUB_TOKEN" != "none" ]; then
  export GITHUB_TOKEN
fi
export USE_GPU

echo "Environment variables set:"
echo "REPO_URL=$REPO_URL"
echo "BRANCH_OR_COMMIT=$BRANCH_OR_COMMIT"
echo "GITHUB_TOKEN=$( [ "$GITHUB_TOKEN" = "none" ] && echo "<none>" || echo "<set>" )"
echo "USE_GPU=$USE_GPU"

python3 ./src/docgen.py
