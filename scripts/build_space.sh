#!/usr/bin/env bash
# Assemble a self-contained Hugging Face Space in dist/space from app/, the package and the demo checkpoints.
#   bash scripts/build_space.sh && cd dist/space && git init && ...  (see README "Live demo")
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist/space && mkdir -p dist/space
cp app/app.py app/README.md app/requirements.txt dist/space/
cp -r masksight dist/space/masksight
cp -r app/examples dist/space/examples
mkdir -p dist/space/models && cp models/*.pt dist/space/models/
echo "*.pt filter=lfs diff=lfs merge=lfs -text" > dist/space/.gitattributes
find dist/space -name __pycache__ -prune -exec rm -rf {} +
du -sh dist/space
