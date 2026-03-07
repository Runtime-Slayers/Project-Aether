#!/usr/bin/env bash
# Create a Kaggle Dataset from a local outputs directory (requires kaggle CLI & ~/.kaggle/kaggle.json)
# Usage: tools/create_kaggle_dataset.sh /path/to/outputs_dir "Dataset Title" "Dataset-Subtitle"

set -euo pipefail
OUT_DIR="$1"
TITLE="$2"
SUBTITLE="$3"

if [ -z "$OUT_DIR" ] || [ -z "$TITLE" ]; then
  echo "Usage: $0 /path/to/outputs_dir "\"Dataset Title\"" "\"Subtitle\"""
  exit 1
fi

if [ ! -d "$OUT_DIR" ]; then
  echo "Output directory not found: $OUT_DIR"
  exit 1
fi

# Prepare dataset metadata
META_DIR="$OUT_DIR/dataset-meta"
mkdir -p "$META_DIR"
cat > "$META_DIR/dataset-metadata.json" <<EOF
{
  "title": "$TITLE",
  "id": "${USER:-user}/gnesai-outputs-$(date +%s)",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ],
  "subtitle": "$SUBTITLE"
}
EOF

# Use kaggle CLI to create dataset
# The CLI expects a directory containing the data and dataset-metadata.json
cp -r "$OUT_DIR"/../"$(basename $OUT_DIR)" "$OUT_DIR"/../dataset_upload_temp || true
kaggle datasets create -p "$OUT_DIR" -m "$META_DIR/dataset-metadata.json"

echo "Dataset creation attempted. Check kaggle CLI output for success."
