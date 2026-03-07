#!/usr/bin/env bash
# Helper to push and run the full training pipeline on Kaggle using the Kaggle CLI.
# Usage: ./tools/run_on_kaggle.sh /path/to/kaggle.json
# Requires: kaggle CLI installed and Python available locally

set -euo pipefail
KAGGLE_JSON_PATH="$1"
KERNEL_DIR="kaggle_kernel_full"
KERNEL_TITLE="G-NeSAI Full Training Pipeline"
KERNEL_SLUG="gnesai-full-training"
USE_INTERNET=0
PUSH_PREPARED=0
if [ "$#" -ge 2 ]; then
  # optional flags: --internet and --push-prepared
  for arg in "${@:2}"; do
    case "$arg" in
      --internet)
        USE_INTERNET=1
        ;;
      --push-prepared)
        PUSH_PREPARED=1
        ;;
    esac
  done
fi

if [ -z "$KAGGLE_JSON_PATH" ]; then
  echo "Usage: $0 /path/to/kaggle.json"
  exit 1
fi

# Ensure kaggle CLI config dir
mkdir -p ~/.kaggle
# Copy provided credentials into place (do not commit this file)
cp "$KAGGLE_JSON_PATH" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Create kernel dir if missing
mkdir -p "$KERNEL_DIR"

# kernel-metadata.json will be created in the kernel directory
if [ "$USE_INTERNET" -eq 1 ]; then
  ENABLE_INTERNET=true
else
  ENABLE_INTERNET=false
fi

cat > "$KERNEL_DIR/kernel-metadata.json" <<EOF
{
  "id": "${KERNEL_SLUG}",
  "title": "${KERNEL_TITLE}",
  "code_file": "run_on_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": ${ENABLE_INTERNET}
}
EOF

# Copy necessary files into kernel dir (select small set to avoid huge uploads)
rsync -av --exclude='outputs' --exclude='__pycache__' \
  ../training_pipeline.py ../training ../data_factory ../perception_layer ../cognitive_core ../neuro_symbolic "$KERNEL_DIR/"

# Create the kernel runner script
cat > "$KERNEL_DIR/run_on_kaggle.py" <<'PY'
import os
import json
from training_pipeline import create_default_config, GNeSAITrainer

# Load config and run training
cfg = create_default_config()
# Use more aggressive training on Kaggle GPU
cfg['num_epochs'] = 100
cfg['batch_size'] = 128
cfg['learning_rate'] = 5e-4
cfg['device'] = 'cuda'
# optional: set seed for reproducibility
cfg['seed'] = 42
cfg['clip_grad_norm'] = 5.0

trainer = GNeSAITrainer(cfg)
results = trainer.run_full_training_pipeline()
print('Training finished. Results:')
print(json.dumps(results, indent=2))
PY

# If internet is enabled, ensure the runner will attempt to create a dataset
if [ "$USE_INTERNET" -eq 1 ]; then
  # prepend an env var setting to the runner so the kernel process sets it
  sed -i "1iimport os\nos.environ['KAGGLE_CREATE_DATASET']='1'\n" "$KERNEL_DIR/run_on_kaggle.py"
  echo "Enabled dataset creation inside kernel runner (KAGGLE_CREATE_DATASET=1)"
fi

# Push kernel
if [ "$PUSH_PREPARED" -eq 1 ]; then
  # push all prepared kernels under kaggle_kernel_runs/
  if [ -d "../kaggle_kernel_runs" ]; then
    for d in ../kaggle_kernel_runs/*/; do
      echo "Pushing prepared kernel: $d"
      pushd "$d" >/dev/null
      kaggle kernels push -p .
      popd >/dev/null
    done
  else
    echo "No prepared kernels found at ../kaggle_kernel_runs"
  fi
else
  pushd "$KERNEL_DIR" >/dev/null
  kaggle kernels push -p .
  popd >/dev/null
fi

echo "Kernel pushed. To run interactively, open the kernel on Kaggle and start the job."

echo "Note: This script copies the provided kaggle.json into ~/.kaggle — be careful with credentials."