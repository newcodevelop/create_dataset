#!/usr/bin/env bash
set -euo pipefail

# ENV_NAME="${1:-xlit310}"
# WORKDIR="${2:-$PWD}"

# MINICONDA=/home/miniconda3

# . $MINICONDA/etc/profile.d/conda.sh

# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# conda create -n "$ENV_NAME" python=3.10 -y

# # Make conda activate work inside non-interactive shell.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$ENV_NAME"

ENV_NAME="${1:-xlit310}"
WORKDIR="${2:-$PWD}"

MINICONDA=/home/miniconda3

# Load conda in non-interactive shell
. "$MINICONDA/etc/profile.d/conda.sh"

# Accept Anaconda default-channel ToS once
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create env if it does not already exist
conda create -n "$ENV_NAME" python=3.10 -y

# Activate env
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install -U "pip<24.1" "setuptools<70" wheel
python -m pip install "numpy<2.0"

# Install deps except TensorFlow. TensorFlow is not needed for Hindi IndicXlit and causes ABI issues via urduhack/tensorboard.
python -m pip install \
  gevent \
  indic-nlp-library \
  mock \
  tensorboardX==2.6.2.2 \
  flask \
  flask-cors \
  pyarrow \
  pydload \
  sacremoses \
  "torch>=2.1,<2.4" \
  "tensorboard==2.15.2"

# Prefer conda-forge fairseq to avoid compiling old fairseq C++ extensions on macOS ARM.
conda install -c conda-forge fairseq -y

# Install IndicXlit without pulling its old dependency stack again.
python -m pip install "ai4bharat-transliteration==1.1.3" --no-deps

# Remove TensorFlow-related packages if they got pulled accidentally.
python -m pip uninstall -y \
  tensorflow \
  tensorflow-macos \
  keras \
  tensorflow-addons \
  tf2crf \
  urduhack \
  tensorboard-data-server \
  tensorboard-plugin-wit || true

# Reinstall clean TensorBoard after uninstall cleanup.
python -m pip install --force-reinstall "tensorboard==2.15.2" "tensorboardX==2.6.2.2"

# Local urduhack stub. Important: filename must be __init__.py, not _init_.py.
mkdir -p "$WORKDIR/urduhack"
cat > "$WORKDIR/urduhack/__init__.py" <<'PY'
def normalize(text):
    return text
PY

python -m pip install pillow

python -m pip install "transformers==4.51.3" sentencepiece sacremoses accelerate huggingface-hub --no-deps

python -m pip install indictranstoolkit


echo "\nEnvironment created: $ENV_NAME"
echo "Run from this directory so the local urduhack stub is used: $WORKDIR"
echo "Use: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python your_script.py ..."
