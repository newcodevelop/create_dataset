#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-xlit310_final}"
WORKDIR="${2:-$PWD}"
MINICONDA="${MINICONDA:-/home/miniconda3}"

# Load conda in this non-interactive shell.
. "$MINICONDA/etc/profile.d/conda.sh"

# Accept Anaconda default-channel ToS once. Ignore if already accepted.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create env only if it does not exist.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

# Install a new C++ runtime inside the env BEFORE installing/importing torch.
conda install -c conda-forge libstdcxx-ng libgcc-ng -y

# Make the newer conda libstdc++ win for this shell.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Persist the fix every time this conda env is activated later.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/libstdcpp.sh" <<'EOF'
export OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/libstdcpp.sh" <<'EOF'
export LD_LIBRARY_PATH="${OLD_LD_LIBRARY_PATH:-}"
unset OLD_LD_LIBRARY_PATH
EOF

python -m pip install -U "pip<24.1" "setuptools<70" wheel
python -m pip install "numpy<2.0"

# Install deps except TensorFlow. TensorFlow is not needed for Hindi IndicXlit and causes ABI issues.
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
  tqdm \
  "torch>=2.1,<2.4" \
  "tensorboard==2.15.2"

# Prefer conda-forge fairseq to avoid compiling old fairseq C++ extensions.
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

# Local urduhack stub. Important: filename must be __init__.py.
mkdir -p "$WORKDIR/urduhack"
cat > "$WORKDIR/urduhack/__init__.py" <<'PY'
def normalize(text):
    return text
PY

python -m pip install pillow
python -m pip install "transformers==4.51.3" sentencepiece sacremoses accelerate huggingface-hub --no-deps
python -m pip install indictranstoolkit

# Verify that conda's libstdc++ has the symbol and that torch imports.
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.31 >/dev/null
python -c "import os, torch; print('CONDA_PREFIX=', os.environ.get('CONDA_PREFIX')); print('torch=', torch.__version__)"

echo ""
echo "Environment created: $ENV_NAME"
echo "Run from this directory so the local urduhack stub is used: $WORKDIR"
echo "Before running manually later, use:"
echo "  . $MINICONDA/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "Then:"
echo "  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python your_script.py ..."
