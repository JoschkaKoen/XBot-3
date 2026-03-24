#!/usr/bin/env bash
# Install Wan2GP Python deps when the upstream Microsoft onnxruntime-gpu wheel
# (extra-index-url) fails to download (0-byte / DNS / region).
#
# Uses onnxruntime-gpu from PyPI (1.24.4) instead of the CUDA-13 nightly dev wheel.
# Set WAN_VIDEO_DIR if Wan2GP is not at ~/Programming/Wan2GP.
#
# Usage:
#   chmod +x scripts/install_wan2gp_requirements.sh
#   ./scripts/install_wan2gp_requirements.sh

set -euo pipefail
WAN2GP="${WAN_VIDEO_DIR:-${HOME}/Programming/Wan2GP}"
if [ ! -d "$WAN2GP/venv" ]; then
  echo "ERROR: No venv at $WAN2GP/venv — run setup_wan2gp.sh (or create venv) first."
  exit 1
fi
# shellcheck disable=SC1091
source "$WAN2GP/venv/bin/activate"

echo "Installing onnxruntime-gpu from PyPI (avoids broken Azure dev wheel download)…"
pip install "onnxruntime-gpu==1.24.4"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
grep -v '^--extra-index-url' "$WAN2GP/requirements.txt" | grep -v '^onnxruntime-gpu' > "$TMP"

echo "Installing remaining packages from requirements.txt (excluding onnx lines + extra-index)…"
pip install --default-timeout=300 -r "$TMP"

echo "Done. Quick check:"
python -c "import onnxruntime, rembg; print('onnxruntime', onnxruntime.__version__, 'OK')"
