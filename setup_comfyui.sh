#!/usr/bin/env bash
# setup_comfyui.sh — One-time ComfyUI setup for XBot 3 WAN test scripts
#
# Run from anywhere:
#   bash setup_comfyui.sh
#
# Disk usage strategy (saves ~34 GB vs a full download):
#   Wan 2.1 models already exist in Wan2GP/ckpts/ — symlinked into ComfyUI,
#   not re-downloaded.  Only Wan 2.2 (the 5B model you don't have yet) is
#   downloaded fresh (~13 GB).
#
# Space breakdown:
#   ComfyUI + PyTorch nightly + deps  ~8 GB
#   Wan 2.2 download (5B + VAE)      ~13 GB
#   ─────────────────────────────────────────
#   Total new disk usage             ~21 GB   (fits in 46 GB free)
#
# What this does:
#   1. Clone ComfyUI to ~/ComfyUI
#   2. Create venv, install PyTorch nightly (CUDA 12.8, RTX 5070 / Blackwell)
#   3. Install ComfyUI requirements + huggingface_hub
#   4. Install ComfyUI Manager
#   5. Create model subdirectories
#   6. Symlink Wan 2.1 models from existing Wan2GP installation (no download)
#   7. Download only Wan 2.2 5B TI2V (~13 GB) and place files
#
# After this script:
#   - Start ComfyUI: cd ~/ComfyUI && source venv/bin/activate && python main.py --normalvram --fp16-vae
#   - Open http://127.0.0.1:8188
#   - Export workflow JSONs to XBot 3/workflows/ (see workflows/README.md)

set -e

COMFYUI_DIR="$HOME/ComfyUI"
WAN2GP_CKPTS="$HOME/Programming/Wan2GP/ckpts"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ComfyUI setup for XBot 3 WAN test scripts"
echo "  ComfyUI   → $COMFYUI_DIR"
echo "  Wan2GP    → $WAN2GP_CKPTS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Sanity-check: Wan2GP ckpts must exist (Wan 2.1 models live there)
if [ ! -d "$WAN2GP_CKPTS" ]; then
    echo "ERROR: Wan2GP ckpts not found at $WAN2GP_CKPTS"
    echo "       Set WAN2GP_CKPTS at the top of this script if your path differs."
    exit 1
fi

# ── Step 1: Clone ComfyUI ─────────────────────────────────────────────────────
if [ -d "$COMFYUI_DIR" ]; then
    echo "✓ ComfyUI already cloned at $COMFYUI_DIR — skipping."
else
    echo "→ Cloning ComfyUI ..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

cd "$COMFYUI_DIR"

# ── Step 2: Create venv + install PyTorch ────────────────────────────────────
if [ -d "$COMFYUI_DIR/venv" ]; then
    echo "✓ venv already exists — skipping creation."
else
    echo "→ Creating Python venv ..."
    python3 -m venv venv
fi

source "$COMFYUI_DIR/venv/bin/activate"

echo "→ Installing PyTorch nightly (CUDA 12.8 — required for RTX 50xx Blackwell) ..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet

echo "→ Installing ComfyUI requirements ..."
pip install -r requirements.txt --quiet

echo "→ Installing huggingface_hub ..."
pip install huggingface_hub --quiet

# ── Step 3: ComfyUI Manager ───────────────────────────────────────────────────
MANAGER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"
if [ -d "$MANAGER_DIR" ]; then
    echo "✓ ComfyUI Manager already installed — skipping."
else
    echo "→ Installing ComfyUI Manager ..."
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$MANAGER_DIR"
fi

# ── Step 3b: ComfyUI-GGUF (enables loading Wan2GP quanto_int8 models) ────────
# Without this, ComfyUI cannot load the mbf16_int8 safetensors from Wan2GP.
GGUF_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF"
if [ -d "$GGUF_DIR" ]; then
    echo "✓ ComfyUI-GGUF already installed — skipping."
else
    echo "→ Installing ComfyUI-GGUF (quanto/int8 model support) ..."
    git clone https://github.com/city96/ComfyUI-GGUF.git "$GGUF_DIR"
    pip install -r "$GGUF_DIR/requirements.txt" --quiet
fi

# ── Step 4: Model directories ────────────────────────────────────────────────
echo "→ Creating model directories ..."
mkdir -p "$COMFYUI_DIR/models/diffusion_models"
mkdir -p "$COMFYUI_DIR/models/vae"
mkdir -p "$COMFYUI_DIR/models/text_encoders"
mkdir -p "$COMFYUI_DIR/models/clip_vision"

# ── Step 5: Symlink Wan 2.1 models from Wan2GP (no download needed) ──────────
echo ""
echo "→ Symlinking Wan 2.1 models from Wan2GP (saves ~34 GB download) ..."

symlink() {
    local src="$1" dst="$2"
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        echo "  ✓ already exists: $(basename "$dst")"
    elif [ ! -f "$src" ]; then
        echo "  ✗ source not found, skipping: $src"
    else
        ln -s "$src" "$dst"
        echo "  linked: $(basename "$src")"
    fi
}

# Wan 2.1 14B I2V diffusion model (int8, 16 GB)
symlink \
    "$WAN2GP_CKPTS/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors" \
    "$COMFYUI_DIR/models/diffusion_models/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors"

# VAE (shared between 2.1 and 2.2 — link both names)
symlink \
    "$WAN2GP_CKPTS/Wan2.1_VAE.safetensors" \
    "$COMFYUI_DIR/models/vae/Wan2.1_VAE.safetensors"

symlink \
    "$WAN2GP_CKPTS/Wan2.2_VAE.safetensors" \
    "$COMFYUI_DIR/models/vae/Wan2.2_VAE.safetensors"

# T5 text encoder (umt5-xxl int8, 6.3 GB)
symlink \
    "$WAN2GP_CKPTS/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors" \
    "$COMFYUI_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# ── Step 6: Download only Wan 2.2 5B TI2V (~13 GB) ───────────────────────────
echo ""
echo "→ Downloading Wan 2.2 TI2V-5B from Hugging Face (~13 GB) ..."
echo "  This is the only model that needs to be downloaded."

WAN22_DL="$COMFYUI_DIR/models/Wan_2.2_download"
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    --local-dir "$WAN22_DL"

# Move diffusion model to correct location
echo ""
echo "→ Placing Wan 2.2 model files ..."
find "$WAN22_DL" -name "wan2.2*ti2v*5B*.safetensors" | while read -r f; do
    dst="$COMFYUI_DIR/models/diffusion_models/$(basename "$f")"
    if [ ! -e "$dst" ]; then
        mv -v "$f" "$dst"
    else
        echo "  ✓ already exists: $(basename "$f")"
    fi
done

# VAE — only copy if we don't already have it symlinked from Wan2GP
find "$WAN22_DL" -name "wan2.2*vae*.safetensors" | while read -r f; do
    dst="$COMFYUI_DIR/models/vae/$(basename "$f")"
    if [ ! -e "$dst" ]; then
        mv -v "$f" "$dst"
    else
        echo "  ✓ already exists: $(basename "$f")"
    fi
done

# Clean up empty download dir if nothing remains
find "$WAN22_DL" -mindepth 1 -maxdepth 1 -name "*.safetensors" | grep -q . \
    || echo "  (download dir is clean)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete.  Disk usage summary:"
echo ""
du -sh "$COMFYUI_DIR" 2>/dev/null | awk '{print "  ComfyUI total:  " $1}'
echo ""
echo "  NEXT STEPS:"
echo "  1. Start ComfyUI:"
echo "     cd $COMFYUI_DIR && source venv/bin/activate"
echo "     python main.py --normalvram --fp16-vae"
echo ""
echo "  2. Open http://127.0.0.1:8188 in your browser"
echo ""
echo "  3. Export workflow JSONs (see $PROJECT_DIR/workflows/README.md):"
echo "     - Wan 2.1: Workflow → Browse Templates → Video → Wan 2.1 Image to Video"
echo "       → Export (API format)"
echo "       → save as $PROJECT_DIR/workflows/comfyui_wan2.1.json"
echo "     - Wan 2.2: Workflow → Browse Templates → Video → Wan2.2 5B video generation"
echo "       → Export (API format)"
echo "       → save as $PROJECT_DIR/workflows/comfyui_wan2.2.json"
echo ""
echo "  4. Run test scripts from XBot 3 root:"
echo "     python test_comfyUI_WAN2.1.py Images/photo.png \"subtle motion\""
echo "     python test_comfyUI_WAN2.2.py Images/photo.png \"subtle motion\""
echo "════════════════════════════════════════════════════════════════"
echo ""
