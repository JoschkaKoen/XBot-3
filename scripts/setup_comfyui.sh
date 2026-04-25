#!/usr/bin/env bash
# setup_comfyui.sh — One-time ComfyUI setup for XBot 3 WAN test scripts
#
# Run from anywhere:
#   bash scripts/setup_comfyui.sh
#
# What this does (fully self-contained, no dependency on Wan2GP):
#   1. Clone ComfyUI to ~/ComfyUI
#   2. Create venv, install PyTorch nightly (CUDA 12.8, RTX 5070 / Blackwell)
#   3. Install ComfyUI requirements + huggingface_hub
#   4. Install ComfyUI Manager
#   5. Create model subdirectories
#   6. Download Wan 2.1 14B I2V models from Comfy-Org/Wan_2.1_ComfyUI_repackaged (~35 GB)
#   7. Download Wan 2.2 TI2V-5B models from Comfy-Org/Wan_2.2_ComfyUI_Repackaged (~13 GB)
#   8. Place all model files in the correct ComfyUI locations
#
# Space needed: ~42 GB total (8 GB toolchain + 21 GB Wan2.1 fp8 + 13 GB Wan2.2)
#
# After this script:
#   - Start ComfyUI: cd ~/ComfyUI && source venv/bin/activate && python main.py --normalvram --fp16-vae
#   - Open http://127.0.0.1:8188
#   - Export workflow JSONs to XBot 3/workflows/ (see workflows/README.md)
#   - Run test scripts:
#       python test_comfyUI_WAN2.1.py Images/photo.png "subtle motion"
#       python test_comfyUI_WAN2.2.py Images/photo.png "subtle motion"

set -e

COMFYUI_DIR="$HOME/ComfyUI"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ComfyUI setup for XBot 3 WAN test scripts"
echo "  Installing at: $COMFYUI_DIR"
echo "  Space needed: ~42 GB (model downloads only, fp8 quantization)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Clone ComfyUI ─────────────────────────────────────────────────────
# Check for main.py — the directory may exist as an empty skeleton without
# the actual ComfyUI codebase (in which case we remove it and re-clone).
if [ -f "$COMFYUI_DIR/main.py" ]; then
    echo "✓ ComfyUI already installed at $COMFYUI_DIR — skipping clone."
else
    if [ -d "$COMFYUI_DIR" ] && [ ! -d "$COMFYUI_DIR/.git" ]; then
        echo "→ Removing incomplete ComfyUI skeleton at $COMFYUI_DIR ..."
        rm -rf "$COMFYUI_DIR"
    fi
    echo "→ Cloning ComfyUI ..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

cd "$COMFYUI_DIR"

# ── Step 2: Create venv + install PyTorch ────────────────────────────────────
if [ -f "$COMFYUI_DIR/venv/bin/python" ]; then
    echo "✓ venv already exists — skipping creation."
else
    echo "→ Creating Python venv ..."
    python3 -m venv venv
fi

source "$COMFYUI_DIR/venv/bin/activate"
HF_CLI="$COMFYUI_DIR/venv/bin/hf"

echo "→ Installing PyTorch nightly (CUDA 12.8 — required for RTX 50xx Blackwell) ..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

echo "→ Installing ComfyUI requirements ..."
pip install -r requirements.txt

echo "→ Installing huggingface_hub ..."
pip install huggingface_hub

# ── Step 3: ComfyUI Manager ───────────────────────────────────────────────────
MANAGER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"
if [ -d "$MANAGER_DIR" ]; then
    echo "✓ ComfyUI Manager already installed — skipping."
else
    echo "→ Installing ComfyUI Manager ..."
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$MANAGER_DIR"
fi

# ── Step 4: Model directories ────────────────────────────────────────────────
echo "→ Creating model directories ..."
mkdir -p "$COMFYUI_DIR/models/diffusion_models"
mkdir -p "$COMFYUI_DIR/models/vae"
mkdir -p "$COMFYUI_DIR/models/text_encoders"
mkdir -p "$COMFYUI_DIR/models/clip_vision"

# ── Helper: move files matching a glob pattern if not already placed ──────────
place_models() {
    local pattern="$1"
    local dest_dir="$2"
    local src_dir="$3"
    local found=0
    while IFS= read -r -d '' f; do
        found=1
        dst="$dest_dir/$(basename "$f")"
        if [ -e "$dst" ]; then
            echo "  ✓ already placed: $(basename "$f")"
        else
            mv -v "$f" "$dst"
        fi
    done < <(find "$src_dir" -name "$pattern" -print0 2>/dev/null)
    if [ "$found" -eq 0 ]; then
        echo "  ✗ no files matched: $pattern (check download output above)"
    fi
}

# ── Step 5: Download Wan 2.1 14B I2V (~21 GB) ────────────────────────────────
# Using --include to download only the 4 files needed (repo total is 543 GB).
# fp8_e4m3fn diffusion model is used (~14 GB, half the size of fp16, fine for
# 8 GB VRAM + 64 GB RAM with CPU offloading).
WAN21_DL="$COMFYUI_DIR/models/Wan_2.1_download"
echo ""
echo "→ Downloading Wan 2.1 14B I2V files from Comfy-Org (~21 GB) ..."
echo "  Repo: Comfy-Org/Wan_2.1_ComfyUI_repackaged"
echo "  Files: diffusion fp8 (~14 GB), T5 encoder fp8 (~6 GB), VAE, CLIP vision"
echo "  Resume-safe — re-running skips already-downloaded files."
"$HF_CLI" download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    --include "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors" \
    --include "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    --include "split_files/vae/wan_2.1_vae.safetensors" \
    --include "split_files/clip_vision/clip_vision_h.safetensors" \
    --local-dir "$WAN21_DL"

echo ""
echo "→ Placing Wan 2.1 model files ..."

place_models "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors" \
    "$COMFYUI_DIR/models/diffusion_models" "$WAN21_DL"

place_models "umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$COMFYUI_DIR/models/text_encoders" "$WAN21_DL"

place_models "wan_2.1_vae.safetensors" \
    "$COMFYUI_DIR/models/vae" "$WAN21_DL"

place_models "clip_vision_h.safetensors" \
    "$COMFYUI_DIR/models/clip_vision" "$WAN21_DL"

# ── Step 6: Download Wan 2.2 TI2V-5B (~13 GB) ────────────────────────────────
# Only the 5B TI2V diffusion model + VAE.
# The T5 text encoder is shared — already downloaded and placed from Wan 2.1.
WAN22_DL="$COMFYUI_DIR/models/Wan_2.2_download"
echo ""
echo "→ Downloading Wan 2.2 TI2V-5B files from Comfy-Org (~13 GB) ..."
echo "  Repo: Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
echo "  Files: TI2V-5B fp16 (~10 GB), VAE (~2.7 GB)"
"$HF_CLI" download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    --include "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors" \
    --include "split_files/vae/wan2.2_vae.safetensors" \
    --local-dir "$WAN22_DL"

echo ""
echo "→ Placing Wan 2.2 model files ..."

place_models "wan2.2_ti2v_5B_fp16.safetensors" \
    "$COMFYUI_DIR/models/diffusion_models" "$WAN22_DL"

place_models "wan2.2_vae.safetensors" \
    "$COMFYUI_DIR/models/vae" "$WAN22_DL"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  Disk usage:"
du -sh "$COMFYUI_DIR/models/diffusion_models" 2>/dev/null \
    | awk '{print "    diffusion_models:  " $1}'
du -sh "$COMFYUI_DIR/models/vae" 2>/dev/null \
    | awk '{print "    vae:               " $1}'
du -sh "$COMFYUI_DIR/models/text_encoders" 2>/dev/null \
    | awk '{print "    text_encoders:     " $1}'
du -sh "$COMFYUI_DIR/models/clip_vision" 2>/dev/null \
    | awk '{print "    clip_vision:       " $1}'
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Start ComfyUI (leave running in a separate terminal):"
echo "     cd $COMFYUI_DIR && source venv/bin/activate"
echo "     python main.py --normalvram --fp16-vae"
echo ""
echo "  2. Open http://127.0.0.1:8188 in your browser"
echo ""
echo "  3. Export workflow JSONs (see $PROJECT_DIR/workflows/README.md):"
echo "     Wan 2.1: Workflow → Browse Templates → Video"
echo "       → Wan 2.1 Image to Video → Export (API format)"
echo "       → save as $PROJECT_DIR/workflows/comfyui_wan2.1.json"
echo ""
echo "     Wan 2.2: Workflow → Browse Templates → Video"
echo "       → Wan2.2 5B video generation → Export (API format)"
echo "       → save as $PROJECT_DIR/workflows/comfyui_wan2.2.json"
echo ""
echo "  4. Run test scripts from XBot 3 root (ComfyUI must be running):"
echo "     python test_comfyUI_WAN2.1.py Images/photo.png \"subtle motion\""
echo "     python test_comfyUI_WAN2.2.py Images/photo.png \"subtle motion\""
echo "════════════════════════════════════════════════════════════════"
echo ""
