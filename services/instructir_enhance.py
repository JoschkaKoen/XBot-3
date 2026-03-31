"""
Optional InstructIR post-enhancement for Z-Image-Turbo outputs.

Runs InstructIR in a **subprocess** so that the CUDA context it creates is
completely released when the subprocess exits.  This leaves the full VRAM
budget available for the next heavy model (e.g. Wan2.1).

Requires a local clone of https://github.com/mv-lab/InstructIR and weights
(``im_instructir-7d.pt``, ``lm_instructir-7d.pt`` in the repo root, or
auto-download via huggingface_hub when missing).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("german_bot.instructir_enhance")

LM_MODEL = "lm_instructir-7d.pt"
MODEL_NAME = "im_instructir-7d.pt"
CONFIG_REL = Path("configs") / "eval5d.yml"

# ── subprocess script ─────────────────────────────────────────────────────────
# Executed as:  python -c _ENHANCE_CODE <image_path> <ir_dir> <prompt>
# The script lives entirely inside the child process; when it exits all GPU
# memory (including the CUDA context) is freed automatically.
_ENHANCE_CODE = r"""
import sys, os, argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
import tempfile

def _dict2namespace(config):
    ns = argparse.Namespace()
    for key, value in config.items():
        setattr(ns, key, _dict2namespace(value) if isinstance(value, dict) else value)
    return ns

def _torch_load(path):
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)

image_path  = sys.argv[1]
ir_dir      = Path(sys.argv[2])
prompt      = sys.argv[3]
output_path = sys.argv[4]

root_str = str(ir_dir)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from models import instructir
from text.models import LanguageModel, LMHead

cfg_path = ir_dir / "configs" / "eval5d.yml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = _dict2namespace(yaml.safe_load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = instructir.create_model(
    input_channels=cfg.model.in_ch,
    width=cfg.model.width,
    enc_blks=cfg.model.enc_blks,
    middle_blk_num=cfg.model.middle_blk_num,
    dec_blks=cfg.model.dec_blks,
    txtdim=cfg.model.textdim,
).to(device)
model.load_state_dict(_torch_load(ir_dir / "im_instructir-7d.pt"), strict=True)
model.eval()

language_model = LanguageModel(model=cfg.llm.model)
lm_head = LMHead(
    embedding_dim=cfg.llm.model_dim,
    hidden_dim=cfg.llm.embd_dim,
    num_classes=cfg.llm.nclasses,
).to(device)
lm_head.load_state_dict(_torch_load(ir_dir / "lm_instructir-7d.pt"), strict=True)
lm_head.eval()

with Image.open(image_path) as im:
    im = im.convert("RGB")
    original_size = im.size
    pil_in = im.copy()

img = np.array(pil_in).astype(np.float32) / 255.0
y   = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

lm_embd = language_model(prompt).to(device)
with torch.no_grad():
    text_embd, _ = lm_head(lm_embd)
    x_hat = model(y, text_embd)

restored = x_hat.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().detach().numpy()
restored = (np.clip(restored, 0.0, 1.0) * 255.0).round().astype(np.uint8)
result   = Image.fromarray(restored)

if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)

out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
fd, tmp_path = tempfile.mkstemp(suffix=".png", dir=out_dir)
os.close(fd)
result.save(tmp_path, format="PNG")
os.replace(tmp_path, output_path)
print(f"enhanced:{output_path}", flush=True)
"""


# ── weight downloader ─────────────────────────────────────────────────────────

def _ensure_weights(ir_dir: Path) -> None:
    im_path = ir_dir / MODEL_NAME
    lm_path = ir_dir / LM_MODEL
    if im_path.is_file() and lm_path.is_file():
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Missing {MODEL_NAME} / {LM_MODEL} under {ir_dir} and huggingface_hub "
            "is not installed — install huggingface-hub or copy the checkpoints from "
            "https://huggingface.co/marcosv/InstructIR"
        ) from exc
    logger.info("InstructIR: downloading checkpoints into %s …", ir_dir)
    hf_hub_download(repo_id="marcosv/InstructIR", filename=MODEL_NAME, local_dir=str(ir_dir))
    hf_hub_download(repo_id="marcosv/InstructIR", filename=LM_MODEL, local_dir=str(ir_dir))


# ── public API ────────────────────────────────────────────────────────────────

def enhance_image_path(path: str) -> str:
    """
    Run InstructIR on *path* in a subprocess when ENABLE_INSTRUCTIR_ENHANCE is
    true; saves result back to the same path (atomic replace).

    The subprocess exits after processing, so its CUDA context is fully freed
    before the next model (e.g. Wan2.1) is loaded.

    Returns *path* on skip or error.
    """
    import config as app_config

    if not app_config.ENABLE_INSTRUCTIR_ENHANCE:
        return path

    ir_dir = Path(app_config.INSTRUCTIR_DIR).expanduser().resolve()

    if not ir_dir.is_dir():
        logger.warning("InstructIR: INSTRUCTIR_DIR is not a directory: %s", ir_dir)
        return path

    cfg_path = ir_dir / CONFIG_REL
    if not cfg_path.is_file():
        logger.warning("InstructIR: missing config %s", cfg_path)
        return path

    try:
        _ensure_weights(ir_dir)
    except Exception as exc:
        logger.warning("InstructIR: could not obtain weights (%s). Enhancement disabled.", exc)
        return path

    prompt = (app_config.INSTRUCTIR_PROMPT or "").strip()
    if not prompt:
        logger.warning("InstructIR: INSTRUCTIR_PROMPT is empty; skipping %s", path)
        return path

    p = Path(path)
    out_path = str(p.parent / (p.stem + "_InstructIR_Improved" + p.suffix))

    logger.info("InstructIR: enhancing %s (subprocess) …", os.path.basename(path))
    try:
        result = subprocess.run(
            [sys.executable, "-c", _ENHANCE_CODE, path, str(ir_dir), prompt, out_path],
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning("InstructIR: subprocess launch failed (%s).", exc)
        return path

    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-600:]
        logger.warning("InstructIR: subprocess exited %d:\n%s", result.returncode, tail)
        return path

    logger.info(
        "InstructIR: %s → %s — VRAM fully released.",
        os.path.basename(path),
        os.path.basename(out_path),
    )
    return out_path
