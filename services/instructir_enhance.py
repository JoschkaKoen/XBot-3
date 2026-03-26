"""
Optional InstructIR post-enhancement for Z-Image-Turbo outputs.

Mirrors the official InstructIR ``app.py`` forward pass (PIL → tensor → model → PIL).
Requires a local clone of https://github.com/mv-lab/InstructIR and weights
(``im_instructir-7d.pt``, ``lm_instructir-7d.pt`` in the repo root, or auto-download
via huggingface_hub when missing).

Resolution: output is forced to match the input image size (resize with LANCZOS only
if the model returns a different spatial size).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

logger = logging.getLogger("german_bot.instructir_enhance")

_lock = threading.Lock()
_pipeline: dict | None = None
_load_failed = False
# Resolved INSTRUCTIR_DIR the loaded weights belong to (str(ir_dir)).
_pipeline_root: str | None = None
# After a hard failure (import, corrupt checkpoint, …), only short-circuit if the
# same directory is still configured — changing INSTRUCTIR_DIR clears the latch.
_failed_for_dir: str | None = None

LM_MODEL = "lm_instructir-7d.pt"
MODEL_NAME = "im_instructir-7d.pt"
CONFIG_REL = Path("configs") / "eval5d.yml"


def _dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = _dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


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


def _torch_load_checkpoint(path: Path):
    """Load a ``.pt`` checkpoint; prefer ``weights_only=True`` when PyTorch supports it."""
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 2.0
        return torch.load(p, map_location="cpu")
    except Exception:
        # Older checkpoint layout or non-tensor entries
        return torch.load(p, map_location="cpu", weights_only=False)


def _load_pipeline() -> dict | None:
    """Load InstructIR once. Caller must hold _lock. Returns None on failure."""
    global _pipeline, _load_failed, _pipeline_root, _failed_for_dir

    import config as app_config

    ir_dir = Path(app_config.INSTRUCTIR_DIR).expanduser().resolve()
    key = str(ir_dir)

    if _pipeline is not None and _pipeline_root == key:
        return _pipeline
    if _pipeline is not None and _pipeline_root != key:
        _pipeline = None

    if _load_failed and _failed_for_dir == key:
        return None
    if _load_failed and _failed_for_dir != key:
        _load_failed = False

    if not ir_dir.is_dir():
        logger.warning("InstructIR: INSTRUCTIR_DIR is not a directory: %s", ir_dir)
        return None

    cfg_path = ir_dir / CONFIG_REL
    if not cfg_path.is_file():
        logger.warning("InstructIR: missing config %s", cfg_path)
        return None

    try:
        _ensure_weights(ir_dir)
    except Exception as exc:
        logger.warning("InstructIR: could not obtain weights (%s). Enhancement disabled.", exc)
        _load_failed = True
        _failed_for_dir = key
        return None

    root_str = str(ir_dir)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from models import instructir
        from text.models import LanguageModel, LMHead
    except ImportError as exc:
        logger.warning(
            "InstructIR: failed to import repo modules from %s (%s). "
            "Clone https://github.com/mv-lab/InstructIR and set INSTRUCTIR_DIR.",
            ir_dir,
            exc,
        )
        _load_failed = True
        _failed_for_dir = key
        return None

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"expected YAML mapping in {cfg_path}, got {type(raw_cfg).__name__}")
        cfg = _dict2namespace(raw_cfg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = instructir.create_model(
            input_channels=cfg.model.in_ch,
            width=cfg.model.width,
            enc_blks=cfg.model.enc_blks,
            middle_blk_num=cfg.model.middle_blk_num,
            dec_blks=cfg.model.dec_blks,
            txtdim=cfg.model.textdim,
        )
        model = model.to(device)
        im_ckpt = ir_dir / MODEL_NAME
        model.load_state_dict(_torch_load_checkpoint(im_ckpt), strict=True)
        model.eval()

        lmodel_name = cfg.llm.model
        language_model = LanguageModel(model=lmodel_name)
        lm_head = LMHead(
            embedding_dim=cfg.llm.model_dim,
            hidden_dim=cfg.llm.embd_dim,
            num_classes=cfg.llm.nclasses,
        )
        lm_head = lm_head.to(device)
        lm_ckpt = ir_dir / LM_MODEL
        lm_head.load_state_dict(_torch_load_checkpoint(lm_ckpt), strict=True)
        lm_head.eval()
    except Exception as exc:
        logger.warning("InstructIR: failed to build pipeline (%s). Enhancement disabled.", exc)
        _load_failed = True
        _failed_for_dir = key
        return None

    _pipeline = {
        "device": device,
        "model": model,
        "language_model": language_model,
        "lm_head": lm_head,
    }
    _pipeline_root = key
    _failed_for_dir = None
    _load_failed = False
    logger.info("InstructIR: pipeline ready (device=%s).", device)
    return _pipeline


def _process_pil(
    image: Image.Image,
    prompt: str,
    *,
    device: torch.device,
    model,
    language_model,
    lm_head,
) -> Image.Image:
    img = np.array(image.convert("RGB"))
    img = img / 255.0
    img = img.astype(np.float32)
    y = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    lm_embd = language_model(prompt)
    lm_embd = lm_embd.to(device)

    with torch.no_grad():
        text_embd, _deg_pred = lm_head(lm_embd)
        x_hat = model(y, text_embd)

    restored = x_hat.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().detach().numpy()
    restored = np.clip(restored, 0.0, 1.0)
    restored = (restored * 255.0).round().astype(np.uint8)
    return Image.fromarray(restored)


def enhance_image_path(path: str) -> str:
    """
    Run InstructIR on *path* when ENABLE_INSTRUCTIR_ENHANCE is true; save result
    back to the same path (atomic replace). Returns *path* on skip or error.
    """
    import config as app_config

    if not app_config.ENABLE_INSTRUCTIR_ENHANCE:
        return path

    original_size: tuple[int, int]
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            original_size = im.size
            pil_in = im.copy()
    except Exception as exc:
        logger.warning("InstructIR: could not open %s (%s).", path, exc)
        return path

    prompt = (app_config.INSTRUCTIR_PROMPT or "").strip()
    if not prompt:
        logger.warning("InstructIR: INSTRUCTIR_PROMPT is empty; skipping %s", path)
        return path

    with _lock:
        pipe = _load_pipeline()
        if pipe is None:
            return path
        try:
            restored = _process_pil(
                pil_in,
                prompt,
                device=pipe["device"],
                model=pipe["model"],
                language_model=pipe["language_model"],
                lm_head=pipe["lm_head"],
            )
        except Exception as exc:
            logger.warning("InstructIR: inference failed for %s (%s).", path, exc)
            return path

    if restored.size != original_size:
        logger.debug(
            "InstructIR: resizing output %s → %s to match input resolution.",
            restored.size,
            original_size,
        )
        restored = restored.resize(original_size, Image.Resampling.LANCZOS)

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".png", dir=out_dir)
        os.close(fd)
        restored.save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("InstructIR: could not save %s (%s).", path, exc)
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return path

    logger.info("InstructIR: enhanced %s (%dx%d).", os.path.basename(path), original_size[0], original_size[1])
    return path
