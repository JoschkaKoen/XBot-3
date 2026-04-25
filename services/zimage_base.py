"""
ZImageBaseClient — Z-Image (base/foundation model) text-to-image via diffusers.

Used by nodes/generate_image.py when IMAGE_PROVIDER=z-image-base.

The base model (Tongyi-MAI/Z-Image) is the undistilled 6B-parameter DiT that
Z-Image-Turbo was distilled from.  It runs 20–50 denoising steps, supports
negative prompts and full CFG, and produces noticeably richer detail, better
lighting, and stronger prompt adherence than the 8-step turbo variant.

Each generation batch runs in a **subprocess** so the CUDA context + model
weights are fully released from GPU memory before WAN2.1 loads — exactly the
same isolation pattern used by InstructIR.

Prerequisites (one-time):
  pip install diffusers transformers accelerate

The model is downloaded automatically from Hugging Face on first use
(~12 GB to ~/.cache/huggingface/).  Pre-download with:
  huggingface-cli download Tongyi-MAI/Z-Image
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import config

logger = logging.getLogger("xbot.zimage_base")

# Overlay venv that contains only transformers 5.x.
# The subprocess still uses the shared venv's Python (torch / diffusers live
# there), but we prepend this path so the newer transformers wins.
_OVERLAY_SITE = str(
    Path(__file__).parent.parent / "venv_zimage" / "lib" / "python3.12" / "site-packages"
)


# ── Subprocess generation script ──────────────────────────────────────────────
# Executed as `python -c _GENERATE_CODE <args_json_path>`.
# The model is loaded *once* for the whole batch, then the process exits and
# VRAM is fully released.
_GENERATE_CODE = r"""
import sys, json, os, tempfile
import torch
from diffusers import ZImagePipeline

args_file = sys.argv[1]
with open(args_file, encoding="utf-8") as fh:
    args = json.load(fh)

model_id      = args["model_id"]
steps         = args["steps"]
guidance      = args["guidance_scale"]
width         = args["width"]
height        = args["height"]
neg_prompt    = args.get("negative_prompt", "") or ""
items         = args["items"]   # list of {prompt, seed, output_path}

print(f"  ⏳  Loading {model_id} …", flush=True)
pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# Model (~19 GB) is larger than VRAM: keep on CPU, move each component to GPU
# on demand, then move it back. Requires accelerate>=0.18.
pipe.enable_model_cpu_offload()
print("  ✔   Model loaded (CPU offload active).", flush=True)

for idx, item in enumerate(items, start=1):
    prompt      = item["prompt"]
    seed        = item["seed"]
    output_path = item["output_path"]

    # CPU generator is required when using enable_model_cpu_offload()
    generator = torch.Generator("cpu").manual_seed(seed) if seed >= 0 else None

    print(f"  ⏳  Generating image {idx}/{len(items)} (seed {seed}) …", flush=True)
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt if neg_prompt else None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".png", dir=out_dir)
    os.close(fd)
    image.save(tmp_path, format="PNG")
    os.replace(tmp_path, output_path)
    print(f"generated:{output_path}", flush=True)
"""


class ZImageBaseClient:
    """
    Text-to-image client for Z-Image Base via diffusers (local GPU).

    Instantiated by nodes/generate_image.py when IMAGE_PROVIDER=z-image-base.
    Call generate_batch(prompts, seeds) → returns a list of image paths, one
    per prompt.  The model is loaded once per call and fully unloaded on return.
    """

    def __init__(self) -> None:
        self._images_dir = Path(config.IMAGES_DIR)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ZImageBaseClient ready — model: %s  steps: %d  %dx%d  cfg: %.1f",
            config.Z_IMAGE_BASE_MODEL_ID,
            config.Z_IMAGE_BASE_STEPS,
            config.Z_IMAGE_BASE_WIDTH,
            config.Z_IMAGE_BASE_HEIGHT,
            config.Z_IMAGE_BASE_GUIDANCE_SCALE,
        )

    def generate_batch(self, prompts: List[str], seeds: List[int]) -> List[str]:
        """
        Generate one image per prompt in a single subprocess call.

        The subprocess loads the model once, generates all images sequentially,
        then exits — fully releasing VRAM before WAN2.1 starts.

        Returns a list of saved PNG paths (same length and order as *prompts*).
        """
        if not prompts:
            return []

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_paths = [
            str(self._images_dir / f"zib_{ts}_{i}_seed{s}.png")
            for i, s in enumerate(seeds)
        ]

        args = {
            "model_id":       config.Z_IMAGE_BASE_MODEL_ID,
            "steps":          config.Z_IMAGE_BASE_STEPS,
            "guidance_scale": config.Z_IMAGE_BASE_GUIDANCE_SCALE,
            "width":          config.Z_IMAGE_BASE_WIDTH,
            "height":         config.Z_IMAGE_BASE_HEIGHT,
            "negative_prompt": config.Z_IMAGE_BASE_NEGATIVE_PROMPT,
            "items": [
                {"prompt": p, "seed": s, "output_path": op}
                for p, s, op in zip(prompts, seeds, output_paths)
            ],
        }

        # Write args to a temp file so very long prompts don't hit argv limits.
        fd, args_path = tempfile.mkstemp(suffix=".json")
        try:
            os.close(fd)
            with open(args_path, "w", encoding="utf-8") as fh:
                json.dump(args, fh, ensure_ascii=False)

            cmd = [sys.executable, "-c", _GENERATE_CODE, args_path]
            logger.info("ZImageBase subprocess starting — %d image(s)", len(prompts))

            # Prepend overlay venv so transformers 5.x takes precedence.
            env = os.environ.copy()
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = _OVERLAY_SITE + (":" + existing_pp if existing_pp else "")

            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=False,   # let stdout/stderr pass through to the terminal
                text=True,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"ZImageBase subprocess exited with code {proc.returncode}."
                )
        finally:
            try:
                os.unlink(args_path)
            except OSError:
                pass

        # Verify all outputs were written.
        missing = [p for p in output_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"ZImageBase: subprocess finished but {len(missing)} output(s) missing: "
                + ", ".join(os.path.basename(p) for p in missing)
            )

        for p in output_paths:
            sz = os.path.getsize(p) / 1024
            logger.info("ZImageBase saved: %s (%.0f KB)", os.path.basename(p), sz)

        return output_paths
