#!/usr/bin/env python3
"""
test_comfyUI_ZIT.py — Z-Image-Turbo (FP8 AIO) text-to-image via ComfyUI HTTP API
================================================================================

Generates a 1024×1024 image from a text prompt using the Z-Image-Turbo FP8 AIO
model running inside ComfyUI.  ComfyUI handles all model loading and VRAM
management; this script is a thin HTTP client that patches the workflow JSON,
submits the job, polls for completion, copies the PNG, scores it, and logs
results.

PREREQUISITES (one-time):
  1. ComfyUI installed at ~/ComfyUI.
  2. Model and workflow downloaded:
       huggingface-cli download SeeSee21/Z-Image-Turbo-AIO \\
         z-image-turbo-fp8-aio.safetensors \\
         --local-dir ~/ComfyUI/models/checkpoints/
       huggingface-cli download SeeSee21/Z-Image-Turbo-AIO \\
         workflows/ZIT-AIO-v1.0.json \\
         --local-dir ~/ComfyUI/workflows/
  3. COMFYUI_DIR and COMFYUI_URL set in settings.env (already done).

NOTE: Use --fp16-vae when launching ComfyUI; do NOT load the BF16 checkpoint
variant — it produces black images on 8 GB VRAM cards.

USAGE (no changes needed — run from XBot 3 project root):
  # 1. Start ComfyUI in a separate terminal (leave it running):
  #    cd ~/ComfyUI && source venv/bin/activate && python main.py --normalvram --fp16-vae

  # 2. Run this script with any Python (torch not required in this venv):
  python test_comfyUI_ZIT.py "a red fox in a snowy forest"
  python test_comfyUI_ZIT.py "sunset over the ocean" --steps 9
  python test_comfyUI_ZIT.py "abstract geometric art" --seed 42

Settings are read automatically from settings.env — no manual exports needed.
================================================================================
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Load settings.env (stdlib only — no python-dotenv needed) ─────────────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_settings_env = _PROJECT_DIR / "settings.env"
if _settings_env.exists():
    with open(_settings_env, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Constants ─────────────────────────────────────────────────────────────────
COMFYUI_URL  = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_DIR  = Path(os.getenv("COMFYUI_DIR", str(Path.home() / "ComfyUI")))
OUTPUTS_DIR  = _PROJECT_DIR / "Images"
HISTORY_FILE = _PROJECT_DIR / "data" / "zit_image_history.jsonl"
ENGINE_TAG   = "ComfyUI-ZIT-FP8"
BANNER       = "Z-Image-Turbo FP8 AIO  ·  ComfyUI  ·  1024×1024"

# Workflow shipped with the model (downloaded by huggingface-cli)
_COMFYUI_WORKFLOW = COMFYUI_DIR / "workflows" / "ZIT-AIO-v1.0.json"
# Fallback: project workflows folder (copy it here if preferred)
_PROJECT_WORKFLOW = _PROJECT_DIR / "workflows" / "ZIT-AIO-v1.0.json"

# Locked generation settings per model card (do NOT change these)
_FIXED_CFG       = 1.0
_FIXED_SAMPLER   = "res_multistep"
_FIXED_SCHEDULER = "simple"
_FIXED_BATCH     = 1
_STEPS_MIN       = 8
_STEPS_MAX       = 9

# ── Inline image scorer (runs in ComfyUI venv which has torch + open_clip) ────
_SCORE_CODE = """\
import sys, json, torch, open_clip
from PIL import Image

def _score(image_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer([prompt]).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img)
        txt_feat = model.encode_text(text)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    score = float((img_feat @ txt_feat.T).squeeze().cpu())
    print(json.dumps({"clip_score": round(score, 4)}))

_score(sys.argv[1], sys.argv[2])
"""


# ── ComfyUI helpers ───────────────────────────────────────────────────────────

def find_workflow() -> Path:
    """Return the workflow file path, checking both locations."""
    if _COMFYUI_WORKFLOW.exists():
        return _COMFYUI_WORKFLOW
    if _PROJECT_WORKFLOW.exists():
        return _PROJECT_WORKFLOW
    print(
        f"\n  ERROR: workflow file not found.\n\n"
        "  Download it with:\n"
        "    huggingface-cli download SeeSee21/Z-Image-Turbo-AIO \\\n"
        "      workflows/ZIT-AIO-v1.0.json \\\n"
        f"      --local-dir {COMFYUI_DIR}/workflows/\n\n"
        f"  Expected path: {_COMFYUI_WORKFLOW}\n",
        file=sys.stderr,
    )
    sys.exit(1)


def check_server() -> None:
    """Abort with a clear message if ComfyUI is not reachable."""
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
    except Exception:
        print(
            f"\n  ERROR: ComfyUI server not reachable at {COMFYUI_URL}\n\n"
            "  Start it first (leave running in a separate terminal):\n"
            f"    cd {COMFYUI_DIR} && source venv/bin/activate\n"
            "    python main.py --normalvram --fp16-vae\n",
            file=sys.stderr,
        )
        sys.exit(1)


def gui_to_api(gui_wf: dict) -> dict:
    """
    Convert a ComfyUI browser/GUI-format workflow (has a 'nodes' list) to the
    flat API format accepted by the /prompt endpoint.

    Handles:
    - Reroute nodes (resolved transparently via link-following)
    - LoraLoader nodes whose LoRA file is missing (bypassed automatically)
    - Display-only custom nodes (MarkdownNote, rgthree UI nodes) — skipped
    - widget_values ordering via /object_info (including control_after_generate)
    """
    if "nodes" not in gui_wf:
        return gui_wf  # already API format

    nodes = gui_wf["nodes"]
    links = gui_wf.get("links", [])

    # link_id → (from_node_id_str, output_index)
    link_map: dict[int, tuple[str, int]] = {
        lnk[0]: (str(lnk[1]), lnk[2]) for lnk in links
    }

    # node_id_str → GUI node dict
    node_by_id: dict[str, dict] = {str(n["id"]): n for n in nodes}

    # Fetch /object_info for every unique node type in one pass
    all_types = set(n["type"] for n in nodes)
    node_info: dict[str, dict] = {}
    for ntype in all_types:
        try:
            url = f"{COMFYUI_URL}/object_info/{urllib.parse.quote(ntype, safe='')}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            if ntype in data:
                node_info[ntype] = data[ntype]
        except Exception:
            pass

    # Nodes with no functional output — skip them entirely
    _DISPLAY_TYPES = {
        "Note", "MarkdownNote",
        "Image Comparer (rgthree)", "Fast Groups Bypasser (rgthree)",
    }

    def _lora_file_exists(lora_name: str) -> bool:
        normalized = lora_name.replace("\\", os.sep).replace("/", os.sep)
        return (COMFYUI_DIR / "models" / "loras" / normalized).exists()

    def _should_bypass_lora(node: dict) -> bool:
        if node["type"] != "LoraLoader":
            return False
        wvals = node.get("widgets_values", [])
        lora_name = wvals[0] if wvals else ""
        return not _lora_file_exists(str(lora_name))

    def resolve_link(link_id: int, _seen: frozenset = frozenset()) -> tuple[str, int]:
        """Follow a link through Reroutes and bypassed LoraLoaders."""
        if link_id not in link_map:
            raise KeyError(f"Unknown link id {link_id}")
        if link_id in _seen:
            raise RuntimeError(f"Circular link detected at link {link_id}")
        _seen = _seen | {link_id}

        src_id, src_out = link_map[link_id]
        src_node = node_by_id.get(src_id)
        if src_node is None:
            return src_id, src_out

        if src_node["type"] == "Reroute":
            reroute_links = [i.get("link") for i in src_node.get("inputs", [])]
            if reroute_links and reroute_links[0] is not None:
                return resolve_link(reroute_links[0], _seen)

        if _should_bypass_lora(src_node):
            # LoraLoader output 0 = MODEL (pass model input through)
            # LoraLoader output 1 = CLIP  (pass clip  input through)
            lora_input_links = {
                i["name"]: i.get("link")
                for i in src_node.get("inputs", [])
                if i.get("link") is not None
            }
            passthrough = "model" if src_out == 0 else "clip"
            next_link = lora_input_links.get(passthrough)
            if next_link is not None:
                return resolve_link(next_link, _seen)

        return src_id, src_out

    def _is_connection(type_info) -> bool:
        """Return True if this schema input type is a node connection (not a widget)."""
        if isinstance(type_info, list):
            return False  # dropdown list → widget
        if not isinstance(type_info, str):
            return False
        # ComfyUI convention: connection types are ALL_CAPS non-primitive strings
        return type_info not in ("INT", "FLOAT", "STRING", "BOOLEAN", "IMAGEUPLOAD") \
            and type_info == type_info.upper()

    api_wf: dict[str, dict] = {}

    for node in nodes:
        nid = str(node["id"])
        ntype = node["type"]

        if ntype in _DISPLAY_TYPES or ntype == "Reroute":
            continue
        if _should_bypass_lora(node):
            print(f"  [gui→api] bypassing LoraLoader {nid} "
                  f"(LoRA '{node.get('widgets_values', ['?'])[0]}' not found)")
            continue

        schema = node_info.get(ntype)
        if schema is None:
            print(f"  [gui→api] skipping unknown node type '{ntype}' (node {nid})")
            continue

        required_schema = schema.get("input", {}).get("required", {})
        optional_schema = schema.get("input", {}).get("optional", {}) or {}
        all_schema: dict = {**required_schema, **optional_schema}

        input_order = (
            schema.get("input_order", {}).get("required", [])
            + schema.get("input_order", {}).get("optional", [])
        )

        # name → link_id for this node's connected slots
        linked: dict[str, int] = {
            i["name"]: i["link"]
            for i in node.get("inputs", [])
            if i.get("link") is not None and i.get("name")
        }

        widget_vals = list(node.get("widgets_values", []))
        w_idx = 0
        inputs: dict = {}

        for inp_name in input_order:
            type_def = all_schema.get(inp_name)
            if type_def is None:
                continue
            inp_type = type_def[0]
            inp_opts = type_def[1] if len(type_def) > 1 else {}

            if inp_name in linked:
                src_id, src_out = resolve_link(linked[inp_name])
                inputs[inp_name] = [src_id, src_out]
            elif not _is_connection(inp_type):
                if w_idx < len(widget_vals):
                    inputs[inp_name] = widget_vals[w_idx]
                    w_idx += 1
                    # control_after_generate adds a GUI-only widget value — skip it
                    if isinstance(inp_opts, dict) and inp_opts.get("control_after_generate"):
                        w_idx += 1

        api_wf[nid] = {"class_type": ntype, "inputs": inputs}

    return api_wf


def patch_workflow(
    workflow: dict,
    prompt: str,
    steps: int,
    seed: int,
    width: int,
    height: int,
) -> dict:
    """
    Patch a ComfyUI API-format workflow dict with runtime values.

    Scans every node by class_type and updates the relevant inputs.
    CFG, sampler, and scheduler are locked to model-card values and always
    enforced regardless of what the source workflow contains.
    ConditioningZeroOut nodes are intentionally left untouched — the ZIT
    workflow uses that node for the negative conditioning instead of text.
    """
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})

        # Positive text prompt
        if ct in ("CLIPTextEncode", "CLIPTextEncodeFlux", "CLIPTextEncodeSD3",
                  "CLIPTextEncodeWan2", "TextEncode"):
            if "text" in inp:
                inp["text"] = prompt

        # Sampler — lock all generation settings to model-card values
        if ct in ("KSampler", "KSamplerAdvanced", "SamplerCustom",
                  "SamplerCustomAdvanced"):
            if "steps" in inp:
                inp["steps"] = steps
            if "cfg" in inp:
                inp["cfg"] = _FIXED_CFG
            if "sampler_name" in inp:
                inp["sampler_name"] = _FIXED_SAMPLER
            if "scheduler" in inp:
                inp["scheduler"] = _FIXED_SCHEDULER
            if seed >= 0:
                for seed_key in ("seed", "noise_seed"):
                    if seed_key in inp:
                        inp[seed_key] = seed

        # Resolution / latent size
        if ct in ("EmptyLatentImage", "EmptySD3LatentImage"):
            if "width" in inp:
                inp["width"] = width
            if "height" in inp:
                inp["height"] = height
            if "batch_size" in inp:
                inp["batch_size"] = _FIXED_BATCH

    return wf


def submit_prompt(workflow: dict) -> str:
    """POST workflow to /prompt, return prompt_id."""
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def poll_until_done(prompt_id: str, poll_interval: int = 5) -> dict:
    """
    Poll /history/{prompt_id} until the job completes or errors.
    Returns the history entry dict.
    """
    dots = 0
    while True:
        time.sleep(poll_interval)
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
            history = json.loads(resp.read())

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print()  # newline after dots
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI reported an error for prompt {prompt_id}: {status}"
                )

        dots += 1
        print(f"\r  ... generating {'.' * (dots % 4):<4}", end="", flush=True)


def find_output_image(history_entry: dict) -> Path | None:
    """
    Locate the generated image file from a completed history entry.
    ComfyUI stores output files under COMFYUI_DIR/output/.
    """
    for node_output in history_entry.get("outputs", {}).values():
        for item in node_output.get("images", []):
            if not isinstance(item, dict):
                continue
            filename = item.get("filename", "")
            subfolder = item.get("subfolder", "")
            if not filename:
                continue
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            candidate = COMFYUI_DIR / "output" / subfolder / filename
            if candidate.exists():
                return candidate
    return None


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_image(image_path: str, prompt: str) -> tuple[dict | None, str | None]:
    """
    Run the inline CLIP scorer in ComfyUI's venv (which has torch + open_clip).
    Returns (score_dict, error_str) — exactly one will be None.
    """
    comfy_python = COMFYUI_DIR / "venv" / "bin" / "python"
    if not comfy_python.exists():
        return None, f"ComfyUI venv not found at {comfy_python}"

    try:
        proc = subprocess.run(
            [str(comfy_python), "-c", _SCORE_CODE, image_path, prompt],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip()), None
        error = proc.stderr.strip() or f"exit code {proc.returncode}"
        return None, error
    except Exception as exc:
        return None, str(exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo FP8 AIO text-to-image via ComfyUI — run from XBot 3 project root",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_comfyUI_ZIT.py \"a red fox in a snowy forest\"\n"
            "  python test_comfyUI_ZIT.py \"sunset over the ocean\" --steps 9\n"
            "  python test_comfyUI_ZIT.py \"abstract geometric art\" --seed 42\n\n"
            "Locked settings (model card): CFG=1.0  sampler=res_multistep  scheduler=simple\n"
            "Close all other GPU apps before running — text encoder alone nears 8 GB VRAM limit.\n"
        ),
    )
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument(
        "--steps", type=int, default=8,
        help=f"Denoising steps (default: 8, recommended range: {_STEPS_MIN}–{_STEPS_MAX})",
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="RNG seed (-1 = random, default: -1)",
    )
    parser.add_argument("--width",  type=int, default=1024, help="Image width  (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    args = parser.parse_args()

    if not (_STEPS_MIN <= args.steps <= _STEPS_MAX):
        print(
            f"  WARNING: --steps {args.steps} is outside the recommended range "
            f"{_STEPS_MIN}–{_STEPS_MAX} for Z-Image-Turbo.",
            file=sys.stderr,
        )

    workflow_file = find_workflow()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    print(f"\n{'═' * 62}")
    print(f"  {BANNER}")
    print(f"{'═' * 62}\n")
    print(f"  Prompt  : {args.prompt[:80]}")
    print(f"  Steps   : {args.steps}   Seed: {args.seed if args.seed >= 0 else '(random)'}")
    print(f"  Size    : {args.width}×{args.height}")
    print(f"  CFG     : {_FIXED_CFG}  Sampler: {_FIXED_SAMPLER}  Scheduler: {_FIXED_SCHEDULER}")
    print(f"  Workflow: {workflow_file}")
    print(f"  Server  : {COMFYUI_URL}")
    print()

    # 1. Verify server
    check_server()
    print("  ComfyUI server: OK")

    # 2. Load workflow — convert GUI format to API format if needed
    with open(workflow_file, encoding="utf-8") as fh:
        workflow = json.load(fh)
    if "nodes" in workflow:
        print("  Converting GUI-format workflow to API format ...")
        workflow = gui_to_api(workflow)
        print(f"  Converted {len(workflow)} nodes.")
    workflow = patch_workflow(
        workflow, args.prompt, args.steps, args.seed, args.width, args.height
    )

    # 3. Submit
    print("  Submitting workflow to ComfyUI ...")
    prompt_id = submit_prompt(workflow)
    print(f"  prompt_id: {prompt_id}")
    print("  ", end="", flush=True)

    # 4. Poll
    history_entry = poll_until_done(prompt_id)
    finished_at = datetime.now().isoformat(timespec="seconds")

    # 5. Retrieve output
    comfy_image = find_output_image(history_entry)
    if comfy_image is None:
        raise RuntimeError(
            "Generation completed but no image found in ComfyUI output.\n"
            f"Check ComfyUI output folder: {COMFYUI_DIR / 'output'}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = OUTPUTS_DIR / f"zit_{ts}{comfy_image.suffix}"
    shutil.copy2(comfy_image, dest)
    image_path = str(dest.resolve())
    size_kb = dest.stat().st_size / 1024
    print(f"\n  Image ready  →  {dest.name}  ({size_kb:.0f} KB)")

    # 6. Score
    print("  Scoring image ...")
    image_score, score_error = score_image(image_path, args.prompt)
    if image_score:
        print(f"  CLIP score: {image_score['clip_score']:.4f}")
    else:
        print(f"  (Scoring skipped: {score_error})")

    # 7. History
    record = {
        "engine":            ENGINE_TAG,
        "started_at":        started_at,
        "finished_at":       finished_at,
        "prompt":            args.prompt,
        "cli": {
            "steps":         args.steps,
            "seed":          args.seed,
            "width":         args.width,
            "height":        args.height,
        },
        "locked": {
            "cfg":           _FIXED_CFG,
            "sampler":       _FIXED_SAMPLER,
            "scheduler":     _FIXED_SCHEDULER,
        },
        "comfyui_url":       COMFYUI_URL,
        "workflow_file":     str(workflow_file),
        "comfyui_prompt_id": prompt_id,
        "output_image_path": image_path,
        "image_score":       image_score,
        "image_score_error": score_error,
    }
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  History     →  {HISTORY_FILE}")
    print()


if __name__ == "__main__":
    main()
