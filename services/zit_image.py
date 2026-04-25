"""
ZITImageClient — Z-Image-Turbo (FP8 AIO) text-to-image via ComfyUI HTTP API.

Used by nodes/generate_image.py when IMAGE_PROVIDER=z-image-turbo.

Prerequisites (one-time setup):
  1. ComfyUI running: cd ~/ComfyUI && source venv/bin/activate && python main.py --normalvram --fp16-vae
  2. Model downloaded:
       huggingface-cli download SeeSee21/Z-Image-Turbo-AIO z-image-turbo-fp8-aio.safetensors \\
         --local-dir ~/ComfyUI/models/checkpoints/
  3. Workflow downloaded:
       huggingface-cli download SeeSee21/Z-Image-Turbo-AIO workflows/ZIT-AIO-v1.0.json \\
         --local-dir ~/ComfyUI/workflows/
  4. COMFYUI_DIR and COMFYUI_URL set in settings.env.

Locked settings (per model card — do NOT change):
  CFG=1.0  sampler=res_multistep  scheduler=simple  batch_size=1
"""

import copy
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List

import config
from utils.ui import err, info, ok, warn

logger = logging.getLogger("xbot.zit_image")

# Clear to end of line after \r so shorter updates don't leave stale characters.
_CLR_EOL = "\033[K"


class ComfyUIUnavailableError(RuntimeError):
    """Raised when ComfyUI cannot be reached after auto-start and the 60-second grace period."""


# Popen handle for a ComfyUI we spawned ourselves this session.
# None if ComfyUI was already running when the bot started.
_comfyui_proc: subprocess.Popen | None = None

# Records the COMFYUI_ARGS string last used when we spawned ComfyUI so we can
# restart if settings.env changes (see ensure_comfyui_running).
_LAUNCH_ARGS_FILE = Path(__file__).resolve().parent.parent / "data" / ".comfyui_launch_args"


def _normalized_comfy_args() -> str:
    return " ".join(config.COMFYUI_ARGS.split())


def _read_recorded_comfy_args() -> str | None:
    try:
        t = _LAUNCH_ARGS_FILE.read_text(encoding="utf-8").strip()
        return t if t else None
    except OSError:
        return None


def _write_recorded_comfy_args(s: str) -> None:
    _LAUNCH_ARGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAUNCH_ARGS_FILE.write_text(s, encoding="utf-8")


def _comfy_url_reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(f"{url}/system_stats", timeout=5)
        return True
    except (urllib.error.URLError, OSError) as exc:
        logger.debug("ComfyUI not reachable at %s: %s", url, exc)
        return False


def _wait_until_comfy_down(u: str, max_wait: float = 35.0) -> None:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if not _comfy_url_reachable(u):
            return
        time.sleep(0.5)
    logger.warning("ComfyUI still reachable at %s after shutdown (%.1fs).", u, max_wait)


def _find_comfyui_pid_by_port(port: int) -> int | None:
    """Return the PID listening on *port* using `ss` (standard on Ubuntu)."""
    import re
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True,
        )
        m = re.search(r"pid=(\d+)", result.stdout)
        if m:
            return int(m.group(1))
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("ss lookup for ComfyUI PID failed: %s", exc)
    return None


def shutdown_comfyui() -> None:
    """
    Terminate ComfyUI to fully release its VRAM (CUDA context + model cache).

    Tries the Popen handle we hold from auto-starting it first; falls back to
    finding the process by port via `ss`.  After termination the next call to
    ensure_comfyui_running() will restart it automatically.
    """
    global _comfyui_proc

    import signal as _signal
    from urllib.parse import urlparse

    pid: int | None = None

    if _comfyui_proc is not None and _comfyui_proc.poll() is None:
        pid = _comfyui_proc.pid
        _comfyui_proc = None

    if pid is None:
        parsed = urlparse(config.COMFYUI_URL)
        port   = parsed.port or 8188
        pid    = _find_comfyui_pid_by_port(port)

    if pid is None:
        logger.warning("shutdown_comfyui: could not find ComfyUI process — skipping.")
        return

    try:
        os.kill(pid, _signal.SIGTERM)
        # Wait up to 8 s for clean exit, then SIGKILL
        for _ in range(16):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)   # still alive?
            except ProcessLookupError:
                break
        else:
            os.kill(pid, _signal.SIGKILL)
        ok("ComfyUI shut down — VRAM released.")
        logger.info("ComfyUI (PID %d) terminated.", pid)
    except ProcessLookupError:
        logger.debug("ComfyUI PID %d already gone.", pid)
    except Exception as exc:
        logger.warning("shutdown_comfyui: failed to kill PID %d: %s", pid, exc)


def ensure_comfyui_running() -> None:
    """
    Non-blocking check-and-spawn called at the top of each bot cycle.

    If ComfyUI is already responding at COMFYUI_URL with the same
    ``COMFYUI_ARGS`` as last time (see ``data/.comfyui_launch_args``), returns
    immediately.  If ``settings.env`` changed ``COMFYUI_ARGS``, shuts down the
    old process and spawns a new one so flags like ``--lowvram`` take effect.

    If nothing is listening, spawns ComfyUI in a detached background process.
    ZITImageClient.generate() waits up to 60 seconds for readiness.
    """
    url = config.COMFYUI_URL.rstrip("/")
    current_args = _normalized_comfy_args()
    recorded = _read_recorded_comfy_args()

    if _comfy_url_reachable(url):
        if recorded == current_args:
            logger.debug("ensure_comfyui_running: already up at %s (args match).", url)
            return
        info("COMFYUI_ARGS changed — restarting ComfyUI for new flags …")
        logger.info(
            "ensure_comfyui_running: COMFYUI_ARGS changed (%r -> %r) — restarting ComfyUI.",
            recorded,
            current_args,
        )
        shutdown_comfyui()
        _wait_until_comfy_down(url)
        if _comfy_url_reachable(url):
            logger.warning(
                "ensure_comfyui_running: ComfyUI still up; new COMFYUI_ARGS may not apply."
            )
            return

    comfyui = Path(config.COMFYUI_DIR)
    venv_python = comfyui / "venv" / "bin" / "python"
    if not venv_python.exists():
        warn("ComfyUI venv not found — cannot auto-start.")
        info(f"Expected: {venv_python}")
        info("Check COMFYUI_DIR in settings.env.")
        logger.warning(
            "ensure_comfyui_running: venv not found at %s — cannot auto-start.", venv_python
        )
        return

    global _comfyui_proc

    if _comfy_url_reachable(url):
        logger.debug("ensure_comfyui_running: external ComfyUI already at %s.", url)
        _write_recorded_comfy_args(current_args)
        return

    extra_args = current_args.split()
    cmd = [str(venv_python), "main.py"] + extra_args
    log_path = comfyui / "comfyui_autostart.log"
    with open(log_path, "a") as log_fh:
        _comfyui_proc = subprocess.Popen(
            cmd,
            cwd=str(comfyui),
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    info("ComfyUI not running — spawning in background …")
    info(f"cmd: {' '.join(cmd)}")
    info(f"log: {log_path}")
    logger.info("ComfyUI spawned in background (cmd=%s).", " ".join(cmd))
    _write_recorded_comfy_args(current_args)

# ── Locked generation settings (model card requirements) ──────────────────────
_FIXED_CFG       = 1.0
_FIXED_SAMPLER   = "res_multistep"
_FIXED_SCHEDULER = "simple"
_FIXED_BATCH     = 1

# ── Node types that are display-only in the GUI and have no API function ───────
_DISPLAY_TYPES = {
    "Note", "MarkdownNote",
    "Image Comparer (rgthree)", "Fast Groups Bypasser (rgthree)",
}


def _find_workflow(comfyui_dir: Path, project_dir: Path) -> Path:
    """
    Return the ZIT workflow file path.
    Checks COMFYUI_DIR/workflows/ first, then the project workflows/ folder.
    Raises RuntimeError with clear download instructions if not found.
    """
    primary  = comfyui_dir / "workflows" / "ZIT-AIO-v1.0.json"
    fallback = project_dir / "workflows"  / "ZIT-AIO-v1.0.json"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise RuntimeError(
        f"ZIT workflow file not found.\n\n"
        "  Download it with:\n"
        "    huggingface-cli download SeeSee21/Z-Image-Turbo-AIO \\\n"
        "      workflows/ZIT-AIO-v1.0.json \\\n"
        f"      --local-dir {comfyui_dir}/workflows/\n\n"
        f"  Expected path: {primary}"
    )


def _gui_to_api(gui_wf: dict, comfyui_url: str, comfyui_dir: Path) -> dict:
    """
    Convert a ComfyUI browser/GUI-format workflow (has a 'nodes' list) to the
    flat API format accepted by the /prompt endpoint.

    Handles:
    - Reroute nodes — resolved transparently via link-following
    - LoraLoader nodes whose LoRA file is missing — bypassed automatically,
      passing model/clip straight from the upstream loader
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
            url = f"{comfyui_url}/object_info/{urllib.parse.quote(ntype, safe='')}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            if ntype in data:
                node_info[ntype] = data[ntype]
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not fetch /object_info for %s: %s", ntype, exc)

    def _lora_file_exists(lora_name: str) -> bool:
        normalized = lora_name.replace("\\", os.sep).replace("/", os.sep)
        return (comfyui_dir / "models" / "loras" / normalized).exists()

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
            # output 0 = MODEL, output 1 = CLIP — pass upstream inputs through
            lora_input_links = {
                i["name"]: i.get("link")
                for i in src_node.get("inputs", [])
                if i.get("link") is not None and i.get("name")
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
        return (
            type_info not in ("INT", "FLOAT", "STRING", "BOOLEAN", "IMAGEUPLOAD")
            and type_info == type_info.upper()
        )

    api_wf: dict[str, dict] = {}

    for node in nodes:
        nid   = str(node["id"])
        ntype = node["type"]

        if ntype in _DISPLAY_TYPES or ntype == "Reroute":
            continue
        if _should_bypass_lora(node):
            logger.info("ZIT gui→api: bypassing LoraLoader %s (LoRA '%s' not found)",
                        nid, node.get("widgets_values", ["?"])[0])
            continue

        schema = node_info.get(ntype)
        if schema is None:
            logger.warning("ZIT gui→api: skipping unknown node type '%s' (node %s)", ntype, nid)
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
                    # control_after_generate is a GUI-only widget — consume but discard
                    if isinstance(inp_opts, dict) and inp_opts.get("control_after_generate"):
                        w_idx += 1

        api_wf[nid] = {"class_type": ntype, "inputs": inputs}

    return api_wf


def _patch_workflow(
    workflow: dict,
    prompt: str,
    steps: int,
    seed: int,
    width: int,
    height: int,
) -> dict:
    """
    Patch an API-format workflow dict with the prompt and generation parameters.

    CFG, sampler, scheduler, and batch size are locked to model-card values.
    Width and height come from Z_IMAGE_TURBO_WIDTH / Z_IMAGE_TURBO_HEIGHT.
    ConditioningZeroOut nodes are intentionally left untouched.
    """
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct  = node.get("class_type", "")
        inp = node.get("inputs", {})

        # Positive text prompt
        if ct in ("CLIPTextEncode", "CLIPTextEncodeFlux", "CLIPTextEncodeSD3",
                  "CLIPTextEncodeWan2", "TextEncode"):
            if "text" in inp:
                inp["text"] = prompt

        # Sampler — enforce locked model-card settings
        if ct in ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"):
            if "steps"        in inp: inp["steps"]        = steps
            if "cfg"          in inp: inp["cfg"]          = _FIXED_CFG
            if "sampler_name" in inp: inp["sampler_name"] = _FIXED_SAMPLER
            if "scheduler"    in inp: inp["scheduler"]    = _FIXED_SCHEDULER
            if seed >= 0:
                for seed_key in ("seed", "noise_seed"):
                    if seed_key in inp:
                        inp[seed_key] = seed

        # Resolution / latent size
        if ct in ("EmptyLatentImage", "EmptySD3LatentImage"):
            if "width"      in inp: inp["width"]      = width
            if "height"     in inp: inp["height"]      = height
            if "batch_size" in inp: inp["batch_size"]  = _FIXED_BATCH

    return wf


def _submit_prompt(workflow: dict, comfyui_url: str) -> str:
    """POST workflow to /prompt, return prompt_id."""
    data = json.dumps({"prompt": workflow}).encode()
    req  = urllib.request.Request(
        f"{comfyui_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def _poll_until_done(prompt_id: str, comfyui_url: str, poll_interval: int = 5) -> dict:
    """Poll /history/{prompt_id} until complete. Returns the history entry."""
    dots = 0
    while True:
        time.sleep(poll_interval)
        with urllib.request.urlopen(f"{comfyui_url}/history/{prompt_id}") as resp:
            history = json.loads(resp.read())

        if prompt_id in history:
            entry  = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print()  # newline after progress dots
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI reported an error for prompt {prompt_id}: {status}"
                )

        dots += 1
        print(
            f"\r  ⏳  Generating image{'.' * (dots % 4):<4}{_CLR_EOL}",
            end="",
            flush=True,
        )


def _find_output_image(history_entry: dict, comfyui_dir: Path) -> Path | None:
    """Locate the generated PNG from a completed ComfyUI history entry."""
    for node_output in history_entry.get("outputs", {}).values():
        for item in node_output.get("images", []):
            if not isinstance(item, dict):
                continue
            filename  = item.get("filename", "")
            subfolder = item.get("subfolder", "")
            if not filename:
                continue
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            candidate = comfyui_dir / "output" / subfolder / filename
            if candidate.exists():
                return candidate
    return None


# ── Public client class ────────────────────────────────────────────────────────

class ZITImageClient:
    """
    Text-to-image client for Z-Image-Turbo via the local ComfyUI HTTP API.

    Instantiated by nodes/generate_image.py when IMAGE_PROVIDER=z-image-turbo.
    Call generate(prompt) → returns a list containing one image path.
    """

    def __init__(self) -> None:
        self._url        = config.COMFYUI_URL.rstrip("/")
        self._comfyui    = Path(config.COMFYUI_DIR)
        self._project    = Path(__file__).parent.parent.resolve()
        self._images_dir = Path(config.IMAGES_DIR)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._workflow_file = _find_workflow(self._comfyui, self._project)
        logger.info("ZITImageClient ready — workflow: %s", self._workflow_file)

    def _is_server_up(self) -> bool:
        try:
            urllib.request.urlopen(f"{self._url}/system_stats", timeout=5)
            return True
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("ComfyUI server check failed at %s: %s", self._url, exc)
            return False

    def _check_server(self) -> None:
        """
        Wait up to 60 s for ComfyUI to become ready.

        ensure_comfyui_running() is called at the start of each cycle so
        ComfyUI should already be warming up by the time this is reached.
        If it still isn't responding after 60 s, raise ComfyUIUnavailableError
        so the caller can skip image+video gracefully for this cycle.
        """
        if self._is_server_up():
            logger.debug("ComfyUI ready at %s.", self._url)
            return

        grace   = 60
        interval = 5
        elapsed  = 0
        dots     = 0
        info(f"Waiting for ComfyUI to become ready (up to {grace}s) …")
        while elapsed < grace:
            time.sleep(interval)
            elapsed += interval
            dots    += 1
            print(
                f"\r  ⏳  Waiting for ComfyUI{'.' * (dots % 4):<4} ({elapsed}s/{grace}s){_CLR_EOL}",
                end="",
                flush=True,
            )
            if self._is_server_up():
                print()
                ok(f"ComfyUI ready at {self._url}")
                logger.info("ComfyUI ready after %ds.", elapsed)
                return

        print()
        err(f"ComfyUI not ready after {grace}s — skipping image+video this cycle.")
        logger.warning("ComfyUI not ready after %ds.", grace)
        raise ComfyUIUnavailableError(
            f"ComfyUI not reachable at {self._url} after {grace}s. "
            f"Check log: {self._comfyui / 'comfyui_autostart.log'}"
        )

    def _post_comfy_free(self) -> bool:
        """POST ComfyUI /free (unload_models + free_memory). Best-effort; never raises."""
        try:
            data = b'{"unload_models": true, "free_memory": true}'
            req = urllib.request.Request(
                f"{self._url}/free",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as exc:
            logger.warning("ComfyUI /free failed (non-fatal): %s", exc)
            return False

    def purge_vram_before_batch(self) -> None:
        """
        Ask ComfyUI to unload any loaded models and clear its GPU cache before
        we submit a Z-Image workflow.

        This only affects the **ComfyUI process** (not other apps or the bot's
        Python process). It helps after manual Comfy sessions or fragmentation;
        it does **not** reduce peak VRAM enough to run resolutions the GPU cannot
        physically fit.
        """
        if self._post_comfy_free():
            ok("ComfyUI VRAM purged before Z-Image batch.")
            logger.info("ComfyUI /free before Z-Image batch.")

    def unload_models(self) -> None:
        """
        Ask ComfyUI to unload models from VRAM.

        Called after image generation is complete so that the next pipeline
        stage (e.g. Wan2.1 video generation) has the full GPU budget available.
        Failures are logged but never raised — freeing VRAM is best-effort.
        """
        if self._post_comfy_free():
            ok("ComfyUI models unloaded from VRAM.")
            logger.info("ComfyUI models unloaded from VRAM.")

    def ensure_ready(self) -> None:
        """
        Public entry point for the caller to verify ComfyUI is up before
        starting a generation loop.  Raises ComfyUIUnavailableError if the
        server is still unreachable after the 60-second grace period.

        Separating this from generate() means retry_call() in the image loop
        will never retry a ComfyUIUnavailableError (which already waited 60s);
        it only retries genuine transient ComfyUI API failures.
        """
        self._check_server()

    def generate(self, prompt: str, seed: int = -1) -> List[str]:
        """
        Generate one image from *prompt* using Z-Image-Turbo.

        Resolution is read from Z_IMAGE_TURBO_WIDTH × Z_IMAGE_TURBO_HEIGHT
        (default 832×480, matching Wan2.1 480p I2V input exactly).

        Returns a single-element list containing the path to the saved PNG,
        matching the List[str] contract of the other image provider clients.

        Call ensure_ready() once before starting a generation loop so that
        ComfyUIUnavailableError is raised exactly once (not retried).
        """
        steps  = config.Z_IMAGE_TURBO_STEPS
        width  = config.Z_IMAGE_TURBO_WIDTH
        height = config.Z_IMAGE_TURBO_HEIGHT
        logger.info(
            "ZIT generate: %dx%d steps=%d seed=%d prompt='%s'",
            width, height, steps, seed, prompt[:80],
        )

        # 1. Load workflow (GUI format is converted automatically)
        with open(self._workflow_file, encoding="utf-8") as fh:
            workflow = json.load(fh)

        if "nodes" in workflow:
            logger.debug("ZIT: converting GUI-format workflow to API format")
            workflow = _gui_to_api(workflow, self._url, self._comfyui)
            logger.debug("ZIT: converted %d nodes", len(workflow))

        workflow = _patch_workflow(workflow, prompt, steps, seed, width, height)

        # 2. Submit
        info("submitting to ComfyUI …")
        prompt_id = _submit_prompt(workflow, self._url)
        logger.info("ZIT prompt_id: %s", prompt_id)

        # 3. Poll
        history_entry = _poll_until_done(prompt_id, self._url)

        # 4. Locate and copy output
        comfy_image = _find_output_image(history_entry, self._comfyui)
        if comfy_image is None:
            raise RuntimeError(
                "ZIT generation completed but no image found in ComfyUI output.\n"
                f"Check ComfyUI output folder: {self._comfyui / 'output'}"
            )

        # Second-only timestamps collide when several images finish in the same second,
        # overwriting files and pairing the wrong PNG with a prompt. Use µs + seed.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest = self._images_dir / f"zit_{ts}_seed{seed}{comfy_image.suffix}"
        shutil.copy2(comfy_image, dest)

        logger.info("ZIT image saved: %s (%.0f KB)", dest.name, dest.stat().st_size / 1024)
        return [str(dest)]
