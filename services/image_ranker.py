"""
Image ranking via ImageReward-v1.0.

Scores a list of image paths against a text prompt and returns the path with
the highest reward score.

Loading strategy
----------------
Call `warmup()` once at bot startup to begin loading the model in a background
thread. The model takes ~25s to load from disk. Because content generation and
Midjourney image creation together take several minutes, the model is fully
ready well before `pick_best_image()` is first called — zero waiting time.

If `warmup()` was not called, the model loads on the first `pick_best_image()`
call instead (lazy fallback, same behaviour as before).

Fallback: if the model fails to load for any reason, the first image path is
returned so the pipeline always continues without crashing.
"""

import logging
import threading

logger = logging.getLogger("german_bot.image_ranker")

# ── Singleton state ────────────────────────────────────────────────────────────
_model = None
_model_load_failed = False
# Lock ensures only one thread ever runs the loader, even if pick_best_image()
# is called while warmup() is still in progress.
_lock = threading.Lock()


def _load_model():
    """Internal: load (or skip if already loaded/failed). Must be called under _lock."""
    global _model, _model_load_failed

    if _model is not None or _model_load_failed:
        return

    try:
        import ImageReward as ir
        logger.info("ImageReward: loading model from cache …")
        _model = ir.load("ImageReward-v1.0", device="cpu")
        logger.info("ImageReward: model ready.")
    except Exception as exc:
        logger.warning("ImageReward: could not load model (%s). Will fall back to first image.", exc)
        _model_load_failed = True


def warmup():
    """
    Start loading ImageReward in a background thread so it is ready before
    the first image ranking call. Call this once at bot startup.
    """
    def _worker():
        with _lock:
            _load_model()

    t = threading.Thread(target=_worker, name="ImageRewardWarmup", daemon=True)
    t.start()
    logger.info("ImageReward: warmup started in background thread.")


def _get_model():
    """Return the loaded model, blocking until it is ready if warmup is still running."""
    with _lock:
        _load_model()   # no-op if already loaded or failed
        return _model   # None if load failed


# ── Public API ─────────────────────────────────────────────────────────────────

def score_image(prompt: str, path: str) -> float:
    """
    Score a single image against *prompt* using ImageReward.
    Returns the reward score (higher = better), or 0.0 on any error.
    """
    model = _get_model()
    if model is None:
        return 0.0
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path).convert("RGB")
        score: float = model.score(prompt, img)
        logger.debug("ImageReward score %.4f → %s", score, path)
        return score
    except Exception as exc:
        logger.warning("Could not score image %s (%s) — returning 0.0.", path, exc)
        return 0.0


def pick_best_image(prompt: str, image_paths: list[str]) -> str:
    """
    Score each image in *image_paths* against *prompt* using ImageReward
    and return the path that achieved the highest score.

    Parameters
    ----------
    prompt:
        The text description used to generate the images (Midjourney prompt).
    image_paths:
        Local file paths to the candidate images (typically 4).

    Returns
    -------
    Path of the best-scoring image. Falls back to image_paths[0] on any error.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    model = _get_model()

    if model is None:
        logger.warning("ImageReward unavailable — using first image as fallback.")
        return image_paths[0]

    from PIL import Image as PILImage

    scores: list[tuple[float, str]] = []

    for path in image_paths:
        try:
            img = PILImage.open(path).convert("RGB")
            score: float = model.score(prompt, img)
            scores.append((score, path))
            logger.debug("ImageReward score %.4f → %s", score, path)
        except Exception as exc:
            logger.warning("Could not score image %s (%s) — skipping.", path, exc)

    if not scores:
        logger.warning("All images failed scoring — using first image.")
        return image_paths[0]

    scores.sort(key=lambda x: x[0], reverse=True)

    best_score, best_path = scores[0]
    rank_summary = "  ".join(
        f"#{i+1} {s:.3f}" for i, (s, _) in enumerate(scores)
    )
    logger.info("ImageReward ranking: %s  →  best: %s (%.3f)", rank_summary, best_path, best_score)

    return best_path
