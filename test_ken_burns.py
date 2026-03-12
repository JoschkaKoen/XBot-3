"""
Standalone Ken Burns effect test.

Applies a slow pan + zoom to a still image and renders a short MP4.
Optionally overlays an audio track (pass as second arg).

Usage:
    python test_ken_burns.py                             # uses most recent image, no audio
    python test_ken_burns.py Images/foo.png              # specific image, no audio
    python test_ken_burns.py Images/foo.png Voices/bar.mp3  # image + audio

Output: Videos/ken_burns_test.mp4
"""

import os
import sys
import numpy as np
from PIL import Image as PILImage
from moviepy import (
    AudioFileClip,
    ImageClip,
)

OUTPUT_PATH = "Videos/ken_burns_test.mp4"
FPS         = 24
DURATION    = 7.0    # seconds — override by audio length when audio is provided

# ── Ken Burns parameters ──────────────────────────────────────────────────────
# How much to zoom over the full duration (1.10 = 10 % zoom-in)
ZOOM_START  = 1.0
ZOOM_END    = 1.08

# Pan direction: fraction of image width/height to drift in x and y
# (0.03, 0.02) = drift 3 % right and 2 % down over the duration
PAN_X       = 0.03    # positive = drift right
PAN_Y       = 0.02    # positive = drift down


def _pick_latest(directory: str, ext: str) -> str | None:
    candidates = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(ext) and "test_output" not in f
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def make_ken_burns_clip(image_path: str, duration: float, fps: int = FPS) -> ImageClip:
    """
    Return a moviepy ImageClip with Ken Burns pan + zoom applied to the image.

    Uses PIL AFFINE transform for sub-pixel accuracy (no integer rounding
    artifacts) and a smooth-step ease-in-out curve for cinematic motion.
    """
    img = PILImage.open(image_path).convert("RGB")
    iw, ih = img.size

    def apply_ken_burns(frame: np.ndarray, t: float) -> np.ndarray:
        # Raw linear progress 0 → 1
        raw = t / max(duration - 1 / fps, 1 / fps)
        raw = max(0.0, min(1.0, raw))

        # Smooth-step ease-in-out: accelerate from rest, decelerate to rest
        progress = raw * raw * (3 - 2 * raw)

        zoom = ZOOM_START + (ZOOM_END - ZOOM_START) * progress

        # scale < 1 means we sample a smaller region → zoomed in
        scale = 1.0 / zoom
        tx = iw * (1 - scale) / 2 + iw * scale * PAN_X * (progress - 0.5)
        ty = ih * (1 - scale) / 2 + ih * scale * PAN_Y * (progress - 0.5)

        # AFFINE matrix (a,b,c,d,e,f): output(x,y) ← input(a·x+b·y+c, d·x+e·y+f)
        result = img.transform(
            (iw, ih),
            PILImage.AFFINE,
            (scale, 0, tx, 0, scale, ty),
            resample=PILImage.BICUBIC,
        )
        return np.array(result)

    base = ImageClip(image_path, duration=duration)
    return base.transform(lambda gf, t: apply_ken_burns(gf(t), t)).with_fps(fps)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else _pick_latest("Images", ".png")
    audio_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not image_path:
        print("No image found in Images/. Pass a path as argument.")
        sys.exit(1)

    print(f"\n  Image : {image_path}")

    duration = DURATION
    audio_clip = None
    if audio_path:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        print(f"  Audio : {audio_path}  ({duration:.1f}s)")
    else:
        print(f"  Audio : none  (using {duration:.1f}s duration)")

    print(f"  Zoom  : {ZOOM_START:.2f} → {ZOOM_END:.2f}  |  Pan: ({PAN_X:+.2f}, {PAN_Y:+.2f})\n")

    clip = make_ken_burns_clip(image_path, duration)
    if audio_clip:
        clip = clip.with_audio(audio_clip)

    os.makedirs("Videos", exist_ok=True)
    clip.write_videofile(
        OUTPUT_PATH,
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        bitrate="8000k",
        preset="medium",
        threads=4,
        logger=None,
    )
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("  Done. Open Videos/ken_burns_test.mp4 to review.")

    import subprocess
    subprocess.run(["open", OUTPUT_PATH])
