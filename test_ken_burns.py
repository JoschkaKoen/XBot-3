"""
Standalone Ken Burns effect test -- PIL AFFINE + moviepy.

PIL img.transform(AFFINE, BICUBIC) operates in true float space, so there is
no integer rounding between frames and no per-pixel stutter.

The crop window is expressed as a fraction of the image NOT shown (the
"slack").  At ZOOM_START=1.0 the slack is exactly 0, so pan has no effect
and tx/ty are guaranteed to be 0 -- no negative offsets, no out-of-bounds
sampling.

Usage:
    python test_ken_burns.py                               # latest image, no audio
    python test_ken_burns.py Images/foo.png                # specific image, no audio
    python test_ken_burns.py Images/foo.png Voices/bar.mp3 # image + audio

Output: Videos/ken_burns_test.mp4
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image as PILImage
from moviepy import AudioFileClip, VideoClip

OUTPUT_PATH = "Videos/ken_burns_test.mp4"
FPS         = 24
DURATION    = 8.0    # seconds -- overridden by audio length when audio is provided

# Ken Burns parameters
ZOOM_START  = 1.0    # 1.0 = full frame visible
ZOOM_END    = 1.20   # 1.08 = 8% zoom-in at end

# Pan: fraction of available slack to drift across.
# Slack = portion of image not shown = iw * (1 - 1/zoom).
# At zoom=1.0 slack=0, so pan is always 0 at the start -- no out-of-bounds.
# 0.5 = drift across half the available slack; 0.0 = pure center zoom.
PAN_X = 0.0    # positive = drift rightward
PAN_Y = 0.0    # positive = drift downward


def _pick_latest(directory: str, ext: str) -> str | None:
    candidates = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(ext) and "test_output" not in f
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def make_ken_burns_clip(image_path: str, duration: float, fps: int = FPS) -> VideoClip:
    """
    Return a moviepy VideoClip with Ken Burns zoom+pan applied.

    For each frame at time t:
      1. Compute a linear progress p in [0, 1].
      2. zoom = ZOOM_START + (ZOOM_END - ZOOM_START) * p
         scale = 1 / zoom  (the fraction of the image we show)
      3. slack_x = iw * (1 - scale)  -- pixels available to pan in x
         slack_y = ih * (1 - scale)  -- pixels available to pan in y
      4. tx = slack_x * (0.5 + PAN_X * (p - 0.5))
             = 0 when zoom=1 (slack=0), grows as zoom increases.
      5. PIL AFFINE matrix (scale, 0, tx, 0, scale, ty) maps each output
         pixel (x, y) to input float position (scale*x + tx, scale*y + ty),
         bicubic-interpolated -- true sub-pixel, no integer rounding.
    """
    img = PILImage.open(image_path).convert("RGB")
    iw, ih = img.size

    def make_frame(t: float) -> np.ndarray:
        progress = max(0.0, min(1.0, t / max(duration, 1.0 / fps)))

        zoom  = ZOOM_START + (ZOOM_END - ZOOM_START) * progress
        scale = 1.0 / zoom

        # Slack = unshown portion of image (0 when zoom=1)
        slack_x = iw * (1.0 - scale)
        slack_y = ih * (1.0 - scale)

        # Pan within slack: 0.5 = centre, < 0.5 = left/up, > 0.5 = right/down
        tx = slack_x * (0.5 + PAN_X * (progress - 0.5))
        ty = slack_y * (0.5 + PAN_Y * (progress - 0.5))

        # Clamp to valid range (safety, not normally needed for small PAN values)
        tx = max(0.0, min(tx, slack_x))
        ty = max(0.0, min(ty, slack_y))

        frame = img.transform(
            (iw, ih),
            PILImage.AFFINE,
            (scale, 0, tx, 0, scale, ty),
            resample=PILImage.BICUBIC,
        )
        return np.array(frame)

    return VideoClip(make_frame, duration=duration).with_fps(fps)


# Main
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else _pick_latest("Images", ".png")
    audio_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not image_path:
        print("No image found in Images/. Pass a path as argument.")
        sys.exit(1)

    duration = DURATION
    audio_clip = None
    if audio_path:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

    print(f"\n  Image : {image_path}")
    print(f"  Audio : {audio_path}  ({duration:.1f}s)" if audio_path else
          f"  Audio : none  (using {duration:.1f}s duration)")
    print(f"  Zoom  : {ZOOM_START:.2f} -> {ZOOM_END:.2f}")
    print(f"  Pan   : ({PAN_X:+.2f}, {PAN_Y:+.2f})\n")

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

    print(f"\n  Saved -> {OUTPUT_PATH}")
    subprocess.run(["open", OUTPUT_PATH])
