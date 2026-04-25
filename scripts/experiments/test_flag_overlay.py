"""
Standalone test: replace the tricolor flag badge with real flag images from flagcdn.com.

Usage:
    python test_flag_overlay.py                    # uses most recent image in Images/
    python test_flag_overlay.py Images/myfile.png  # uses a specific image

Output: Images/flag_test_output.png
"""

import os
import sys
import io
import requests
from PIL import Image, ImageDraw


# ── Config (edit to try different pairs) ──────────────────────────────────────
SOURCE_CODE = "de"   # ISO 639-1 / ISO 3166-1 alpha-2 code for the source language
TARGET_CODE = "us"   # flagcdn.com uses country codes, not language codes (us not en)
OUTPUT_PATH = "Images/flag_test_output.png"

# Badge geometry (% of image width)
BADGE_WIDTH_RATIO  = 0.10   # badge width  ~10 % of image width
BADGE_HEIGHT_RATIO = 0.60   # badge height ~60 % of badge width (matches flag aspect ratio)
PADDING_RATIO      = 0.015  # gap from top-right corner
CORNER_RADIUS_RATIO = 0.20  # rounded-corner radius as fraction of badge height
OPACITY            = 0.90   # badge opacity (0–1)
BORDER_WIDTH       = 2      # px white border around the whole badge

_FLAGCDN = "https://flagcdn.com/w{w}/{code}.png"


_SUPPORTED_WIDTHS = [20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 320]

def _fetch_flag(code: str, desired_width: int) -> Image.Image:
    # Pick the smallest supported width >= desired_width for crisp rendering
    fetch_w = next((w for w in _SUPPORTED_WIDTHS if w >= desired_width), 160)
    url = _FLAGCDN.format(w=fetch_w, code=code.lower())
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    print(f"  Downloaded {code.upper()} flag  {img.size}  ({fetch_w}px) from {url}")
    return img


def _apply_rounded_corners(img: Image.Image, radius: int) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    img.putalpha(mask)
    return img


import math


def _fit_flag(flag: Image.Image, w: int, h: int) -> Image.Image:
    """Scale-to-fill: zoom until both dimensions are covered, then centre-crop."""
    ratio = max(w / flag.width, h / flag.height)
    new_w = int(flag.width  * ratio)
    new_h = int(flag.height * ratio)
    flag = flag.resize((new_w, new_h), Image.LANCZOS)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    return flag.crop((x0, y0, x0 + w, y0 + h))


def _build_badge(source_code: str, target_code: str, badge_w: int, badge_h: int) -> Image.Image:
    """
    Gradient-blended badge: target flag fades into source flag left→right.
    Uses a cosine ease-in/out curve so there is no visible seam in the middle.
    Left edge = 100% target, right edge = 100% source.
    """
    src_flag = _fetch_flag(source_code, badge_w * 2)
    tgt_flag = _fetch_flag(target_code, badge_w * 2)

    src_img = _fit_flag(src_flag, badge_w, badge_h).convert("RGB")
    tgt_img = _fit_flag(tgt_flag, badge_w, badge_h).convert("RGB")

    # Cosine gradient: smooth S-curve from 255 (left/target) to 0 (right/source)
    gradient = bytes(
        [int(128 + 127 * math.cos(math.pi * x / max(badge_w - 1, 1)))
         for x in range(badge_w)] * badge_h
    )
    mask = Image.frombytes("L", (badge_w, badge_h), gradient)

    return Image.composite(tgt_img, src_img, mask)


def apply_flagcdn_overlay(image_path: str, output_path: str,
                          source_code: str, target_code: str) -> str:
    img = Image.open(image_path).convert("RGBA")
    iw, ih = img.size

    badge_w  = max(int(iw * BADGE_WIDTH_RATIO), 100)
    badge_h  = max(int(badge_w * BADGE_HEIGHT_RATIO), 40)
    padding  = max(int(iw * PADDING_RATIO), 12)
    radius   = max(int(badge_h * CORNER_RADIUS_RATIO), 4)

    badge = _build_badge(source_code, target_code, badge_w, badge_h)

    # Draw border BEFORE rounding so it gets clipped cleanly with the corners
    d = ImageDraw.Draw(badge)
    d.rounded_rectangle(
        [0, 0, badge_w - 1, badge_h - 1],
        radius=radius,
        outline=(255, 255, 255),
        width=BORDER_WIDTH,
    )

    # Rounded corners — clips both badge content and border in one pass
    badge = _apply_rounded_corners(badge, radius)

    # Apply opacity
    r, g, b, a = badge.split()
    a = a.point(lambda v: int(v * OPACITY))
    badge = Image.merge("RGBA", (r, g, b, a))

    # Paste into top-right corner
    img.paste(badge, (iw - badge_w - padding, padding), badge)
    img.convert("RGB").save(output_path, format="PNG")
    print(f"\n  Saved → {output_path}")
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Pick the most recently modified image in Images/
        images_dir = "Images"
        candidates = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not candidates:
            print("No images found in Images/. Pass a path as argument.")
            sys.exit(1)
        input_path = max(candidates, key=os.path.getmtime)

    print(f"\n  Input image : {input_path}")
    print(f"  Source flag : {SOURCE_CODE.upper()}  (right half of badge)")
    print(f"  Target flag : {TARGET_CODE.upper()}  (left half of badge)\n")

    apply_flagcdn_overlay(input_path, OUTPUT_PATH, SOURCE_CODE, TARGET_CODE)
    print("  Done. Open Images/flag_test_output.png to inspect the result.")
