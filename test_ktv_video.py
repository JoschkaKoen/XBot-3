"""
Quick smoke-test for the KTV overlay pipeline.

Tests TWO things without running the full bot or calling any external API:
  1. create_ktv_video()             — still image base  (existing behaviour)
  2. create_ktv_video_from_motion() — animated video base  (new Grok path,
                                      using a pre-existing local video or,
                                      if none exists, a copy of test #1's output)

Run from the project root:
    source venv/bin/activate
    python test_ktv_video.py
"""

import os
import sys
import shutil
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_ktv")

# ── test assets ───────────────────────────────────────────────────────────────

IMAGE_PATH = "Images/mj_00262459-4cc1-4e30-9df8-279efb5cbb58_1.png"
AUDIO_PATH = "Voices/german_ktv_20260225_113142.mp3"

GERMAN_TEXT = "Er schaut die Pizza an und lächelt selig."

# Fake word timings (no real ElevenLabs data needed for the visual test)
WORD_TIMINGS = [
    {"word": "Er",       "start": 0.0,  "end": 0.4},
    {"word": "schaut",   "start": 0.4,  "end": 0.9},
    {"word": "die",      "start": 0.9,  "end": 1.1},
    {"word": "Pizza",    "start": 1.1,  "end": 1.6},
    {"word": "an",       "start": 1.6,  "end": 1.8},
    {"word": "und",      "start": 1.8,  "end": 2.0},
    {"word": "lächelt",  "start": 2.0,  "end": 2.6},
    {"word": "selig.",   "start": 2.6,  "end": 3.2},
]


def _check_assets():
    missing = [p for p in (IMAGE_PATH, AUDIO_PATH) if not os.path.exists(p)]
    if missing:
        print(f"ERROR: test assets not found: {missing}")
        print("Edit IMAGE_PATH / AUDIO_PATH at the top of this file to point at real files.")
        sys.exit(1)


# ── test 1: KTV from still image ──────────────────────────────────────────────

def test_ktv_from_image():
    from nodes.create_video import create_ktv_video

    print("\n─── Test 1: KTV overlay on still image ───")
    out = create_ktv_video(
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        german_text=GERMAN_TEXT,
        word_timings=WORD_TIMINGS,
    )
    assert os.path.exists(out), f"Output file not found: {out}"
    size_kb = os.path.getsize(out) // 1024
    print(f"✅  Created: {out}  ({size_kb} KB)")
    return out


# ── test 2: KTV from animated video base ──────────────────────────────────────

def test_ktv_from_video(base_video_path: str):
    from nodes.create_video import create_ktv_video_from_motion

    print("\n─── Test 2: KTV overlay on animated video base ───")
    out = create_ktv_video_from_motion(
        base_video_path=base_video_path,
        audio_path=AUDIO_PATH,
        german_text=GERMAN_TEXT,
        word_timings=WORD_TIMINGS,
    )
    assert os.path.exists(out), f"Output file not found: {out}"
    size_kb = os.path.getsize(out) // 1024
    print(f"✅  Created: {out}  ({size_kb} KB)")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_assets()

    # Test 1 — still image base
    ktv_image_out = test_ktv_from_image()

    # Test 2 — use the output of test 1 as a stand-in for a Grok-animated video.
    # In production this would be the path returned by grok_video.generate_video().
    print(f"\nUsing '{ktv_image_out}' as a stand-in base video for test 2.")
    test_ktv_from_video(ktv_image_out)

    print("\n✅  All tests passed.  Check the Videos/ folder for the output files.")
