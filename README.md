# XBot 3 — Language Learning X Bot

An autonomous X (Twitter) bot that teaches vocabulary from any source language to any target language. Posts every ~5 hours with AI-generated images, karaoke-style animated videos, and native-speaker TTS audio. Self-improving: the bot analyses its own engagement and updates its content strategy automatically.

---

## What the bot does

Every cycle the bot:

1. **Picks a vocabulary word** — chosen by the LLM based on an evolving content strategy (CEFR level, theme, style).
2. **Writes a short example sentence** in the source language and translates both the word and the sentence to the target language.
3. **Determines CEFR level** (A1–C2) and looks up the grammatical article (der/die/das for German nouns).
4. **Generates an image** — via Grok Imagine or Midjourney, matching the example sentence. Multiple images are scored and ranked automatically.
5. **Animates the image** — optionally via Grok Imagine I2V or a local Wan2.1/2.2 model, producing a short MP4.
6. **Generates TTS audio** — via ElevenLabs. Voice is selected per-tweet by the LLM based on the sentence mood.
7. **Mixes background music** onto the voice track.
8. **Renders a KTV video** — karaoke-style word-by-word highlighting synced to the audio, overlaid on the animated image.
9. **Posts the tweet** with the video attached to X.
10. **Waits**, then fetches engagement metrics (impressions, likes, reposts, replies, bookmarks).
11. **Scores** the post and updates `data/post_history.json`.
12. **Analyses** recent posts with the LLM and updates the content strategy for the next cycle.
13. **Loops forever.** Catches its own errors and resumes automatically.

### Example tweet

```
🇩🇪  der Führerschein  (B1)
🇺🇸  driving licence  🚗

🇩🇪  Herzlichen Glückwunsch, du hast deinen Führerschein bestanden!
🇺🇸  Congratulations, you passed your driving test!  🎉

#LearnGerman #GermanVocabulary
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/JoschkaKoen/XBot-3.git
cd "XBot-3"
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install ffmpeg imagemagick fonts-lato

# Allow ImageMagick to write files (required for KTV video rendering)
sudo sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' \
    /etc/ImageMagick-6/policy.xml
```

### 5. Configure API keys (`.env`)

Create `.env` in the project root with your secrets (this file is gitignored):

```bash
cp .env.example .env
nano .env
```

| Variable | Description |
|---|---|
| `XAI_API_KEY` | xAI API key — used for Grok LLM, Grok Imagine image generation, and Grok I2V |
| `TWITTER_CONSUMER_KEY` | X Developer App Consumer Key |
| `TWITTER_CONSUMER_SECRET` | X Developer App Consumer Secret |
| `TWITTER_ACCESS_TOKEN` | X OAuth 1.0a Access Token |
| `TWITTER_ACCESS_TOKEN_SECRET` | X OAuth 1.0a Access Token Secret |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for TTS |
| `TT_API_KEY` | TTAPI.io key for Midjourney access (only if `IMAGE_PROVIDER=midjourney`) |
| `SCW_SECRET_KEY` | Scaleway secret key (only if `AI_PROVIDER=scaleway`) |
| `SCW_DEFAULT_PROJECT_ID` | Scaleway project ID (only if `AI_PROVIDER=scaleway`) |

**Optional:** set `WAN_VIDEO_DIR` here if you use local Wan video (`ENABLE_VIDEO=wan`) and don’t want to edit `settings.env`.

### Local runtime data & privacy

Tweet history, strategy state, and voice cache live under `data/` as JSON files. Those files are **gitignored** in this repo so clones don’t inherit your posts or account-specific state. Empty templates are in `data/examples/` (see `data/examples/README.md`). The bot creates what it needs on first run; you usually don’t have to copy anything.

See **`SECURITY.md`** for secrets and safe publishing.

### 6. Configure bot behaviour (`settings.env`)

All non-secret settings live in `settings.env`. Key parameters:

| Setting | Default | Description |
|---|---|---|
| `SOURCE_LANGUAGE` | `German` | Language being taught |
| `TARGET_LANGUAGE` | `English` | Language of the audience |
| `AI_PROVIDER` | `grok` | `grok` or `scaleway` |
| `IMAGE_PROVIDER` | `grok` | `grok` or `midjourney` |
| `IMAGE_STYLE` | `photographic` | `photographic`, `disney`, or comma-separated cycle |
| `TWEET_STYLE` | `funny,normal` | `funny`, `normal`, or comma-separated cycle |
| `VIDEO_STYLE` | `ktv` | `ktv` (karaoke highlights) or `simple` (static) |
| `ENABLE_VIDEO` | `grok` | `off`, `grok` (Grok I2V), or `wan` (local Wan2.1) |
| `VIDEO_FREQUENCY` | `2` | Generate video every N tweets |
| `WAN_VIDEO_STEPS` | `10` | Denoising steps for Wan video (higher = slower + better) |
| `WAN_VIDEO_DIR` | `/path/to/Wan2GP` (template) | Path to Wan2GP installation — set your real path here or via `WAN_VIDEO_DIR` in `.env` |
| `ENABLE_KEN_BURNS` | `false` | Apply Ken Burns zoom/pan when no animated video |
| `FLAG_OVERLAY` | `true` | Add source→target language flag badge to images |
| `POST_INTERVAL_SECONDS` | `18000` | Seconds between posts (18000 = 5 hours) |
| `MAX_TWEET_LENGTH` | `500` | Max characters (280 for standard, up to 25000 for Premium) |
| `USE_TRENDS` | `false` | `true` / `false`, or a comma cycle (e.g. `true,false,false,false`) — trends only on selected cycles; same index as `TWEET_STYLE` / `IMAGE_STYLE` |
| `ANALYZE_LAST_N` | `10` | How many recent posts the strategy LLM sees (engagement summary) |
| `METRICS_FETCH_MAX_TWEETS` | _(see below)_ | Max `get_tweet` calls per metrics refresh. **Default if unset:** `max(ANALYZE_LAST_N, 30)` — only the **newest** N posts are refreshed; older rows keep stored scores. Set `0` or `all` for no cap (entire `post_history.json`). |
| `STRATEGY_UPDATE_INTERVAL_HOURS` | `24` | Hours between X metrics refresh + strategy re-analysis. Use `false` / `off` / `never` / `disabled` to **disable both** (no metric API calls). Integers or expressions like `168` or `24*7` (weekly) are OK. |
| `AUTO_UPDATE` | `true` | Auto-pull from `origin/main` and restart between cycles |
| `ENABLE_SELF_IMPROVEMENT` | `false` | Enable automatic code self-improvement |

### 7. Add background music

Place a royalty-free loopable MP3 at:

```
Background Music/music.mp3
```

The voice audio is mixed with this track (music lowered by 7 dB, faded out at the end). If no file is found the bot uses voice-only audio and continues normally.

---

## Running the bot

```bash
source venv/bin/activate
python main.py
```

The bot logs to both the terminal and `data/bot.log`. Stop it with `Ctrl+C` — it finishes the current cycle cleanly before exiting.

---

## Video generation

### KTV overlay (karaoke)

When `VIDEO_STYLE=ktv`, the bot renders a word-by-word karaoke highlight over the video: each word in the source-language sentence turns white as it is spoken, with a semi-transparent black bar and light-blue text styling.

### Animated video engines

Set `ENABLE_VIDEO` to control what plays behind the KTV overlay:

| Value | Description | Speed |
|---|---|---|
| `off` | Static image (or Ken Burns pan if `ENABLE_KEN_BURNS=true`) | instant |
| `grok` | Grok Imagine I2V — cloud API, 720p, ~15 s | ~15 s |
| `wan` | Local Wan2.1/2.2 via Wan2GP — 480p, runs entirely on your GPU | ~7–40 min |

`VIDEO_FREQUENCY=N` skips animation every N-1 out of N tweets to reduce cost or generation time.

### Local Wan2GP setup (optional)

To use `ENABLE_VIDEO=wan` you need [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) installed alongside this project and `run_i2v.py` present in the Wan2GP directory. Set `WAN_VIDEO_DIR` in `settings.env` to its path.

---

## Language pair configuration

Set `SOURCE_LANGUAGE` and `TARGET_LANGUAGE` in `settings.env` to any pair (e.g. `French` / `English`, `Spanish` / `German`). On first run the bot calls the LLM to derive language codes, flag emojis, flag colors, and the Google Trends country — the result is cached in `data/language_config.json` and reused on subsequent starts.

---

## Self-improvement loop (content strategy)

After every `STRATEGY_UPDATE_INTERVAL_HOURS` hours the bot:

1. Reads the last `ANALYZE_LAST_N` posts from `data/post_history.json`.
2. Sends words, CEFR levels, sentences, and engagement scores to the LLM.
3. The LLM identifies which themes, CEFR levels, and sentence styles perform best.
4. Returns a JSON strategy object saved to `data/strategy.json` and injected into every subsequent content generation call.

The strategy evolves continuously. After ~20 posts the bot noticeably steers toward what your audience engages with most.

---

## Code self-improvement engine

When `ENABLE_SELF_IMPROVEMENT=true`, the bot runs an automated 4-phase pipeline during the wait period to improve its own source code using Claude Code CLI.

### How it works

```
PHASE 1 — Improvement
  • Creates branch: auto-improve/YYYYMMDD-HHMM
  • Passes top/bottom performing tweets to Claude Code
  • Claude makes targeted code changes (prompts, scoring, strategy logic, etc.)

PHASE 2+3 — Live Verification (up to 3 attempts)
  • Runs python main.py --single-cycle on the improved branch
  • Posts a REAL tweet to X
  • Four checks must ALL pass:
      ✅ Tweet found on X (Tweepy v2 lookup)
      ✅ Tweet text quality ≥ 7/10 (AI review: format, grammar, CEFR, emojis)
      ✅ Image quality score > -1.0 (ImageReward model)
      ✅ Terminal output quality ≥ 7/10 (AI review: no crashes, all stages)
  • On failure, Claude Code reviews and may fix or give up

PHASE 4 — Decision
  • All checks passed → new bot process starts on the improved branch
  • All failed → failed tweets deleted from X, branch discarded, original continues
```

The improved branch is **never auto-merged into main** — you review and merge manually:

```bash
bash merge_improvement.sh
```

### Prerequisites

```bash
npm install -g @anthropic-ai/claude-code
```

### Manual trigger

```bash
python improve_with_claude_code.py          # respects ENABLE_SELF_IMPROVEMENT
python improve_with_claude_code.py --force  # always runs
```

---

## Crash recovery

LangGraph checkpoints state after every node using `SqliteSaver` (`data/checkpoints.sqlite`). If the bot crashes mid-pipeline it resumes from the last completed node on restart — no duplicate posts, no wasted API calls.

---

## Project structure

```
XBot 3/
├── main.py                        # Entry point; --single-cycle mode for verification
├── graph.py                       # LangGraph pipeline (nodes, edges, checkpointing)
├── state.py                       # LangGraph TypedDict state schema
├── config.py                      # Settings loader, folder creation, logging
├── scaffolds.py                   # Tweet format templates
├── improve_with_claude_code.py    # Code self-improvement engine
├── verify_quality.py              # Standalone quality checker
├── merge_improvement.sh           # Interactive branch review & merge helper
├── requirements.txt
├── settings.env                   # Public bot configuration (committed)
├── .env                           # Secret API keys (gitignored)
├── .env.example                   # Template for .env
├── nodes/
│   ├── generate_content.py        # Word selection, sentence, translation, tweet assembly
│   ├── generate_image.py          # Image generation (Grok Imagine or Midjourney) + ranking
│   ├── generate_audio.py          # ElevenLabs TTS — voice selection + karaoke timings
│   ├── create_video.py            # Audio mix, video animation, KTV overlay
│   ├── publish.py                 # Post tweet with video to X
│   ├── fetch_metrics.py           # Fetch engagement metrics via Tweepy v2
│   ├── score.py                   # Engagement scoring + post history update
│   └── analyze.py                 # LLM strategy analysis + update
├── services/
│   ├── ai_client.py               # Switchable AI provider router
│   ├── grok_ai.py                 # xAI Grok LLM client
│   ├── scaleway_ai.py             # Scaleway Llama client
│   ├── grok_video.py              # Grok Imagine I2V service
│   ├── wan_video.py               # Local Wan2.1/2.2 I2V service (via Wan2GP)
│   ├── image_ranker.py            # ImageReward scoring for image selection
│   ├── voice_pool.py              # ElevenLabs voice management
│   ├── language_config.py         # AI-derived language pair config + caching
│   └── x_trends.py                # Google Trends integration
├── utils/
│   ├── ui.py                      # Terminal banner and cycle output formatting
│   └── retry.py                   # Exponential back-off decorator
├── data/                          # post_history.json, strategy.json, bot.log, checkpoints.sqlite
├── Background Music/              # Place music.mp3 here
├── Images/                        # Generated images saved here
├── Voices/                        # ElevenLabs audio saved here
├── Voices with Background Music/  # Mixed audio saved here
└── Videos/                        # Final MP4 videos saved here
```
