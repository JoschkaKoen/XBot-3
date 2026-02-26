# German Learning X Bot

An autonomous, fully self-improving X (Twitter) bot that teaches German vocabulary
to English speakers. Powered by LangGraph, ElevenLabs, Midjourney, and your choice
of Grok or Scaleway as the AI backend.

---

## What the bot does

Every ~5.25 hours the bot:

1. **Picks a German word** (noun, verb, adjective, or common phrase) chosen by the LLM
   based on an evolving content strategy.
2. **Writes a short, funny German example sentence** containing the word and translates
   both the word and the sentence to English via DeepL.
3. **Determines the CEFR level** (A1–C1) and looks up the grammatical article
   (der/die/das) for nouns.
4. **Assembles the tweet** in the standard format (see below).
5. **Generates a Midjourney image** (16:9, beautiful photography) matching the sentence.
6. **Generates German TTS audio** via ElevenLabs (Matilda voice, slowed for learners).
7. **Overlays background music** on the voice track.
8. **Renders an MP4 video** — either a simple static image+audio video or a
   karaoke-style video with gold word-by-word highlighting (configurable).
9. **Posts the tweet** with the video attached to X.
10. **Waits ~5.25 hours**, then fetches engagement metrics (likes, reposts, replies,
    quotes, impressions).
11. **Scores** the post and saves it to `data/post_history.json`.
12. **Analyses** the last N posts with the LLM and updates the content strategy
    (preferred CEFR levels, themes, etc.) for future posts.
13. **Loops forever.**

### Tweet format

```
#DeutschLernen A2

🇩🇪  der Führerschein
🇬🇧  driving license  🚗🚗

🇩🇪  Herzlichen Glückwunsch, du hast deinen Führerschein!
🇬🇧  Congratulations, you got your license!  🎉🎉
```

---

## Setup

### 1. Clone / copy the project

```bash
cd ~/Programming
# Already done if you're reading this inside german-bot/
```

### 2. Use the shared virtual environment

This bot shares the venv from your existing `XBot 1` project (all required
packages are already installed or will be added there):

```bash
source "/home/tobias/Programming/XBot 1/venv/bin/activate"
```

If you prefer an isolated venv just for this bot:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
# Make sure the venv is active first (see step 2)
pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

### 4. Install system dependencies

`moviepy` needs **ffmpeg**. The karaoke video mode also needs **ImageMagick**
so that `TextClip` can render text.

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install ffmpeg imagemagick

# Allow ImageMagick to write files (security policy sometimes blocks this)
sudo sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' \
    /etc/ImageMagick-6/policy.xml
# If TextClip still fails, also edit /etc/ImageMagick-6/policy.xml and change
# the @PS/@EPS/@PDF/@XPS lines from rights="none" to rights="read|write".
```

### 5. Configure `.env`

```bash
cp .env.example .env
nano .env   # or your preferred editor
```

| Variable | Description |
|---|---|
| `AI_PROVIDER` | `grok` (xAI) or `scaleway` |
| `XAI_API_KEY` | API key for Grok (xAI). Get it at [console.x.ai](https://console.x.ai) |
| `SCW_SECRET_KEY` | Scaleway secret key (only needed if `AI_PROVIDER=scaleway`) |
| `SCW_DEFAULT_PROJECT_ID` | Scaleway project ID (only needed if `AI_PROVIDER=scaleway`) |
| `TWITTER_CONSUMER_KEY` | X Developer App Consumer Key |
| `TWITTER_CONSUMER_SECRET` | X Developer App Consumer Secret |
| `TWITTER_ACCESS_TOKEN` | X OAuth 1.0a Access Token |
| `TWITTER_ACCESS_TOKEN_SECRET` | X OAuth 1.0a Access Token Secret |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |
| `DEEPL_AUTH_KEY` | DeepL free API key (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx`) |
| `TT_API_KEY` | TTAPI.io key for Midjourney access |
| `POST_INTERVAL_SECONDS` | Seconds between posts (default `18900` ≈ 5.25 h) |
| `HISTORY_FILE` | Path for post history JSON (default `data/post_history.json`) |
| `LOG_FILE` | Path for log file (default `data/bot.log`) |
| `VIDEO_STYLE` | `karaoke` (gold word highlights) or `simple` (static image+audio) |
| `ANALYZE_LAST_N` | How many recent posts to review during self-improvement (default `10`) |

### 6. Add background music

Place your music file at:

```
Background Music/music.mp3
```

The bot will automatically mix the voice audio with this music (volume reduced by
7 dB, faded out at the end). Any royalty-free loopable music works well.

If no music file is found the bot logs a warning and uses voice-only audio — it
will still post successfully.

---

## Running the bot

```bash
source "/home/tobias/Programming/XBot 1/venv/bin/activate"
cd /home/tobias/Programming/german-bot
python main.py
```

The bot logs to both the terminal and `data/bot.log`. It runs indefinitely.
Stop it with `Ctrl+C`.

---

## Deployment on a VPS (systemd)

Create a service file:

```bash
sudo nano /etc/systemd/system/german-bot.service
```

```ini
[Unit]
Description=German Learning X Bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/Programming/german-bot
ExecStart=/home/tobias/Programming/XBot\ 1/venv/bin/python /home/tobias/Programming/german-bot/main.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable german-bot
sudo systemctl start german-bot

# Monitor logs
sudo journalctl -u german-bot -f
# or
tail -f data/bot.log
```

---

## Self-improvement loop

After every post cycle the `analyze_and_improve` node:

1. Reads the last N posts from `data/post_history.json` (configurable via
   `ANALYZE_LAST_N`).
2. Sends the words, CEFR levels, sentences, and engagement scores to the LLM.
3. The LLM identifies which CEFR levels, themes, and sentence styles perform best.
4. It returns a JSON strategy object that is injected into the next
   `generate_content` call, steering word/theme selection.

The strategy includes:
- **`preferred_cefr`** — CEFR levels that got the most engagement
- **`preferred_themes`** — themes to focus on (food, travel, emotions, etc.)
- **`focus`** — a short free-text instruction (e.g. "use funnier sentences")
- **`avoid_words`** — words already used recently (prevents repetition)

The strategy evolves every cycle. After ~20 posts you'll see clear learning
of what your audience responds to.

---

## Code Self-Improvement Pipeline

The bot can automatically improve its own **code** using Claude Code CLI,
triggered during the wait period after posting.

### How it works (4 phases, up to 3 live verification attempts)

```
PHASE 1 — Code Improvement
  • Creates a git branch: auto-improve/YYYYMMDD-HHMM
  • Passes performance data (top/bottom tweets) to Claude Code CLI
  • Claude Code reads source files and makes targeted improvements
  • If requirements.txt was changed, pip install runs automatically
  • Basic import checks verify nothing is broken

PHASE 2+3 — Live Cycle + Verification (repeated up to 3 times)
  • Runs python main.py --single-cycle on the improved branch
  • This runs a REAL complete cycle and posts a REAL tweet to X
  • Four checks must ALL pass:
      ✅ Tweet exists on X (via Tweepy v2 lookup)
      ✅ Tweet text quality ≥ 7/10 (AI review: format, grammar, CEFR, emojis)
      ✅ Image quality score > -1.0 (ImageReward-v1.0)
      ✅ Terminal output score ≥ 7/10 (AI review: all stages present, no crashes)
  • If an attempt fails, Claude Code is asked to review and fix the issue
  • Claude Code may respond FIXED, NO_CHANGE, or GIVE_UP
  • One successful attempt is enough to proceed

PHASE 4 — Decision
  • If ANY attempt passed: kills old bot, starts new bot running on the branch
  • If ALL attempts failed (or GIVE_UP): deletes ALL tweets from failed attempts,
    discards branch, old bot continues
```

### Important: the branch is NEVER auto-merged

The improvement engine does **not** merge branches into main. The flow is:

- Old bot runs on `main`
- Improvement creates branch `auto-improve/YYYYMMDD-HHMM`
- If verification passes → new bot starts **on the branch** (not main)
- Old bot is killed
- You review and merge manually when satisfied

To review and merge:
```bash
bash merge_improvement.sh
```

To go back to main if you don't like the changes:
```bash
git checkout main
python main.py
```

### What Claude Code may (and may not) modify

**Never touched:**
- `.env` — API keys and secrets
- `improve_with_claude_code.py` — self-modification forbidden
- `verify_quality.py` — verifier must stay unchanged

**Fair game (anything else), including:**
- `nodes/generate_content.py` — tweet prompts, word selection, scaffold
- `nodes/generate_image.py` — Midjourney prompt construction
- `nodes/analyze.py` — strategy analysis prompts
- `nodes/score.py` — engagement score weighting
- `services/grok_ai.py` — model constants, call parameters
- `data/strategy.json` — strategy parameters
- Any other file where a genuine improvement is found

### Attempt tracking — all tweets cleaned up on failure

All tweets posted during failed verification attempts are tracked and **deleted
from X** if the overall improvement fails. Their entries are also removed from
`data/post_history.json`. No orphaned test tweets left behind.

### Prerequisites

```bash
npm install -g @anthropic-ai/claude-code   # Claude Code CLI
```

### Configuration

Set in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ENABLE_SELF_IMPROVEMENT` | `false` | Enable automatic improvement runs |
| `IMPROVEMENT_INTERVAL_CYCLES` | `5` | Run improvement every N cycles |
| `IMPROVEMENT_SCORE_THRESHOLD` | `9999` | Only improve if avg score < threshold |

### Manual trigger

```bash
python improve_with_claude_code.py          # respects ENABLE_SELF_IMPROVEMENT setting
python improve_with_claude_code.py --force  # runs regardless of setting
```

### Standalone quality check (after a manual --single-cycle run)

```bash
python main.py --single-cycle      # run one real cycle and write test_cycle_output.json
python verify_quality.py           # check tweet text, image, terminal output
```

### Logs

All improvement engine activity is logged to `data/improvement.log` in addition
to the terminal. Artifacts (`data/test_cycle_output.json`, `data/test_cycle_terminal.txt`)
are automatically deleted after each improvement run.

---

## Video style: simple vs karaoke

Set `VIDEO_STYLE` in `.env`:

| Value | Description |
|---|---|
| `karaoke` | German sentence overlaid at bottom of image; words highlighted gold as they're spoken |
| `simple` | Static image with audio — no text overlay |

Karaoke mode requires ImageMagick (see system dependencies above).

---

## Switching AI providers

Set `AI_PROVIDER` in `.env`:

| Value | Model | Notes |
|---|---|---|
| `grok` | `grok-4-1-fast-non-reasoning` | Fast, excellent for German content |
| `scaleway` | `llama-3.3-70b-instruct` | Free-tier friendly |

To add a new provider:
1. Create `services/my_provider_ai.py` with a `get_my_provider_response()` function
   matching the signature `(user_message, system_prompt, max_tokens, temperature) -> str`.
2. Add an `elif` branch in `services/ai_client.py`.
3. Set `AI_PROVIDER=my_provider` in `.env`.

---

## Project structure

```
german-bot/
├── main.py                      # Entry point (+ --single-cycle mode)
├── graph.py                     # LangGraph graph (nodes, edges, checkpointing)
├── state.py                     # LangGraph TypedDict state schema
├── config.py                    # Env var loading, folder creation, logging
├── improve_with_claude_code.py  # Self-improvement engine (4-phase pipeline)
├── verify_quality.py            # Standalone quality checker
├── merge_improvement.sh         # Interactive branch review & merge helper
├── requirements.txt
├── .env.example
├── README.md
├── nodes/
│   ├── generate_content.py      # Word selection, sentence, translation, tweet assembly
│   ├── generate_image.py        # Midjourney prompt + image generation
│   ├── generate_audio.py        # ElevenLabs TTS (simple + karaoke timings)
│   ├── create_video.py          # combine_audio + simple/karaoke video render
│   ├── publish.py               # Post to X with video
│   ├── fetch_metrics.py         # Fetch tweet engagement via tweepy v2
│   ├── score.py                 # Engagement scoring + JSON history
│   └── analyze.py               # LLM self-improvement strategy update
├── services/
│   ├── ai_client.py             # Switchable AI provider router
│   ├── grok_ai.py               # Grok (xAI) client
│   ├── scaleway_ai.py           # Scaleway Llama client
│   ├── deepl.py                 # DeepL translation
│   └── get_article.py           # German noun article lookup
├── utils/
│   └── retry.py                 # Exponential back-off decorator + helper
├── data/                        # post_history.json, bot.log, checkpoints.sqlite
├── Background Music/            # Place music.mp3 here
├── Images/                      # Midjourney PNGs saved here
├── Voices/                      # ElevenLabs MP3s saved here
├── Voices with Background Music/ # Mixed audio saved here
└── Videos/                      # Final MP4s saved here
```

---

## Crash recovery

LangGraph uses `SqliteSaver` to checkpoint state after every node. If the bot
crashes mid-pipeline (e.g. during Midjourney generation), on the next restart it
automatically resumes from the last successfully completed node — no duplicate
posts, no wasted API calls.

The checkpoint database is stored at `data/checkpoints.sqlite`.
