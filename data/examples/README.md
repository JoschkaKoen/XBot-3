# Example runtime data (public repo)

These files are **templates** for the JSON files the bot writes under `data/`.  
The real files are **gitignored** (they contain tweet URLs, engagement history, your voice pool, etc.).

## Do I need to copy these?

Usually **no**. On first run the bot creates empty or default state as needed:

- Missing `post_history.json` → first scored tweet creates it  
- Missing `strategy.json` → in-code defaults from `nodes/analyze.py`  
- Missing `voice_pool.json` → empty pool; voices are discovered on run  
- Missing `theme_recent.json` → created when the first **pool** theme is picked (`USE_TRENDS` cycle includes `pool`)

`post_history.json` rows may include:

- `used_trend` — non-empty when the word came from the **trends** path (headline from the trend list).
- `pool_theme` — non-empty when the cycle used **pool** (string from `data/themes_german_for_english_learners.json`).

Optional: copy any file if you want a non-empty starting point:

```bash
cd /path/to/XBot-3
cp data/examples/post_history.json data/post_history.json
# … same for other files if desired
```

Never commit your real `data/*.json` state files to a public repository.
