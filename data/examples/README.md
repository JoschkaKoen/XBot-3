# Example runtime data (public repo)

These files are **templates** for the JSON files the bot writes under `data/`.  
The real files are **gitignored** (they contain tweet URLs, engagement history, your voice pool, etc.).

## Do I need to copy these?

Usually **no**. On first run the bot creates empty or default state as needed:

- Missing `post_history.json` → first scored tweet creates it  
- Missing `strategy.json` → in-code defaults from `nodes/analyze.py`  
- Missing `voice_pool.json` → empty pool; voices are discovered on run  

Optional: copy any file if you want a non-empty starting point:

```bash
cd /path/to/XBot-3
cp data/examples/post_history.json data/post_history.json
# … same for other files if desired
```

Never commit your real `data/*.json` state files to a public repository.
