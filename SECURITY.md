# Security

## Secrets

- **Never commit** `.env`, `api-keys.env`, or `*.local.env`. They are listed in `.gitignore`.
- API keys belong only in `.env` on your machine (copy from `.env.example`).
- If a key was ever committed or pasted into a public issue, **rotate it** in the provider’s dashboard.

## Local runtime data

The following files under `data/` are **ignored** because they can contain tweet URLs, engagement metrics, strategy text, and ElevenLabs voice metadata:

`post_history.json`, `strategy.json`, `strategy_history.json`, `metrics_refresh.json`, `video_state.json`, `voice_pool.json`, `scaffold_state.json`

Use `data/examples/` for empty templates if you need them. Do not push real history to a public repo.

## Historical leaks

If you suspect secrets existed in git history, use [git-filter-repo](https://github.com/newren/git-filter-repo) or [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/), then force-push and rotate any exposed credentials.
