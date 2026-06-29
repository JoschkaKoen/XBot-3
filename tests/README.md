# Tests

Fast, dependency-light regression tests for XBot-3's pure-logic helpers. They use
the standard-library `unittest` (no `pytest` needed) and avoid the network, the
GPU, and the X API, so they run in well under a second.

## Run

From the project root, with the venv active:

```bash
python -m unittest discover -s tests -v
```

Or a single module:

```bash
python -m unittest tests.test_word_timings -v
```

## Coverage

| File | What it guards |
|---|---|
| `test_config_parsers.py` | `config_parsers.*` parsing + `config._int_env/_float_env` falling back on bad `settings.env` input instead of crashing startup |
| `test_retry.py` | `utils.retry.with_retry` / `retry_call` — retry-then-succeed, exhaust-and-raise, `max_attempts<=0` still runs once, only listed exception types retried |
| `test_io_and_score.py` | `utils.io.atomic_json_write` / `safe_json_read` round-trip + corrupt-file fallback; `services.history.compute_score` weighting |
| `test_word_timings.py` | `generate_audio._character_alignment_to_word_timings` — one entry per word, monotonic starts, repeated/substring/missing words |
| `test_tweet_is_gone.py` | `fetch_metrics._tweet_is_gone` — typed 404 / structured codes only, never a `"404"` substring that would prune a live tweet |

Tests that need the heavier node imports (`generate_audio`, `fetch_metrics`) skip
themselves cleanly if those optional dependencies aren't installed.
