"""Durable, per-channel posting ledger. Ambiguous writes never auto-replay."""
from pathlib import Path
import time
from utils.runtime import Health, category

class DeliveryPending(RuntimeError):
    pass


def publish_once(channel, cycle, text, video, send, root='data', record=None):
    ledger = Health(Path(root) / f'delivery-{channel}.json')
    old = ledger.read()
    if old.get('cycle') == cycle:
        if old.get('status') == 'ok':
            return old['tweet_id'], old['tweet_url']
        if old.get('status') in ('posting', 'uncertain'):
            raise DeliveryPending('Uncertain X write; reconcile this cycle before retrying')
    elif old.get('status') in ('posting', 'uncertain'):
        # Even a new cycle must not hide a previous response-loss write.
        raise DeliveryPending('Previous X write needs reconciliation')
    health = Health(Path(root) / 'health.json')
    if old.get('status') == 'billing' and old.get('retry_after', 0) > time.time():
        health.delivery(f'x:{channel}', 'failed', failure='billing')
        raise DeliveryPending('X billing hold; retry on the next eligible slot')
    ledger.update(cycle=cycle, status='posting', started_at=time.time(), record=record or {})
    try:
        tweet_id, tweet_url = send(text, video)
    except Exception as exc:
        kind = category(exc)
        # Typed 4xx response proves rejection. Transport loss may follow a write.
        from utils.runtime import http_status
        code = http_status(exc)
        status = kind if code and 400 <= code < 500 else 'uncertain'
        ledger.update(status=status, retry_after=time.time() + 3600 if kind == 'billing' else 0)
        health.delivery(f'x:{channel}', 'failed', failure=kind)
        raise
    ledger.update(status='ok', tweet_id=tweet_id, tweet_url=tweet_url,
                  last_success_at=time.time())
    health.delivery(f'x:{channel}', 'ok', tweet_id=tweet_id, tweet_url=tweet_url, failure=None)
    health.update(last_success_at=time.time())
    return tweet_id, tweet_url


def recover_primary_history(root='data'):
    """Repair a confirmed-post/history-save crash without any API requests."""
    from datetime import datetime, timezone
    from utils.io import safe_json_read, atomic_json_write
    root = Path(root)
    ledger = Health(root / 'delivery-primary.json').read()
    record = ledger.get('record')
    if ledger.get('status') != 'ok' or not record:
        return
    path = root / 'post_history.json'
    history = safe_json_read(path, default=[])
    if any(str(r.get('tweet_id')) == str(ledger['tweet_id']) for r in history):
        return
    history.append({**record, 'tweet_id': ledger['tweet_id'], 'tweet_url': ledger['tweet_url'],
        'timestamp': datetime.fromtimestamp(ledger['last_success_at'], timezone.utc).isoformat(),
        'metrics': {}, 'engagement_score': 0})
    atomic_json_write(path, history, ensure_ascii=False, indent=2)
