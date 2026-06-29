"""
Node: publish

Uploads the MP4 to X (Twitter) and posts the tweet text.

================================================================================
 X/TWITTER API KEYS (set in .env, never settings.env)
================================================================================
  TWITTER_CONSUMER_KEY / TWITTER_CONSUMER_SECRET
    → from your app at developer.twitter.com (OAuth 1.0a App-only)

  TWITTER_ACCESS_TOKEN / TWITTER_ACCESS_TOKEN_SECRET
    → generated under "Keys and Tokens" for your specific account

  Required app permissions: Read and Write (to post tweets) + media upload.

Two API versions are used:
  v1.1 — media upload (tweepy.API) — v2 does not support video upload yet
  v2   — tweet creation (tweepy.Client)
================================================================================

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   full_tweet, video_path
  Writes to state:    tweet_id, tweet_url
  Side effects:       posts a tweet on X (Twitter)
================================================================================
"""

import time
import logging
import tweepy

from config import (
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET,
)
from utils.retry import with_retry
from utils.ui import stage_banner, ok, warn as ui_warn

logger = logging.getLogger("xbot.publish")


def _build_clients(creds=None):
    """Build (api_v1, client_v2) for the default account, or for *creds* (an
    AccountCreds-like object with consumer_key/secret + access_token/secret) when
    posting to a secondary account."""
    if creds is not None:
        ck, cs = creds.consumer_key, creds.consumer_secret
        at, ats = creds.access_token, creds.access_token_secret
    else:
        ck, cs = TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET
        at, ats = TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET

    if not all([ck, cs, at, ats]):
        raise ValueError("❌ Missing Twitter/X API keys!")

    auth = tweepy.OAuth1UserHandler(
        consumer_key=ck,
        consumer_secret=cs,
        access_token=at,
        access_token_secret=ats,
    )
    api_v1 = tweepy.API(auth)
    client_v2 = tweepy.Client(
        consumer_key=ck,
        consumer_secret=cs,
        access_token=at,
        access_token_secret=ats,
    )
    return api_v1, client_v2


@with_retry(max_attempts=4, base_delay=10.0, label="upload_video")
def _upload_video(api_v1, video_path: str) -> int:
    """Upload video once and return media_id."""
    logger.info("Uploading video: %s", video_path)
    media = api_v1.media_upload(filename=video_path, media_category="tweet_video")
    logger.info("Media uploaded, ID: %s", media.media_id)
    return media.media_id


@with_retry(max_attempts=6, base_delay=15.0, label="post_tweet")
def _create_tweet(client_v2, text: str, media_id: int) -> tuple:
    """Post the tweet with the already-uploaded media_id."""
    response = client_v2.create_tweet(text=text, media_ids=[media_id])
    tweet_id = str(response.data["id"])
    tweet_url = f"https://x.com/i/web/status/{tweet_id}"
    logger.info("Tweet posted: %s", tweet_url)
    return tweet_id, tweet_url


def post_tweet_with_video(text: str, video_path: str, creds=None) -> tuple:
    """
    Upload video and post tweet. Pass *creds* (AccountCreds-like) to post to a
    secondary account; omit it to use the default account's .env keys.
    Returns (tweet_id, tweet_url).
    """
    api_v1, client_v2 = _build_clients(creds)

    media_id = _upload_video(api_v1, video_path)

    # Give X time to process the video
    time.sleep(5)

    return _create_tweet(client_v2, text, media_id)


# ── node ──────────────────────────────────────────────────────────────────────

def publish(state: dict) -> dict:
    stage_banner(8)
    logger.info("Node: publish")

    text: str = state.get("full_tweet", "")
    video_path = state.get("video_path")

    # create_video returns video_path=None when image generation was skipped
    # (e.g. ComfyUI unavailable, or the provider returned no images). That path
    # is meant to degrade gracefully — posting None to tweepy's media_upload
    # would crash the whole cycle, so skip publishing instead.
    if not video_path:
        ui_warn("No video available — skipping publish for this cycle (nothing posted).")
        logger.warning("publish: video_path missing — tweet not posted this cycle.")
        return {**state, "tweet_id": "", "tweet_url": ""}
    if not text:
        ui_warn("No tweet text available — skipping publish for this cycle (nothing posted).")
        logger.warning("publish: full_tweet missing — tweet not posted this cycle.")
        return {**state, "tweet_id": "", "tweet_url": ""}

    tweet_id, tweet_url = post_tweet_with_video(text, video_path)
    ok(f"Tweet posted → {tweet_url}")

    return {**state, "tweet_id": tweet_id, "tweet_url": tweet_url}
