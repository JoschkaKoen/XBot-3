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
from utils.ui import stage_banner, ok

logger = logging.getLogger("german_bot.publish")


def _build_clients():
    if not all([
        TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
    ]):
        raise ValueError("❌ Missing Twitter/X API keys in .env!")

    auth = tweepy.OAuth1UserHandler(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
    )
    api_v1 = tweepy.API(auth)
    client_v2 = tweepy.Client(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
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


def post_tweet_with_video(text: str, video_path: str) -> tuple:
    """
    Upload video and post tweet.
    Returns (tweet_id, tweet_url).
    """
    api_v1, client_v2 = _build_clients()

    media_id = _upload_video(api_v1, video_path)

    # Give X time to process the video
    time.sleep(5)

    return _create_tweet(client_v2, text, media_id)


# ── node ──────────────────────────────────────────────────────────────────────

def publish(state: dict) -> dict:
    stage_banner(7)
    logger.info("Node: publish")

    text: str = state["full_tweet"]
    video_path: str = state["video_path"]

    tweet_id, tweet_url = post_tweet_with_video(text, video_path)
    ok(f"Tweet posted → {tweet_url}")

    return {**state, "tweet_id": tweet_id, "tweet_url": tweet_url}
