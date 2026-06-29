"""Tests for services.history scoring/age/ranking helpers."""

import unittest
from datetime import datetime, timedelta, timezone

from services.history import (
    tweet_age_hours,
    normalized_score,
    get_top_tweets,
    _MIN_AGE_HOURS,
)


def _iso_hours_ago(h: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()


class TweetAgeHoursTests(unittest.TestCase):
    def test_missing_timestamp_defaults_24(self):
        self.assertEqual(tweet_age_hours({}), 24.0)
        self.assertEqual(tweet_age_hours({"timestamp": ""}), 24.0)

    def test_bad_timestamp_defaults_24(self):
        self.assertEqual(tweet_age_hours({"timestamp": "not-a-date"}), 24.0)

    def test_recent_is_clamped_to_min(self):
        # 1 hour ago → clamped up to the 6h floor
        self.assertEqual(tweet_age_hours({"timestamp": _iso_hours_ago(1)}), float(_MIN_AGE_HOURS))

    def test_older_returns_actual(self):
        age = tweet_age_hours({"timestamp": _iso_hours_ago(10)})
        self.assertAlmostEqual(age, 10, delta=0.5)

    def test_naive_timestamp_treated_as_utc(self):
        naive = (datetime.now(timezone.utc) - timedelta(hours=12)).replace(tzinfo=None).isoformat()
        age = tweet_age_hours({"timestamp": naive})
        self.assertAlmostEqual(age, 12, delta=0.5)


class NormalizedScoreTests(unittest.TestCase):
    def test_score_divided_by_age(self):
        rec = {"engagement_score": 60.0, "timestamp": _iso_hours_ago(10)}
        # ~60 / ~10 ≈ 6 per hour
        self.assertAlmostEqual(normalized_score(rec), 6.0, delta=0.5)

    def test_zero_score(self):
        self.assertEqual(normalized_score({"engagement_score": 0.0, "timestamp": _iso_hours_ago(10)}), 0.0)


class GetTopTweetsTests(unittest.TestCase):
    def test_excludes_zero_score_and_empty_tweet(self):
        history = [
            {"full_tweet": "a", "engagement_score": 0.0, "timestamp": _iso_hours_ago(10)},   # score 0 → excluded
            {"full_tweet": "",  "engagement_score": 50.0, "timestamp": _iso_hours_ago(10)},  # empty → excluded
            {"full_tweet": "c", "engagement_score": 50.0, "timestamp": _iso_hours_ago(10)},  # kept
        ]
        top = get_top_tweets(history, n=5)
        self.assertEqual([r["full_tweet"] for r in top], ["c"])

    def test_orders_by_normalized_score(self):
        history = [
            {"full_tweet": "older_high", "engagement_score": 100.0, "timestamp": _iso_hours_ago(50)},  # 2/h
            {"full_tweet": "newer_mid",  "engagement_score": 60.0,  "timestamp": _iso_hours_ago(10)},  # 6/h
        ]
        top = get_top_tweets(history, n=2)
        # normalized: newer_mid (6/h) should rank above older_high (2/h)
        self.assertEqual(top[0]["full_tweet"], "newer_mid")

    def test_respects_n(self):
        history = [
            {"full_tweet": f"t{i}", "engagement_score": float(i + 1), "timestamp": _iso_hours_ago(10)}
            for i in range(5)
        ]
        self.assertEqual(len(get_top_tweets(history, n=2)), 2)


if __name__ == "__main__":
    unittest.main()
