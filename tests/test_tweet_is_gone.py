"""
Tests for nodes.fetch_metrics._tweet_is_gone — must use typed 404 / structured
codes, NOT a substring search for "404"/"not found" that would falsely prune a
LIVE tweet from history on an unrelated error.
"""

import unittest

try:
    import tweepy
    from nodes.fetch_metrics import _tweet_is_gone
    _IMPORT_ERR = None
except Exception as exc:
    _tweet_is_gone = None
    tweepy = None
    _IMPORT_ERR = exc


@unittest.skipIf(_tweet_is_gone is None, f"fetch_metrics import unavailable: {_IMPORT_ERR}")
class TweetIsGoneTests(unittest.TestCase):
    def test_unrelated_error_mentioning_404_is_not_gone(self):
        class FakeErr(Exception):
            pass

        self.assertFalse(_tweet_is_gone(FakeErr("rate limited, see https://x.com/err/404")))
        self.assertFalse(_tweet_is_gone(FakeErr("connection reset; not found in cache layer")))

    def test_typed_not_found_is_gone(self):
        exc = tweepy.NotFound.__new__(tweepy.NotFound)
        self.assertTrue(_tweet_is_gone(exc))

    def test_structured_404_status_is_gone(self):
        class WithResponse(Exception):
            class _R:
                status_code = 404
            response = _R()

        self.assertTrue(_tweet_is_gone(WithResponse("boom")))

    def test_api_code_144_is_gone(self):
        class WithCodes(Exception):
            api_codes = [144]

        self.assertTrue(_tweet_is_gone(WithCodes("no status found")))

    def test_generic_error_is_not_gone(self):
        self.assertFalse(_tweet_is_gone(Exception("temporary failure")))


if __name__ == "__main__":
    unittest.main()
