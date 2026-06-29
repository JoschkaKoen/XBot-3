"""Tests for utils.io (atomic JSON) and services.history.compute_score."""

import os
import tempfile
import unittest

from utils.io import atomic_json_write, safe_json_read, stream_to_file
from services.history import compute_score


class _FakeResp:
    """Minimal duck-typed requests.Response for stream_to_file."""

    def __init__(self, body: bytes, content_length=None, chunk=4):
        self._body = body
        self._chunk = chunk
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def iter_content(self, size):
        for i in range(0, len(self._body), self._chunk):
            yield self._body[i:i + self._chunk]


class AtomicJsonTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="xbot_io_")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.unlink(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def test_round_trip(self):
        path = os.path.join(self.dir, "a.json")
        atomic_json_write(path, {"k": 1, "list": [1, 2, 3]}, indent=2)
        self.assertEqual(safe_json_read(path), {"k": 1, "list": [1, 2, 3]})

    def test_no_temp_files_left_behind(self):
        path = os.path.join(self.dir, "b.json")
        atomic_json_write(path, {"x": "y"})
        leftovers = [n for n in os.listdir(self.dir) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_safe_read_missing_returns_default(self):
        self.assertEqual(safe_json_read(os.path.join(self.dir, "nope.json"), default=[]), [])

    def test_safe_read_corrupt_returns_default(self):
        path = os.path.join(self.dir, "corrupt.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        self.assertEqual(safe_json_read(path, default={"fallback": True}), {"fallback": True})

    def test_overwrite_is_atomic_and_complete(self):
        path = os.path.join(self.dir, "c.json")
        atomic_json_write(path, {"v": 1})
        atomic_json_write(path, {"v": 2})
        self.assertEqual(safe_json_read(path), {"v": 2})


class StreamToFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="xbot_dl_")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.unlink(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def test_complete_download(self):
        body = b"hello world" * 100
        path = os.path.join(self.dir, "ok.bin")
        written = stream_to_file(_FakeResp(body, content_length=len(body)), path)
        self.assertEqual(written, len(body))
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), body)

    def test_truncated_download_raises(self):
        body = b"only half arrived"
        path = os.path.join(self.dir, "trunc.bin")
        # advertises more bytes than the body delivers
        with self.assertRaises(IOError):
            stream_to_file(_FakeResp(body, content_length=len(body) + 50), path)

    def test_no_content_length_is_accepted(self):
        body = b"unknown length"
        path = os.path.join(self.dir, "nolen.bin")
        written = stream_to_file(_FakeResp(body, content_length=None), path)
        self.assertEqual(written, len(body))


class ComputeScoreTests(unittest.TestCase):
    def test_empty_metrics_is_zero(self):
        self.assertEqual(compute_score({}), 0.0)

    def test_weighted_formula(self):
        # likes + 3*reposts + 5*replies + 2*quotes + impressions/100
        metrics = {
            "like_count": 10,
            "retweet_count": 2,
            "reply_count": 1,
            "quote_count": 3,
            "impression_count": 1000,
        }
        expected = 10 + 3 * 2 + 5 * 1 + 2 * 3 + 1000 / 100
        self.assertEqual(compute_score(metrics), round(expected, 2))

    def test_partial_metrics(self):
        self.assertEqual(compute_score({"like_count": 5}), 5.0)
        self.assertEqual(compute_score({"impression_count": 250}), 2.5)


if __name__ == "__main__":
    unittest.main()
