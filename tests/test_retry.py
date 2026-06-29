"""Tests for utils.retry — including the max_attempts<=0 and exclude-exception fixes."""

import unittest

from utils.retry import with_retry, retry_call


class WithRetryTests(unittest.TestCase):
    def test_success_first_try(self):
        calls = []

        @with_retry(max_attempts=3, base_delay=0.0)
        def f():
            calls.append(1)
            return "ok"

        self.assertEqual(f(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        state = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.0)
        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise ValueError("boom")
            return "done"

        self.assertEqual(flaky(), "done")
        self.assertEqual(state["n"], 3)

    def test_exhausts_and_raises(self):
        @with_retry(max_attempts=2, base_delay=0.0)
        def always_fail():
            raise KeyError("nope")

        with self.assertRaises(KeyError):
            always_fail()

    def test_max_attempts_zero_runs_once(self):
        """Misconfigured max_attempts<=0 must still run once, not silently return None."""
        calls = []

        @with_retry(max_attempts=0, base_delay=0.0)
        def f():
            calls.append(1)
            return "ran"

        self.assertEqual(f(), "ran")
        self.assertEqual(len(calls), 1)

    def test_only_listed_exceptions_retried(self):
        """An exception type outside `exceptions` propagates immediately (no retry)."""
        state = {"n": 0}

        @with_retry(max_attempts=5, base_delay=0.0, exceptions=(ValueError,))
        def f():
            state["n"] += 1
            raise TypeError("not retried")

        with self.assertRaises(TypeError):
            f()
        self.assertEqual(state["n"], 1)


class RetryCallTests(unittest.TestCase):
    def test_inline_retry(self):
        state = {"n": 0}

        def flaky(x):
            state["n"] += 1
            if state["n"] < 2:
                raise RuntimeError("retry me")
            return x * 2

        self.assertEqual(retry_call(flaky, 21, max_attempts=3, base_delay=0.0), 42)


if __name__ == "__main__":
    unittest.main()
