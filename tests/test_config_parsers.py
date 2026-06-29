"""Tests for config_parsers and config._int_env/_float_env (bad-input tolerance)."""

import os
import unittest

import config_parsers as cp


class StrategyIntervalTests(unittest.TestCase):
    def test_default_when_unset(self):
        self.assertEqual(cp.parse_strategy_update_interval(None), (True, 24))
        self.assertEqual(cp.parse_strategy_update_interval("  "), (True, 24))

    def test_disabled_tokens(self):
        for tok in ("false", "off", "no", "never", "disabled", "NONE"):
            enabled, hours = cp.parse_strategy_update_interval(tok)
            self.assertFalse(enabled, tok)

    def test_integer_and_expression(self):
        self.assertEqual(cp.parse_strategy_update_interval("168"), (True, 168))
        self.assertEqual(cp.parse_strategy_update_interval("24*7"), (True, 168))
        self.assertEqual(cp.parse_strategy_update_interval("24 * 7"), (True, 168))

    def test_garbage_falls_back(self):
        self.assertEqual(cp.parse_strategy_update_interval("soon"), (True, 24))


class MetricsFetchMaxTests(unittest.TestCase):
    def test_default_is_max_30(self):
        self.assertEqual(cp.parse_metrics_fetch_max(None, 10), 30)
        self.assertEqual(cp.parse_metrics_fetch_max(None, 50), 50)

    def test_unlimited_tokens(self):
        for tok in ("0", "all", "unlimited", "none"):
            self.assertEqual(cp.parse_metrics_fetch_max(tok, 10), 0, tok)

    def test_explicit_value(self):
        self.assertEqual(cp.parse_metrics_fetch_max("5", 10), 5)


class UseTrendsCycleTests(unittest.TestCase):
    def test_default_strategy(self):
        self.assertEqual(cp.parse_use_trends_mode_cycle(None), ["strategy"])

    def test_aliases(self):
        self.assertEqual(cp.parse_use_trends_mode_cycle("true"), ["trends"])
        self.assertEqual(cp.parse_use_trends_mode_cycle("false"), ["strategy"])
        self.assertEqual(cp.parse_use_trends_mode_cycle("pool"), ["pool"])

    def test_mixed_cycle(self):
        self.assertEqual(
            cp.parse_use_trends_mode_cycle("trends,trends,pool,pool,strategy"),
            ["trends", "trends", "pool", "pool", "strategy"],
        )

    def test_unknown_token_is_strategy(self):
        self.assertEqual(cp.parse_use_trends_mode_cycle("bogus"), ["strategy"])


class OnOffTests(unittest.TestCase):
    def test_true_values(self):
        for tok in ("true", "1", "yes", "on", "TRUE"):
            self.assertTrue(cp.parse_on_off_env(tok), tok)

    def test_false_values(self):
        for tok in ("false", "0", "no", "off"):
            self.assertFalse(cp.parse_on_off_env(tok), tok)

    def test_default_on_unknown(self):
        self.assertTrue(cp.parse_on_off_env("maybe", default=True))
        self.assertFalse(cp.parse_on_off_env(None, default=False))


class KtvFontSizeTests(unittest.TestCase):
    def test_default(self):
        self.assertEqual(cp.parse_ktv_font_size(None), 80)

    def test_clamped(self):
        self.assertEqual(cp.parse_ktv_font_size("5"), 12)
        self.assertEqual(cp.parse_ktv_font_size("9999"), 200)

    def test_garbage_falls_back(self):
        self.assertEqual(cp.parse_ktv_font_size("big"), 80)


class FlagColorsTests(unittest.TestCase):
    def test_valid_three_colors(self):
        out = cp.parse_flag_colors("000000,DD0000,FFCE00", default=["x"])
        self.assertEqual(out, [(0, 0, 0), (0xDD, 0, 0), (0xFF, 0xCE, 0)])

    def test_too_few_colors_uses_default(self):
        self.assertEqual(cp.parse_flag_colors("000000,DD0000", default=["dflt"]), ["dflt"])

    def test_garbage_uses_default(self):
        self.assertEqual(cp.parse_flag_colors("zzzzzz,gg,hh", default=["dflt"]), ["dflt"])


class IntFloatEnvTests(unittest.TestCase):
    """config._int_env / _float_env must never raise on bad input (startup safety)."""

    def setUp(self):
        import config
        self.config = config

    def _with_env(self, key, value, fn, *args):
        old = os.environ.get(key)
        try:
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
            return fn(*args)
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    def test_int_valid(self):
        self.assertEqual(self._with_env("X_T", "42", self.config._int_env, "X_T", 7), 42)

    def test_int_bad_falls_back(self):
        self.assertEqual(self._with_env("X_T", "forty", self.config._int_env, "X_T", 7), 7)
        self.assertEqual(self._with_env("X_T", "4.5", self.config._int_env, "X_T", 7), 7)

    def test_int_empty_and_unset(self):
        self.assertEqual(self._with_env("X_T", "", self.config._int_env, "X_T", 7), 7)
        self.assertEqual(self._with_env("X_T", None, self.config._int_env, "X_T", 7), 7)

    def test_int_fallback_key(self):
        os.environ.pop("X_NEW", None)
        os.environ.pop("X_OLD", None)
        try:
            os.environ["X_OLD"] = "9"
            self.assertEqual(self.config._int_env("X_NEW", 1, fallback_key="X_OLD"), 9)
        finally:
            os.environ.pop("X_OLD", None)
            os.environ.pop("X_NEW", None)

    def test_float_valid_and_bad(self):
        self.assertAlmostEqual(self._with_env("X_F", "1.5", self.config._float_env, "X_F", 0.1), 1.5)
        self.assertAlmostEqual(self._with_env("X_F", "1,5", self.config._float_env, "X_F", 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
