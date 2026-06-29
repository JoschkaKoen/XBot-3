"""
Regression tests for XBot-3 pure-logic helpers.

Stdlib ``unittest`` only (no pytest dependency). Run from the project root:

    python -m unittest discover -s tests -v

These cover the parsing, retry, IO, scoring, word-timing, and metrics-pruning
helpers — the pieces with real logic that are cheap to test without the network,
a GPU, or the X API.
"""

import os
import sys

# Ensure the project root is importable even when the discoverer's cwd differs.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
