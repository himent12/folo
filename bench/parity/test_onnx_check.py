"""Tests for onnx_check.py's end-to-end row comparison (needs numpy, onnx and onnxruntime: the FOLO ONNX venv).

    python -m unittest discover -s bench/parity -p "test_*.py"
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import onnx_check as oc  # noqa: E402

TOL = 1e-4


def rows(*r):
    return np.array([r], dtype=np.float64).reshape(1, -1, 6)


A = [0.0, 0.0, 10.0, 10.0, 0.9, 0.0]
B = [20.0, 20.0, 40.0, 40.0, 0.5, 1.0]
C = [50.0, 0.0, 60.0, 10.0, 0.5, 2.0]  # ties with B: just past the cutoff of a K = 2 list
D = [70.0, 0.0, 80.0, 10.0, 0.3, 3.0]


class CompareRows(unittest.TestCase):
    def ok(self, got, want, pool=None):
        return oc.compare_rows(got, want, TOL, pool)["ok"]

    def test_identical_and_reordered_ties(self):
        want = rows(A, B, C)
        self.assertTrue(self.ok(want, want))
        self.assertTrue(self.ok(rows(A, C, B), want))

    def test_corrupted_boxes_or_classes_fail(self):
        want = rows(A, B)
        moved = [v + 1000.0 for v in B[:4]] + B[4:]
        self.assertFalse(self.ok(rows(A, moved), want))
        self.assertFalse(self.ok(rows(A, B[:5] + [99.0]), want))
        # Every row corrupted at equal scores (no pool): nothing to excuse them.
        tie = [[x + 1000.0 for x in r[:4]] + [0.5, 99.0] for r in (B, C)]
        self.assertFalse(self.ok(rows(*tie), rows(B, C)))

    def test_cutoff_swap_needs_the_pool(self):
        want, pool = rows(A, B), rows(A, B, C, D)
        # C ties with B at the cutoff: allowed when the pool holds it.
        self.assertTrue(self.ok(rows(A, C), want, pool))
        # Without a longer pool the swap cannot be checked, so it fails.
        self.assertFalse(self.ok(rows(A, C), want))
        # A row at the cutoff score that is in no FOLO list fails.
        stranger = [90.0, 90.0, 99.0, 99.0, 0.5, 4.0]
        self.assertFalse(self.ok(rows(A, stranger), want, pool))
        # A row far from the cutoff may not go missing.
        self.assertFalse(self.ok(rows(C, B), want, pool))

    def test_scores_must_agree(self):
        lower = A[:4] + [0.8, 0.0]
        self.assertFalse(self.ok(rows(lower, B), rows(A, B)))


if __name__ == "__main__":
    unittest.main()
