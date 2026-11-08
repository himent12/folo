"""Tests for render_tables.py (pure Python, no torch).

    python -m unittest discover -s bench/baselines -p "test_*.py"
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_tables as rt  # noqa: E402

ENV = {
    "gpu": "GPU-A",
    "nvidia_smi": {"memory.total": "16376", "driver_version": "1.0"},
    "cpu": "CPU-A",
    "cpu_physical": 20,
    "cpu_logical": 28,
    "os": "OS",
    "python": "3.12",
    "torch": "2.11",
    "torch_cuda": "12.8",
    "cudnn": 1,
    "ultralytics": "8.4",
}
CFG = {"synthetic_dataset": {"train": 256, "val": 32}, "epochs": 3, "quick": False}


def train_row(**kw):
    epoch = {
        "images_per_s": 100.0,
        "median_batch_ms": 80.0,
        "cpu_cores_busy": {"ours": 2.0, "others": 1.0},
    }
    row = {"model": "yolo26n", "amp": True, "repeat": 0, "batch": 16, "steady_images_per_s": 120.0, "epochs": [epoch]}
    row.update(kw)
    return row


def render(d: dict) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rt.render(d)
    return buf.getvalue()


class ContentionLabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(rt.contention_label({"contention_samples": 0, "contention_errors": 3}), "unobserved (3 failed queries)")
        self.assertEqual(rt.contention_label({"contention_samples": 0}), "unobserved")
        self.assertEqual(rt.contention_label({"contention_samples": 4, "other_gpu_load_pct": {}}), "none")
        self.assertEqual(rt.contention_label({"other_gpu_load_pct": {}}), "none seen (legacy sampler)")
        self.assertEqual(
            rt.contention_label({"contention_samples": 2, "contention_errors": 1, "other_gpu_load_pct": {"pid 9 x": 7.0}}),
            "pid 9 x 7.0% (1 failed queries)",
        )


class RenderTest(unittest.TestCase):
    def doc(self, rows, meta_env=None):
        return {
            "date": "today",
            "environment": ENV,
            "config": CFG,
            "sections_meta": {"train": {"environment": meta_env or ENV, "config": CFG}},
            "train": rows,
        }

    def test_trainer_table_uses_effective_batch_config_and_unknown_contention(self):
        out = render(self.doc([train_row(effective_batch=8, contention_samples=0, contention_errors=3), train_row()]))
        self.assertIn("synthetic 256-image set, batch 16 requested", out)
        self.assertIn("epoch-3 img/s", out)
        lines = [ln for ln in out.splitlines() if ln.startswith("| yolo26n")]
        self.assertIn("| 8 |", lines[0])  # OOM-halved batch is shown
        self.assertIn("unobserved (3 failed queries)", lines[0])
        self.assertIn("| 16 |", lines[1])  # older rows fall back to the requested batch
        self.assertIn("none seen (legacy sampler)", lines[1])

    def test_mixed_environments_are_not_merged(self):
        d = self.doc([train_row()], meta_env={**ENV, "gpu": "GPU-B"})
        d["train_step"] = []
        d["sections_meta"]["train_step"] = {"environment": ENV, "config": CFG}
        out = render(d)
        self.assertIn("`train`:", out)
        self.assertIn("`train_step`:", out)
        self.assertIn("GPU-B", out)

    def test_single_environment_is_one_table(self):
        out = render(self.doc([train_row()]))
        self.assertEqual(out.count("| GPU |"), 1)
        self.assertNotIn("`train`:", out)

    def test_schema1_falls_back_to_top_level(self):
        d = self.doc([train_row()])
        del d["sections_meta"]
        self.assertIn("GPU-A", render(copy.deepcopy(d)))


if __name__ == "__main__":
    unittest.main()
