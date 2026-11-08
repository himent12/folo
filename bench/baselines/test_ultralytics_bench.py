"""CPU-only tests for the bookkeeping in ultralytics_bench.py (no torch, no GPU, no Ultralytics import).

    python -m unittest discover -s bench/baselines -p "test_*.py"
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ultralytics_bench as ub  # noqa: E402


class StatsTest(unittest.TestCase):
    def test_percentile_and_summarize(self):
        ms = [float(i) for i in range(1, 11)]  # 1..10
        self.assertEqual(ub.percentile(ms, 0.9), 9.0)  # index round(0.9 * 9) = 8
        self.assertEqual(ub.percentile(ms, 0.0), 1.0)
        self.assertEqual(ub.percentile(ms, 1.0), 10.0)
        r = ub.summarize(list(reversed(ms)), batch=4)
        self.assertEqual((r["median_ms"], r["p90_ms"], r["min_ms"], r["iters"]), (5.5, 9.0, 1.0, 10))
        self.assertAlmostEqual(r["images_per_s_at_median"], 4 * 1000 / 5.5, places=2)

    def test_cpu_cores_busy(self):
        a, b = (0.0, 100.0, 10.0), (2.0, 120.0, 14.0)  # 2 s wall; system +20 CPU-s, ours +4 CPU-s
        self.assertEqual(ub.cpu_cores_busy(a, b), {"ours": 2.0, "others": 8.0})
        self.assertEqual(ub.cpu_cores_busy(a, (2.0, 101.0, 14.0))["others"], 0.0)  # clamped, never negative


class TrainingSummaryTest(unittest.TestCase):
    def _epochs(self, sizes_per_epoch, dt=0.1):
        stamps, counts, wall, cpu = {}, {}, {}, {}
        t = 0.0
        for e, sizes in enumerate(sizes_per_epoch):
            stamps[e] = [t] + [t + dt * (i + 1) for i in range(len(sizes))]
            counts[e] = list(sizes)
            wall[e] = [t, t + dt * len(sizes)]
            cpu[e] = [(t, 0.0, 0.0), (t + dt * len(sizes), 0.0, 0.0)]
            t += dt * len(sizes)
        return stamps, counts, wall, cpu

    def test_partial_last_batch_counts_real_images(self):
        # 129 images at batch 16 -> 8 full batches + 1 image; must not be reported as 9 * 16 = 144
        sizes = [16] * 8 + [1]
        out = ub.summarize_training(*self._epochs([sizes, sizes]))
        last = out["epochs"][-1]
        self.assertEqual(last["images"], 129)
        self.assertAlmostEqual(last["images_per_s"], 129 / 0.9, places=1)
        self.assertEqual(out["steady_batches"], 9 - 5 + 9)  # first 5 batches of the run are skipped

    def test_mismatched_counts_raise(self):
        stamps, counts, wall, cpu = self._epochs([[16] * 4])
        counts[0].pop()
        with self.assertRaises(ValueError):
            ub.summarize_training(stamps, counts, wall, cpu)


class ContentionTest(unittest.TestCase):
    def _run(self, returncode=0, stdout="", exc=None):
        def fake_run(*a, **k):
            if exc:
                raise exc
            return subprocess.CompletedProcess(a, returncode, stdout=stdout, stderr="")

        with (
            mock.patch.object(ub.sys, "platform", "win32"),
            mock.patch.object(ub.subprocess, "run", fake_run),
            mock.patch.object(ub, "proc_label", lambda pid: f"pid {pid}"),
        ):
            return ub.foreign_gpu_load()

    def test_query_failure_is_unknown_not_quiet(self):
        self.assertIsNone(self._run(returncode=1))
        self.assertIsNone(self._run(exc=subprocess.TimeoutExpired("powershell", 30)))
        self.assertIsNone(self._run(exc=FileNotFoundError("powershell")))
        self.assertEqual(self._run(stdout=""), {})  # a successful query with nothing above 0.5 % is quiet

    def test_parse_sums_engines_and_excludes_ours(self):
        me = os.getpid()
        out = "\n".join(
            [
                "pid_1234_luid_0x0_0x1_phys_0_eng_0_engtype_3d 4.0",
                "pid_1234_luid_0x0_0x1_phys_0_eng_1_engtype_compute_0 3,5",  # comma decimal mark
                f"pid_{me}_luid_0x0_0x1_phys_0_eng_0_engtype_3d 90.0",  # this process
                "pid_4_luid_0x0_0x1_phys_0_eng_10_engtype_copy 30.0",  # System (paging for whoever runs)
                "pid_77_luid_0x0_0x1_phys_0_eng_0_engtype_3d 1.0",  # below BUSY_PCT
            ]
        )
        self.assertEqual(self._run(stdout=out), {"pid 1234": 7.5})

    def test_tag_counts_only_successful_samples(self):
        w = ub.ContentionWatch()
        now = ub.time.perf_counter()
        w.window_start = now - 10
        w.samples = [(now - 9, now - 8, {}), (now - 7, now - 6, None), (now - 5, now - 4, {"pid 9": 50.0})]
        row = w.tag({})
        self.assertEqual(row["contention_samples"], 2)
        self.assertEqual(row["contention_errors"], 1)
        self.assertEqual(row["other_gpu_load_pct"], {"pid 9": 50.0})
        self.assertEqual(w.tag({})["contention_samples"], 0)  # next window: nothing observed yet

    def test_wait_gives_up_on_repeated_query_failure(self):
        with mock.patch.object(ub, "foreign_gpu_load", lambda: None), mock.patch.object(ub.time, "sleep", lambda s: None):
            self.assertIsNone(ub.wait_for_quiet_gpu(max_wait_s=60))
        with mock.patch.object(ub, "foreign_gpu_load", lambda: {}):
            self.assertEqual(ub.wait_for_quiet_gpu(max_wait_s=60), {})


if __name__ == "__main__":
    unittest.main()
