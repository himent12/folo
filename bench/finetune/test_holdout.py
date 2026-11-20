"""Tests of the hold-out selection (holdout.py) and of what run_folo.py does with it: the `holdout` variant, the
dataset it registers, and `last.folo` scored next to `best.folo`.

    python -m unittest discover -s bench/finetune -p "test_*.py"

No GPU or server: run_folo.py runs against fakes, and `folo val` + the scorer are replaced by a recorder. The carving
tests need Pillow (image sizes) and are skipped without it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import datasets  # noqa: E402
import fake_folo  # noqa: E402
import holdout  # noqa: E402
import render_table  # noqa: E402
import run_folo  # noqa: E402
from folo_api import ApiError  # noqa: E402

HAS_PIL = importlib.util.find_spec("PIL") is not None
TOY = datasets.Spec(
    name="toy", title="Toy", domain="test", url="https://example.invalid/toy.zip", sha256="0" * 64, licence="MIT",
    licence_url="", source="", kind="yolo_zip", classes=("signature",), eval_split="val",
)


def make_toy(root: Path, train: int = 5) -> None:
    """A laid-out toy dataset: `train` train images (all but the second labelled) and one val image."""
    from PIL import Image

    for split, n in (("train", train), ("val", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (40, 20), (i * 40, 0, 0)).save(root / "images" / split / f"{i:03d}.jpg")
            if i != 1:
                (root / "labels" / split / f"{i:03d}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (root / "data.yaml").write_text(datasets.data_yaml(list(TOY.classes), ["train", "val"]))
    gt = datasets.coco_ground_truth([("000.jpg", 40, 20, [(0, 0.5, 0.5, 0.5, 0.5)])], list(TOY.classes))
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_val.json").write_text(json.dumps(gt))


class Names(unittest.TestCase):
    def test_the_split_is_the_first_lanes_and_does_not_depend_on_the_input_order(self):
        names = [f"{i:03d}.jpg" for i in range(10)]
        # Pinned: random.Random("toy-0") shuffling the sorted names, the first round(10 * 0.2).
        self.assertEqual(holdout.holdout_names(names, "toy", 0, 0.2), {"006.jpg", "007.jpg"})
        self.assertEqual(holdout.holdout_names(list(reversed(names)), "toy", 0, 0.2), {"006.jpg", "007.jpg"})
        self.assertNotEqual(holdout.holdout_names(names, "toy", 1, 0.2), {"006.jpg", "007.jpg"})
        self.assertEqual(len(holdout.holdout_names(names[:2], "toy", 0, 0.2)), 1, "at least one image")
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                holdout.holdout_names(names, "toy", 0, bad)


@unittest.skipUnless(HAS_PIL, "needs Pillow")
class Carve(unittest.TestCase):
    def test_carve_splits_train_builds_the_ground_truth_and_leaves_the_eval_split_out(self):
        with tempfile.TemporaryDirectory() as t:
            src, dest = Path(t) / "toy", Path(t) / "ho"
            make_toy(src)
            meta = holdout.carve(TOY, src, dest, 0)
            out = dest / "toy-ho0"
            self.assertEqual(meta["holdout"], ["000.jpg"])  # holdout_names of 5 names, seed 0
            self.assertEqual(sorted(p.name for p in (out / "images" / "val").iterdir()), ["000.jpg"])
            self.assertEqual(
                sorted(p.name for p in (out / "images" / "train").iterdir()), ["001.jpg", "002.jpg", "003.jpg", "004.jpg"]
            )
            self.assertEqual(sorted(p.name for p in (out / "labels" / "train").iterdir()), ["002.txt", "003.txt", "004.txt"])
            # The source's val (the evaluation split) is nowhere in the hold-out folder.
            self.assertEqual(sorted(p.name for p in (out / "images").iterdir()), ["train", "val"])
            gt = json.loads((out / "annotations" / "instances_val.json").read_text())
            self.assertEqual([im["file_name"] for im in gt["images"]], ["000.jpg"])
            for got, want in zip(gt["annotations"][0]["bbox"], [10.0, 5.0, 20.0, 10.0]):
                self.assertAlmostEqual(got, want)
            self.assertEqual(meta["splits"], {"train": {"images": 4}, "val": {"images": 1, "boxes": 1}})
            self.assertEqual((meta["source"], meta["seed"], meta["fraction"]), ("toy", 0, 0.2))
            self.assertEqual(meta["source_layout_sha256"], datasets.layout_digest(src))
            self.assertEqual(meta["layout_sha256"], datasets.layout_digest(out))
            self.assertIn("val: images/val", (out / "data.yaml").read_text())
            # Carving again keeps the folder; another carve into it, or a changed file, is refused.
            self.assertEqual(holdout.carve(TOY, src, dest, 0), meta)
            with self.assertRaises(SystemExit):
                holdout.carve(TOY, src, dest, 0, 0.4)
            (out / "labels" / "train" / "002.txt").unlink()  # a hard link: writing it would change the source too
            with self.assertRaises(SystemExit):
                holdout.carve(TOY, src, dest, 0)
            # Another seed is another folder; the same source gives the same digest anywhere.
            self.assertEqual(holdout.carve(TOY, src, dest, 1)["holdout"], ["002.jpg"])
            self.assertEqual(holdout.carve(TOY, src, Path(t) / "elsewhere", 0)["layout_sha256"], meta["layout_sha256"])


class Variant(unittest.TestCase):
    def test_holdout_is_a_variant_from_weights_and_summarized_with_last(self):
        key = common.RunKey("folo", "folo-n", "bccd", "holdout", 2)
        self.assertEqual(key.stem, "folo-n-bccd-holdout-s2")
        self.assertEqual([common.from_weights(v) for v in common.VARIANTS], [True, False, True])

        class A:
            epochs, batch, imgsz, weights, device = 100, 16, 640, "folo-n-coco-v5", "cpu"

        self.assertEqual(run_folo.train_config(key, "ds_1", A, None, "ft-x")["weights"], "folo-n-coco-v5")
        recs = [
            common.result_record(
                common.RunKey("folo", "folo-n", "bccd", "holdout", s), metrics={"ap": 0.6 + s / 100},
                metrics_last={"ap": 0.5 + s / 100} if s < 2 else None, epochs_planned=100,
            )
            for s in range(3)
        ]
        g = common.summarize(recs)["groups"][0]
        self.assertEqual((g["variant"], g["complete"], g["ap"]["n"], g["ap_last"]["n"]), ("holdout", True, 3, 2))
        self.assertAlmostEqual(g["ap_last"]["mean"], 0.505)
        self.assertIsNone(common.result_record(key)["metrics_last"])

    def test_the_published_tables_leave_holdout_runs_out(self):
        from test_finetune import Render, result

        rs = [result("folo", "bccd", "pretrained", 0, 0.5)] + [result("folo", "bccd", "holdout", s, 0.9) for s in range(3)]
        text, summary = render_table.render(rs, Render.META, epochs=100, seeds=3)
        self.assertIn("**folo-n against yolo26n, partial**: 1 of the 7 runs", text)
        self.assertNotIn("0.900", text)
        self.assertIn("holdout", [g["variant"] for g in summary["groups"]], "summary.json keeps them")


class FakeModels:
    """The model calls of run_folo.evaluate: a best model, and its `-last` model when `last` is a file."""

    def __init__(self, best: Path, last: Path | None):
        self.best, self.last = best, last

    def get(self, path):
        if path == "/api/models/m_1":
            return {"id": "m_1", "arch": "folo-n", "path": str(self.best)}
        if path == "/api/models/m_1-last":
            if self.last is None:
                raise ApiError(404, '{"error": {"code": "not_found", "message": "no such model"}}')
            return {"id": "m_1-last", "arch": "folo-n", "path": str(self.last)}
        if path == "/api/system":
            return {"data_dir": "D:/fake-data"}
        if path == "/api/health":
            return {"version": "0-fake"}
        raise AssertionError(f"unexpected GET {path}")


@unittest.skipUnless(HAS_PIL, "needs Pillow")
class Runner(fake_folo.Case):
    def setUp(self):
        super().setUp()
        patch = mock.patch.dict(datasets.SPECS, {"toy": TOY})
        patch.start()
        self.addCleanup(patch.stop)
        make_toy(self.root / "a" / "data" / "toy")

    def test_a_holdout_run_trains_on_the_carved_split_and_scores_the_eval_split(self):
        m = self.matrix(compute=None, device="cpu", holdout_seed=0, holdout_fraction=0.2, folo=sys.executable)
        m.base = {"id": "folo-n-coco-v5", "arch": "folo-n", "sha256": "cd" * 32}
        ho = common.RunKey("folo", "folo-n", "toy", "holdout", 0)
        pre = common.RunKey("folo", "folo-n", "toy", "pretrained", 0)
        m.layouts = {}
        self.assertEqual(m.train_dir(ho), (self.root / "a" / "holdout" / "toy-ho0").resolve())
        self.assertEqual(m.train_dir(pre), (self.root / "a" / "data" / "toy").resolve())
        spec = m.run_spec(ho)
        self.assertEqual((spec["selection"]["seed"], spec["selection"]["fraction"]), (0, 0.2))
        self.assertEqual(spec["weights"]["name"], "folo-n-coco-v2")
        self.assertNotIn("selection", m.run_spec(pre), "the other variants' fingerprints do not change")
        # Each folder is its own dataset on the server.
        self.assertEqual((m.dataset_id(ho), m.dataset_id(pre)), ("ds_ft-toy-ho0", "ds_ft-toy"))

        best, last = self.root / "best.folo", self.root / "last.folo"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        scored = []

        def score(weights, data_dir, split, det_path, gt):
            scored.append((Path(weights).name, Path(data_dir).name, split, [im["file_name"] for im in gt["images"]]))
            ap = 0.7 if Path(weights).name == "best.folo" else 0.6
            return {"map50_95": ap, "map50": 1.0, "device": "cpu"}, {"ap": ap, "ap50": 1.0}

        m.score_weights = score
        run = {"id": "run_1", "model_id": "m_1", "config": {"epochs": 100, "imgsz": 640, "batch": 16}, "epoch": 100}
        m.api = FakeModels(best, last)
        rec = m.evaluate(ho, run, spec["dataset_layout_sha256"])
        # Both checkpoints on the source's evaluation split (its val image), not on the hold-out it chose on.
        self.assertEqual(scored, [("best.folo", "toy", "val", ["000.jpg"]), ("last.folo", "toy", "val", ["000.jpg"])])
        self.assertEqual((rec["metrics"]["ap"], rec["metrics_last"]["ap"]), (0.7, 0.6))
        self.assertEqual(rec["versions"]["selection"], spec["selection"])
        # A server without the run's last.folo: no metrics_last, the best still scored.
        m.api = FakeModels(best, None)
        self.assertIsNone(m.evaluate(pre, run, spec["dataset_layout_sha256"])["metrics_last"])

        # A result file of this split is done; one chosen on another split is refused, not skipped as done.
        results = self.root / "a" / "results"
        results.mkdir(parents=True, exist_ok=True)
        common.write_json_atomic(ho.result_path(results), rec)
        run_folo.check_selection(ho, spec["selection"], results)
        other = dict(spec["selection"], seed=1)
        with self.assertRaises(SystemExit):
            run_folo.check_selection(ho, other, results)


if __name__ == "__main__":
    unittest.main()
