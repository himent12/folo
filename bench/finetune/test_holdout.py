"""Tests of the hold-out selection (holdout.py) and of what run_folo.py does with it: the `holdout` variant, the
dataset it registers, and `last.folo` scored next to `best.folo`.

    python -m unittest discover -s bench/finetune -p "test_*.py"

No GPU or server: run_folo.py runs against fakes, and `folo val` + the scorer are replaced by a recorder. The carving
and runner tests need Pillow (image sizes) and are skipped without it.
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
# Scored on `test`, as BCCD is: its own images, with a box unlike train's.
TOY = datasets.Spec(
    name="toy", title="Toy", domain="test", url="https://example.invalid/toy.zip", sha256="0" * 64, licence="MIT",
    licence_url="", source="", kind="yolo_zip", classes=("signature",), eval_split="test",
)
TEST_BOX = (0, 0.25, 0.25, 0.5, 0.5)  # x 0-20, y 0-10 px of a 40 x 20 image


def make_toy(root: Path, train: int = 5) -> None:
    """A laid-out toy dataset: `train` train images 000.jpg... (all but the second labelled), one val image v00.jpg,
    one test image t00.jpg (the evaluation split) with its ground truth."""
    from PIL import Image

    for split, names in (("train", [f"{i:03d}.jpg" for i in range(train)]), ("val", ["v00.jpg"]), ("test", ["t00.jpg"])):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i, name in enumerate(names):
            Image.new("RGB", (40, 20), (i * 40, 0, 0)).save(root / "images" / split / name)
            if not (split == "train" and i == 1):
                row = TEST_BOX if split == "test" else (0, 0.5, 0.5, 0.5, 0.5)
                (root / "labels" / split / (name[:-4] + ".txt")).write_text(" ".join(str(v) for v in row) + "\n")
    (root / "data.yaml").write_text(datasets.data_yaml(list(TOY.classes), ["train", "val", "test"]))
    gt = datasets.coco_ground_truth([("t00.jpg", 40, 20, [TEST_BOX])], list(TOY.classes))
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_test.json").write_text(json.dumps(gt))


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
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.src, self.dest = Path(t.name) / "toy", Path(t.name) / "ho"
        self.t = Path(t.name)
        make_toy(self.src)

    def test_carve_splits_train_builds_the_ground_truth_and_leaves_the_eval_split_out(self):
        src, dest = self.src, self.dest
        meta = holdout.carve(TOY, src, dest, 0)
        out = dest / "toy-ho0"
        self.assertEqual(meta["holdout"], ["000.jpg"])  # holdout_names of 5 names, seed 0
        self.assertEqual(sorted(p.name for p in (out / "images" / "val").iterdir()), ["000.jpg"])
        self.assertEqual(
            sorted(p.name for p in (out / "images" / "train").iterdir()), ["001.jpg", "002.jpg", "003.jpg", "004.jpg"]
        )
        self.assertEqual(sorted(p.name for p in (out / "labels" / "train").iterdir()), ["002.txt", "003.txt", "004.txt"])
        # Neither the source's val nor its test (the evaluation split) is in the hold-out folder.
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
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["toy-ho0"], "no staging folder left")
        # Carving again keeps the folder; another carve into it, or a changed file, is refused.
        self.assertEqual(holdout.carve(TOY, src, dest, 0), meta)
        with self.assertRaises(SystemExit):
            holdout.carve(TOY, src, dest, 0, 0.4)
        (out / "labels" / "train" / "002.txt").unlink()  # a hard link: writing it would change the source too
        with self.assertRaises(SystemExit):
            holdout.carve(TOY, src, dest, 0)
        # Another seed is another folder; the same source gives the same digest anywhere.
        self.assertEqual(holdout.carve(TOY, src, dest, 1)["holdout"], ["002.jpg"])
        self.assertEqual(holdout.carve(TOY, src, self.t / "elsewhere", 0)["layout_sha256"], meta["layout_sha256"])

    def test_an_interrupted_carve_is_never_taken_for_one(self):
        # What a carve cut short leaves (its own staging folder, with links of another fraction) is never read.
        partial = self.dest / ".toy-ho0.partial-1-dead"
        (partial / "images" / "train").mkdir(parents=True)
        (partial / "images" / "train" / "000.jpg").write_bytes(b"stale")
        meta = holdout.carve(TOY, self.src, self.dest, 0)
        self.assertTrue(partial.exists(), "left alone: it may be another carve's, still running")
        self.assertEqual(meta["layout_sha256"], datasets.layout_digest(self.dest / "toy-ho0"))
        self.assertNotIn("000.jpg", [p.name for p in (self.dest / "toy-ho0" / "images" / "train").iterdir()])
        # A folder in the carve's place without meta.json is refused, not completed or blessed.
        bare = self.t / "bare"
        (bare / "toy-ho0" / "images" / "train").mkdir(parents=True)
        with self.assertRaises(SystemExit):
            holdout.carve(TOY, self.src, bare, 0)

    def test_of_two_carves_at_once_the_second_checks_the_first(self):
        real = holdout.os.rename

        def race(fraction, dest):
            # Another carve finishes just before this one renames its staging folder into place.
            def rename(a, b):
                holdout.os.rename = real
                holdout.carve(TOY, self.src, dest, 0, fraction)
                return real(a, b)
            return rename

        with mock.patch.object(holdout.os, "rename", race(0.2, self.dest)):
            meta = holdout.carve(TOY, self.src, self.dest, 0)
        self.assertEqual(meta["layout_sha256"], datasets.layout_digest(self.dest / "toy-ho0"), "the same carve: kept")
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()), ["toy-ho0"], "its own staging folder removed")
        other = self.t / "other"
        real = holdout.os.rename
        with mock.patch.object(holdout.os, "rename", race(0.4, other)), self.assertRaises(SystemExit):
            holdout.carve(TOY, self.src, other, 0)  # the first published another fraction: refused, not mixed
        self.assertEqual(json.loads((other / "toy-ho0" / "meta.json").read_text())["fraction"], 0.4)

    def test_a_holdout_that_leaves_nothing_to_train_on_is_refused_before_any_file(self):
        with self.assertRaises(SystemExit):
            holdout.carve(TOY, self.src, self.dest, 0, 0.999)  # round(5 * 0.999) = all 5 images
        self.assertFalse(self.dest.exists())


class Variant(unittest.TestCase):
    def test_holdout_is_a_variant_from_weights_and_summarized_with_last(self):
        key = common.RunKey("folo", "folo-n", "bccd", "holdout", 2, holdout.split_label(0))
        self.assertEqual(key.stem, "folo-n-bccd-holdout-ho0-s2")
        self.assertEqual(holdout.split_label(1, 0.3), "ho1-f0.3")
        for bad in (("holdout", ""), ("pretrained", "ho0")):
            with self.assertRaises(ValueError):
                common.RunKey("folo", "folo-n", "bccd", bad[0], 0, bad[1])
        self.assertEqual([common.from_weights(v) for v in common.VARIANTS], [True, False, True])

        class A:
            epochs, batch, imgsz, weights, device = 100, 16, 640, "folo-n-coco-v5", "cpu"

        self.assertEqual(run_folo.train_config(key, "ds_1", A, None, "ft-x")["weights"], "folo-n-coco-v5")
        sel = {"seed": 0, "fraction": 0.2, "layout_sha256": "ab"}
        recs = [
            common.result_record(
                common.RunKey("folo", "folo-n", "bccd", "holdout", s, "ho0"), metrics={"ap": 0.6 + s / 100},
                metrics_last={"ap": 0.5 + s / 100} if s < 2 else None, epochs_planned=100,
                versions={"selection": sel},
            )
            for s in range(3)
        ]
        [g] = common.summarize(recs)["groups"]
        self.assertEqual((g["variant"], g["complete"], g["ap"]["n"], g["ap_last"]["n"]), ("holdout", True, 3, 2))
        self.assertEqual(g["selection"], sel)
        self.assertAlmostEqual(g["ap_last"]["mean"], 0.505)
        self.assertIsNone(common.result_record(key)["metrics_last"])
        # Seeds of two hold-out splits are two groups, neither complete, not one group of three.
        recs[2]["versions"] = {"selection": dict(sel, seed=1)}
        groups = common.summarize(recs)["groups"]
        self.assertEqual(sorted((g["selection"]["seed"], g["seeds"], g["complete"]) for g in groups),
                         [(0, [0, 1], False), (1, [2], False)])
        # So are seeds of one seed and fraction carved from different files (another layout digest).
        recs[2]["versions"] = {"selection": dict(sel, layout_sha256="cd")}
        self.assertEqual(len(common.summarize(recs)["groups"]), 2)

    def test_the_published_tables_leave_holdout_runs_out(self):
        from test_finetune import Render, result

        held = [
            dict(result("folo", "bccd", "pretrained", s, 0.9), variant="holdout", versions={"selection": {"seed": 0}})
            for s in range(3)
        ]
        held[0]["gpu"] = "a CPU"
        held[1]["epochs_planned"] = 3
        rs = [result("folo", "bccd", "pretrained", 0, 0.5)] + held
        text, summary = render_table.render(rs, Render.META, epochs=100, seeds=3)
        self.assertIn("**folo-n against yolo26n, partial**: 1 of the 7 runs", text)
        self.assertNotIn("0.900", text)
        self.assertNotIn("a CPU", text, "the GPUs line names only the published runs' GPUs")
        self.assertNotIn("another epoch count", text, "a holdout run of another length is not counted as left out")
        self.assertEqual(summary["ignored_other_epochs"], 0)
        self.assertIn("holdout", [g["variant"] for g in summary["groups"]], "summary.json keeps them")


class FakeModels:
    """The model calls of run_folo.evaluate: a best model, and its `-last` model when `last` is a file (or the
    error `last_error` answers)."""

    def __init__(self, best: Path, last: Path | None, last_error: ApiError | None = None):
        self.best, self.last, self.last_error = best, last, last_error

    def get(self, path):
        if path == "/api/models/m_1":
            return {"id": "m_1", "arch": "folo-n", "path": str(self.best)}
        if path == "/api/models/m_1-last":
            if self.last_error is not None:
                raise self.last_error
            if self.last is None:
                raise ApiError(404, '{"error": {"code": "not_found", "message": "no such model"}}')
            return {"id": "m_1-last", "arch": "folo-n", "path": str(self.last)}
        if path == "/api/system":
            return {"data_dir": "D:/fake-data"}
        if path == "/api/health":
            return {"version": "0-fake"}
        raise AssertionError(f"unexpected GET {path}")


class Recording(fake_folo.FakeServer):
    """The fake server, keeping every dataset registration and run creation it is sent."""

    def __init__(self):
        super().__init__()
        self.registered: list[dict] = []
        self.created: list[dict] = []

    def call(self, method, path, body=None, **kw):
        if method == "POST" and path == "/api/datasets":
            self.registered.append(body)
            return {"id": f"ds_{len(self.registered)}", "classes": ["signature"]}
        if method == "POST" and path == "/api/runs":
            self.created.append(body)
        return super().call(method, path, body, **kw)


@unittest.skipUnless(HAS_PIL, "needs Pillow")
class Runner(fake_folo.Case):
    def setUp(self):
        super().setUp()
        patch = mock.patch.dict(datasets.SPECS, {"toy": TOY})
        patch.start()
        self.addCleanup(patch.stop)
        make_toy(self.root / "a" / "data" / "toy")
        self.server = Recording()

    def matrix_toy(self):
        m = self.matrix(compute=None, device="cpu", holdout_seed=0, holdout_fraction=0.2, folo=sys.executable)
        m.base = {"id": "m_base", "name": "folo-n-coco-v2", "arch": "folo-n", "sha256": "cd" * 32}
        m.layouts = {}
        return m

    def test_a_holdout_run_trains_on_the_carved_split_and_scores_the_eval_split(self):
        m = self.matrix_toy()
        ho = common.RunKey("folo", "folo-n", "toy", "holdout", 0, "ho0")
        pre = common.RunKey("folo", "folo-n", "toy", "pretrained", 0)
        carved = (self.root / "a" / "holdout" / "toy-ho0").resolve()
        self.assertEqual(m.train_dir(ho), carved)
        self.assertEqual(m.train_dir(pre), (self.root / "a" / "data" / "toy").resolve())
        spec = m.run_spec(ho)
        self.assertEqual((spec["selection"]["seed"], spec["selection"]["fraction"]), (0, 0.2))
        self.assertEqual(spec["selection"]["layout_sha256"], datasets.layout_digest(carved))
        self.assertEqual(spec["weights"]["name"], "folo-n-coco-v2")
        self.assertNotIn("selection", m.run_spec(pre), "the other variants' fingerprints do not change")

        # The run is created on the carved folder, registered by its path, from the weights.
        run = m.launch(ho, *fake_folo.identity(m, ho))
        [reg] = self.server.registered
        self.assertEqual((reg["name"], Path(reg["path"])), ("ft-toy-ho0", carved))
        [created] = self.server.created
        self.assertEqual((created["dataset_id"], created["weights"]), ("ds_1", "m_base"))
        self.assertEqual(m.dataset_id(pre), "ds_2", "the source is another dataset on the server")
        self.assertEqual(Path(self.server.registered[1]["path"]), (self.root / "a" / "data" / "toy").resolve())

        best, last = self.root / "best.folo", self.root / "last.folo"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        scored = []

        def score(weights, data_dir, split, det_path, gt):
            scored.append((Path(weights).name, Path(data_dir), split, [im["file_name"] for im in gt["images"]],
                           [a["bbox"] for a in gt["annotations"]]))
            ap = 0.7 if Path(weights).name == "best.folo" else 0.6
            return {"map50_95": ap, "map50": 1.0, "device": "cpu"}, {"ap": ap, "ap50": 1.0}

        m.score_weights = score
        run = dict(run, model_id="m_1", config={"epochs": 100, "imgsz": 640, "batch": 16}, epoch=100)
        m.api = FakeModels(best, last)
        rec = m.evaluate(ho, run, spec["dataset_layout_sha256"])
        # Both checkpoints on the source's evaluation split (its test image and box), not on the hold-out.
        src = (self.root / "a" / "data" / "toy").resolve()
        want = (src, "test", ["t00.jpg"], [[0.0, 0.0, 20.0, 10.0]])
        self.assertEqual([s[1:] for s in scored], [want, want])
        self.assertEqual([s[0] for s in scored], ["best.folo", "last.folo"])
        self.assertEqual((rec["metrics"]["ap"], rec["metrics_last"]["ap"]), (0.7, 0.6))
        self.assertEqual(rec["versions"]["selection"], spec["selection"])
        # A server without the run's last.folo: no metrics_last, the best still scored; any other error fails.
        m.api = FakeModels(best, None)
        self.assertIsNone(m.evaluate(pre, run, spec["dataset_layout_sha256"])["metrics_last"])
        m.api = FakeModels(best, last, ApiError(500, '{"error": {"code": "internal"}}'))
        with self.assertRaises(ApiError):
            m.evaluate(pre, run, spec["dataset_layout_sha256"])
        # A run whose best model is its last.folo already (resumed, never improved on the checkpoint): scored once.
        scored.clear()
        m.api = FakeModels(last, None)
        m.api.get = lambda path: {"id": "m_1-last", "arch": "folo-n", "path": str(last)}
        rec = m.evaluate(pre, dict(run, model_id="m_1-last"), spec["dataset_layout_sha256"])
        self.assertEqual([s[0] for s in scored], ["last.folo"])
        self.assertEqual(rec["metrics_last"], rec["metrics"])

    def test_result_files_that_cannot_count_as_done_are_refused_before_anything_starts(self):
        m = self.matrix_toy()
        keys = [common.RunKey("folo", "folo-n", "toy", "holdout", s, "ho0") for s in (0, 1)]
        results = Path(m.args.results)
        results.mkdir(parents=True, exist_ok=True)
        same = m.run_spec(keys[0])["selection"]
        path = keys[0].result_path(results)
        self.assertEqual(path.name, "folo-n-toy-holdout-ho0-s0.json", "the split is in the file name")
        ok = common.result_record(keys[0], metrics={"ap": 0.5}, versions={"selection": same})
        # Of this split but carved from other files (another layout digest), or not a readable result: refused.
        for bad in (dict(ok, versions={"selection": dict(same, layout_sha256="00")}), "{", {"version": 0}):
            path.write_text(bad if isinstance(bad, str) else json.dumps(bad))
            with self.assertRaises(SystemExit):
                run_folo.run_matrix(m, keys, ["holdout"])
            self.assertEqual((self.server.registered, self.server.created), ([], []))
        # The same split's result: done (only seed 1 is left to run).
        common.write_json_atomic(path, ok)
        run_folo.check_results(keys, {k.stem: same for k in keys}, results)


if __name__ == "__main__":
    unittest.main()
