"""Bookkeeping tests of the fine-tuning harness: labels, ground truth, id mapping, results, tables.

    python -m unittest discover -s bench/finetune -p "test_*.py"

No GPU, network, torch or Ultralytics needed; the scorer tests are skipped without numpy and pycocotools, the layout
tests without Pillow. test_run_folo.py covers run_folo.py against a fake server.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import datasets  # noqa: E402
import render_table  # noqa: E402
import run_folo  # noqa: E402
import run_ultralytics  # noqa: E402
import scorer  # noqa: E402

HAS_PIL = importlib.util.find_spec("PIL") is not None
# Imported, not only found: pycocotools on the path without numpy (another environment's site-packages) cannot score.
COCO_MISSING = scorer.missing()


class Labels(unittest.TestCase):
    def test_parse_drops_and_counts_problems(self):
        text = "\n".join(
            [
                "0 0.5 0.5 0.2 0.2",
                "0 0.5 0.5 0.2 0.2",  # exact repeat
                "3 0.5 0.5 0.2 0.2",  # class out of range for nc=2
                "1 0.5 0.5 0.2",  # 4 numbers
                "0.5 0.5 0.5 0.2 0.2",  # non-integer class
                "1 0.95 0.5 0.2 0.2",  # sticks out on the right: clipped to x 0.85..1
                "1 1.2 0.5 0.1 0.1",  # entirely outside: empty
                "",
            ]
        )
        rows, fixes = datasets.parse_yolo_label(text, nc=2)
        self.assertEqual(fixes, {"bad_rows": 3, "clipped": 1, "empty": 1, "duplicates": 1})
        self.assertEqual(rows[0], (0, 0.5, 0.5, 0.2, 0.2))
        c, cx, cy, w, h = rows[1]
        self.assertEqual(c, 1)
        self.assertAlmostEqual(cx - w / 2, 0.85, places=6)
        self.assertAlmostEqual(cx + w / 2, 1.0, places=6)

    def test_six_decimal_edge_box_is_not_clipped(self):
        # A box touching the left edge, written with 6 decimals, starts 5e-7 outside: that is rounding, kept as is.
        rows, fixes = datasets.parse_yolo_label("0 0.060937 0.379167 0.121875 0.237500\n", nc=1)
        self.assertEqual(fixes["clipped"], 0)
        self.assertEqual(rows, [(0, 0.060937, 0.379167, 0.121875, 0.2375)])

    def test_format_then_parse_round_trips(self):
        rows = [(0, 0.1, 0.2, 0.05, 0.1), (2, 0.223457, 0.5, 0.3, 0.25)]
        back, fixes = datasets.parse_yolo_label(datasets.format_label(rows), nc=3)
        self.assertEqual(back, rows)
        self.assertFalse(any(fixes.values()))

    def test_voc_conversion_matches_ultralytics_voc(self):
        # Ultralytics VOC.yaml: centre (xmin + xmax) / 2 - 1, size xmax - xmin, normalised by the image size.
        cx, cy, w, h = datasets.voc_to_yolo(11, 21, 111, 71, 200, 100)
        self.assertAlmostEqual(cx, 60 / 200)
        self.assertAlmostEqual(cy, 45 / 100)
        self.assertAlmostEqual(w, 100 / 200)
        self.assertAlmostEqual(h, 50 / 100)
        self.assertIsNone(datasets.voc_to_yolo(50, 10, 40, 20, 100, 100))

    def test_ground_truth_is_in_pixels_with_ids_from_one(self):
        gt = datasets.coco_ground_truth(
            [("a.jpg", 200, 100, [(1, 0.5, 0.5, 0.5, 0.2)]), ("b.jpg", 10, 10, [])], ["x", "y"]
        )
        self.assertEqual([im["id"] for im in gt["images"]], [1, 2])
        self.assertEqual(gt["categories"], [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}])
        (a,) = gt["annotations"]
        self.assertEqual(a["image_id"], 1)
        self.assertEqual(a["category_id"], 2)
        for got, want in zip(a["bbox"], [50.0, 40.0, 100.0, 20.0]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(a["area"], 2000.0)
        self.assertEqual(a["iscrowd"], 0)

    def test_data_yaml_has_no_path_key(self):
        y = datasets.data_yaml(["a b", 'q"x'], ["train", "val"])
        self.assertNotIn("path:", y)
        self.assertIn("train: images/train", y)
        self.assertIn('names: ["a b", "q\\"x"]', y)
        self.assertIn("nc: 2", y)

    def test_specs_are_complete(self):
        for s in datasets.SPECS.values():
            self.assertEqual(len(s.sha256), 64, s.name)
            self.assertEqual(len(s.layout_sha256), 64, s.name)
            self.assertIn(s.eval_split, ("val", "test"))
            self.assertTrue(s.url.startswith("https://"))
        with self.assertRaises(SystemExit):
            datasets.pick("bccd,nope")
        self.assertEqual([s.name for s in datasets.pick("signature, bccd")], ["signature", "bccd"])


@unittest.skipUnless(HAS_PIL, "needs Pillow")
class Layout(unittest.TestCase):
    def make_src(self, root: Path):
        from PIL import Image

        for split, n in (("train", 2), ("val", 1)):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(n):
                Image.new("RGB", (40, 20), (i * 50, 0, 0)).save(root / "images" / split / f"{i:04d}.jpg")
                if i == 0:
                    (root / "labels" / split / f"{i:04d}.txt").write_text("0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0.5 0.5\n")

    def test_lay_out_writes_the_canonical_layout_deterministically(self):
        spec = datasets.Spec(
            name="toy", title="Toy", domain="test", url="https://example.invalid/toy.zip", sha256="0" * 64,
            licence="MIT", licence_url="", source="", kind="yolo_zip", classes=("thing",), eval_split="val",
        )
        with tempfile.TemporaryDirectory() as t:
            src, a, b = Path(t) / "src", Path(t) / "a", Path(t) / "b"
            self.make_src(src)
            meta = datasets.lay_out(spec, src, a)
            meta_b = datasets.lay_out(spec, src, b)
            self.assertEqual(meta["layout_sha256"], meta_b["layout_sha256"])
            self.assertEqual(meta["splits"]["train"]["images"], 2)
            self.assertEqual(meta["splits"]["train"]["boxes"], 1)
            self.assertEqual(meta["splits"]["train"]["background_images"], 1)
            self.assertEqual(meta["fixes"], {"duplicates": 2})
            gt = json.loads((a / "annotations" / "instances_val.json").read_text())
            self.assertEqual([im["file_name"] for im in gt["images"]], ["0000.jpg"])
            for got, want in zip(gt["annotations"][0]["bbox"], [10.0, 5.0, 20.0, 10.0]):
                self.assertAlmostEqual(got, want)
            self.assertEqual((a / "labels" / "train" / "0001.txt").read_text(), "")
            # Any change to a file the tools read changes the digest.
            (a / "labels" / "train" / "0001.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            self.assertNotEqual(datasets.layout_digest(a), meta["layout_sha256"])
            # Ultralytics' label cache next to the labels is not part of it.
            (b / "labels" / "train.cache").write_bytes(b"derived")
            self.assertEqual(datasets.layout_digest(b), meta["layout_sha256"])

    def test_a_tampered_cached_dataset_is_laid_out_again_and_refused_by_the_runners(self):
        import zipfile

        with tempfile.TemporaryDirectory() as t:
            src, root = Path(t) / "src", Path(t) / "root"
            self.make_src(src)
            archive = root / "_downloads" / "toy.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as z:
                for f in sorted(src.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(src).as_posix())
            spec = datasets.Spec(
                name="toy", title="Toy", domain="test", url="https://example.invalid/toy.zip",
                sha256=datasets.sha256_file(archive), licence="MIT", licence_url="", source="", kind="yolo_zip",
                classes=("thing",), eval_split="val",
            )
            spec = dataclasses.replace(spec, layout_sha256=datasets.fetch(spec, root)["layout_sha256"])
            dest = root / "toy"
            self.assertEqual(datasets.verify_layout(spec, dest), spec.layout_sha256)
            label = dest / "labels" / "train" / "0000.txt"
            good = label.read_text()
            label.write_text("0 0.1 0.1 0.1 0.1\n")  # meta.json still names the recorded digest
            with self.assertRaises(SystemExit) as e:
                datasets.verify_layout(spec, dest)  # what run_folo.py / run_ultralytics.py check before a run
            self.assertIn("changed", str(e.exception))
            meta = datasets.fetch(spec, root)  # not taken from meta.json: hashed, found changed, laid out again
            self.assertEqual(label.read_text(), good)
            self.assertEqual(meta["layout_sha256"], spec.layout_sha256)
            self.assertEqual(datasets.verify_layout(spec, dest), spec.layout_sha256)
            (dest / "images" / "val" / "0000.jpg").unlink()  # a deleted image too
            with self.assertRaises(SystemExit):
                datasets.verify_layout(spec, dest)
            datasets.fetch(spec, root)
            self.assertTrue((dest / "images" / "val" / "0000.jpg").is_file())


FAKE_ULTRALYTICS = '''
import os
from pathlib import Path
from types import SimpleNamespace

__version__ = "0-fake"


class YOLO:
    def __init__(self, path):
        self.path = path

    def train(self, data, project, name, **kw):
        if os.environ.get("FAKE_TAMPER") == "1":  # the dataset changes while the run trains
            (Path(data).parent / "labels" / "train" / "0000.txt").write_text("0 0.2 0.2 0.1 0.1\\n")
        save = Path(project) / name
        (save / "weights").mkdir(parents=True, exist_ok=True)
        (save / "weights" / "best.pt").write_bytes(b"fake")
        self.trainer = SimpleNamespace(save_dir=str(save), amp=False)

    def add_callback(self, *a):
        pass

    def val(self, **kw):
        raise SystemExit("fake ultralytics: validation reached")
'''

FAKE_TORCH = '''
__version__ = "0-fake"


class cuda:
    is_available = staticmethod(lambda: False)


class version:
    cuda = None
'''

ULTRALYTICS_DRIVER = '''
import sys
from argparse import Namespace
here, fake, root, work, results, weights, digest = sys.argv[1:]
sys.path[:0] = [fake, here]
import datasets
datasets.SPECS["toy"] = datasets.Spec(
    name="toy", title="Toy", domain="test", url="https://example.invalid/toy.zip", sha256="0" * 64, licence="MIT",
    licence_url="", source="", kind="yolo_zip", classes=("thing",), eval_split="val", layout_sha256=digest,
)
import run_ultralytics
run_ultralytics.run_one(Namespace(
    weights=weights, dataset="toy", seed=0, data_root=root, work_dir=work, results=results, epochs=1, imgsz=64,
    batch=1, device="cpu", workers=None, compute_label=None, jobs=1,
))
'''


@unittest.skipUnless(HAS_PIL, "needs Pillow")
class UltralyticsRunner(unittest.TestCase):
    """run_ultralytics.run_one against a fake torch and ultralytics (in a subprocess: it must import them fresh)."""

    def run_one(self, tamper: bool):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            fake = t / "fake"
            (fake / "ultralytics" / "utils").mkdir(parents=True)
            (fake / "ultralytics" / "__init__.py").write_text(FAKE_ULTRALYTICS)
            (fake / "ultralytics" / "utils" / "__init__.py").write_text("SETTINGS = {}\n")
            (fake / "torch.py").write_text(FAKE_TORCH)
            spec = datasets.Spec(
                name="toy", title="Toy", domain="test", url="https://example.invalid/toy.zip", sha256="0" * 64,
                licence="MIT", licence_url="", source="", kind="yolo_zip", classes=("thing",), eval_split="val",
            )
            Layout.make_src(None, t / "src")
            digest = datasets.lay_out(spec, t / "src", t / "data" / "toy")["layout_sha256"]
            (t / "yolo26s.pt").write_bytes(b"weights")
            driver = t / "driver.py"
            driver.write_text(ULTRALYTICS_DRIVER)
            argv = [str(HERE), str(fake), str(t / "data"), str(t / "work"), str(t / "results"), str(t / "yolo26s.pt"), digest]
            env = dict(os.environ, FAKE_TAMPER="1" if tamper else "0")
            env.pop("PYTHONPATH", None)
            p = subprocess.run([sys.executable, str(driver), *argv], capture_output=True, text=True, env=env, timeout=60)
            return p, sorted((t / "results").glob("*.json")) if (t / "results").is_dir() else []

    def test_a_dataset_changed_during_training_is_refused_before_validation(self):
        p, results = self.run_one(tamper=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("changed since they were laid out", p.stderr)
        self.assertNotIn("validation reached", p.stderr)
        self.assertEqual(results, [])
        p, _ = self.run_one(tamper=False)  # the control: unchanged files go on to validation
        self.assertIn("validation reached", p.stderr)


class Scorer(unittest.TestCase):
    GT = {
        "images": [
            {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "cat.png", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    }

    def test_ultralytics_records_map_by_file_name_not_stem(self):
        # Ultralytics turns the stem 0001 into image_id 1 and a non-numeric stem into a string; neither is trusted.
        recs = [
            {"image_id": 1, "file_name": "0001.jpg", "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": "cat", "file_name": "cat.png", "category_id": 2, "bbox": [50, 50, 30, 30], "score": 0.8},
        ]
        out = scorer.ultralytics_to_gt(recs, self.GT, class_map=[1, 2])
        self.assertEqual([(d["image_id"], d["category_id"]) for d in out], [(1, 1), (2, 2)])

    def test_coco91_class_map_goes_through_the_class_index(self):
        gt = dict(self.GT, categories=[{"id": 7, "name": "a"}, {"id": 9, "name": "b"}])
        recs = [{"image_id": 1, "file_name": "0001.jpg", "category_id": 3, "bbox": [0, 0, 1, 1], "score": 0.5}]
        out = scorer.ultralytics_to_gt(recs, gt, class_map=[1, 3])  # class 1 written as id 3
        self.assertEqual(out[0]["category_id"], 9)

    def test_mapping_errors_are_loud(self):
        bad_file = [{"image_id": 5, "file_name": "nope.jpg", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}]
        with self.assertRaises(ValueError):
            scorer.ultralytics_to_gt(bad_file, self.GT, [1, 2])
        bad_cat = [{"image_id": 1, "file_name": "0001.jpg", "category_id": 80, "bbox": [0, 0, 1, 1], "score": 0.5}]
        with self.assertRaises(ValueError):
            scorer.ultralytics_to_gt(bad_cat, self.GT, [1, 2])
        with self.assertRaises(ValueError):
            scorer.check_detections(self.GT, [{"image_id": 3, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1}])
        with self.assertRaises(ValueError):
            scorer.check_detections(self.GT, [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1], "score": 1}])

    def test_inverted_boxes_become_empty_false_positives(self):
        dets = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, -5, 20], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.5},
        ]
        out, n = scorer.degenerate_to_empty(dets)
        self.assertEqual(n, 1)
        self.assertEqual(out[0]["bbox"], [10, 10, 0.0, 20])
        self.assertEqual((out[0]["score"], out[1]), (0.9, dets[1]))
        self.assertEqual(dets[0]["bbox"], [10, 10, -5, 20])  # the input is not changed

    @unittest.skipIf(COCO_MISSING, f"needs numpy and pycocotools ({COCO_MISSING})")
    def test_inverted_box_counts_as_a_false_positive(self):
        dets = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, -20, 20], "score": 0.9},  # ranks first, matches nothing
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.5},
            {"image_id": 2, "category_id": 2, "bbox": [50, 50, 30, 30], "score": 0.8},
        ]
        s = scorer.score(self.GT, dets)
        self.assertEqual(s["inverted_boxes"], 1)
        self.assertLess(s["ap"], 1.0)
        self.assertGreater(s["ap"], 0.5)

    @unittest.skipIf(COCO_MISSING, f"needs numpy and pycocotools ({COCO_MISSING})")
    def test_score_perfect_half_and_empty(self):
        perfect = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": 2, "category_id": 2, "bbox": [50, 50, 30, 30], "score": 0.8},
        ]
        s = scorer.score(self.GT, perfect)
        self.assertAlmostEqual(s["ap"], 1.0)
        self.assertAlmostEqual(s["ap50"], 1.0)
        self.assertEqual(s["images_without_detections"], 0)
        half = scorer.score(self.GT, perfect[:1])
        self.assertAlmostEqual(half["ap"], 0.5)  # category b has no detection: AP 0, mean over a and b
        self.assertEqual(half["images_without_detections"], 1)
        empty = scorer.score(self.GT, [])
        self.assertEqual((empty["ap"], empty["detections"]), (0.0, 0))


def result(tool, dataset, variant, seed, ap, epochs=100, wall=60.0, model=None):
    model = model or {"folo": "folo-n", "ultralytics": "yolo26n"}[tool]
    key = common.RunKey(tool, model, dataset, variant, seed)
    return common.result_record(
        key, epochs_planned=epochs, metrics={"ap": ap, "ap50": ap + 0.2, "ap75": ap}, train_wall_s=wall, gpu="RTX 4090"
    )


class Results(unittest.TestCase):
    def test_run_key(self):
        k = common.RunKey("folo", "folo-n", "bccd", "pretrained", 2)
        self.assertEqual(k.stem, "folo-n-bccd-pretrained-s2")
        self.assertEqual(k.result_path(Path("r")), Path("r") / "folo-n-bccd-pretrained-s2.json")
        self.assertEqual(common.RunKey("ultralytics", "yolo26s", "bccd", "pretrained", 0).stem, "yolo26s-bccd-pretrained-s0")
        with self.assertRaises(ValueError):
            common.RunKey("yolo", "yolo26n", "bccd", "pretrained", 0)
        with self.assertRaises(ValueError):
            common.RunKey("folo", "folo-n", "bccd", "finetuned", 0)
        with self.assertRaises(ValueError):
            common.RunKey("folo", "yolo26n", "bccd", "pretrained", 0)  # model of the other tool
        with self.assertRaises(ValueError):
            common.RunKey("ultralytics", "folo-s", "bccd", "pretrained", 0)
        self.assertEqual([common.size_of(m) for m in ("folo-n", "yolo26n", "folo-s", "yolo26s", "rtdetr")], ["n", "n", "s", "s", "rtdetr"])
        with self.assertRaises(ValueError):
            common.result_record(k, colour="red")

    def test_parse_seeds(self):
        self.assertEqual(common.parse_seeds("0,1,2"), [0, 1, 2])
        self.assertEqual(common.parse_seeds("0-2,5,1"), [0, 1, 2, 5])
        with self.assertRaises(ValueError):
            common.parse_seeds("")

    def test_summarize_groups_and_marks_partial(self):
        rs = [
            result("folo", "bccd", "pretrained", 0, 0.5),
            result("folo", "bccd", "pretrained", 1, 0.6),
            result("ultralytics", "bccd", "pretrained", 0, 0.4),
            result("ultralytics", "bccd", "pretrained", 1, 0.5),
            result("ultralytics", "bccd", "pretrained", 2, 0.6),
            result("folo", "bccd", "scratch", 0, 0.1),
        ]
        s = common.summarize(rs, expected_seeds=3)
        g = {(x["tool"], x["variant"]): x for x in s["groups"]}
        self.assertFalse(g[("folo", "pretrained")]["complete"])
        self.assertTrue(g[("ultralytics", "pretrained")]["complete"])
        self.assertTrue(g[("folo", "scratch")]["complete"])
        u = g[("ultralytics", "pretrained")]["ap"]
        self.assertAlmostEqual(u["mean"], 0.5)
        self.assertEqual((u["min"], u["max"], u["n"]), (0.4, 0.6, 3))
        self.assertEqual(g[("folo", "pretrained")]["seeds"], [0, 1])

    def test_shared_gpu_times_are_kept_apart(self):
        alone = result("ultralytics", "bccd", "pretrained", 0, 0.5, wall=100.0)
        alone["compute"] = {"concurrent_jobs": 1}
        a = result("ultralytics", "bccd", "pretrained", 1, 0.5, wall=180.0)
        b = result("ultralytics", "bccd", "pretrained", 2, 0.5, wall=190.0)
        a["compute"] = b["compute"] = {"concurrent_jobs": 2}
        (g,) = common.summarize([alone, a, b])["groups"]
        self.assertEqual((g["train_wall_s"]["mean"], g["train_wall_s"]["n"]), (100.0, 1))
        self.assertEqual((g["train_wall_shared_s"]["mean"], g["train_wall_shared_s"]["n"]), (185.0, 2))
        self.assertEqual(g["ap"]["n"], 3)  # accuracy counts every seed

    def test_load_results_skips_other_json(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            r = result("folo", "bccd", "pretrained", 0, 0.5)
            common.write_json_atomic(t / "folo-bccd-pretrained-s0.json", r)
            (t / "summary.json").write_text(json.dumps({"version": 1, "groups": []}))
            (t / "crosscheck-coco.json").write_text(json.dumps({"folo": {}}))
            (t / "broken.json").write_text("{")
            self.assertEqual(common.load_results(t), [r])


class Render(unittest.TestCase):
    META = [
        {
            "name": "bccd", "title": "BCCD", "source": "https://x", "domain": "microscopy", "licence": "MIT",
            "classes": ["a", "b", "c"], "eval_split": "test",
            "splits": {"train": {"images": 205, "boxes": 2804}, "test": {"images": 72, "boxes": 945}},
        }
    ]

    def test_no_results_renders_an_empty_table_without_numbers(self):
        text, summary = render_table.render([], self.META, epochs=100, seeds=3)
        self.assertIn("folo-n against yolo26n: no 100-epoch run has been measured yet", text)
        self.assertNotIn("folo-s", text)  # no folo-s table without folo-s results
        self.assertIn("| bccd | not run | not run | not run |", text)
        self.assertEqual(summary["runs"], 0)

    def test_partial_results_and_other_epochs(self):
        rs = [
            result("folo", "bccd", "pretrained", 0, 0.5),
            result("folo", "bccd", "pretrained", 1, 0.7),
            result("ultralytics", "bccd", "pretrained", 0, 0.4),
            result("folo", "bccd", "pretrained", 2, 0.9, epochs=3),  # a smoke run: left out
        ]
        text, summary = render_table.render(rs, self.META, epochs=100, seeds=3)
        self.assertIn("**folo-n against yolo26n, partial**: 3 of the 7 runs", text)
        self.assertIn("1 result files of runs with another epoch count", text)
        self.assertIn("0.600 (0.500–0.700), 2 of 3 seeds", text)
        self.assertIn("0.400, 1 of 3 seeds", text)
        self.assertIn("+0.200", text)
        self.assertEqual(summary["ignored_other_epochs"], 1)

    def test_each_size_gets_its_own_table(self):
        rs = [
            result("folo", "bccd", "pretrained", 0, 0.5),
            result("folo", "bccd", "pretrained", 0, 0.6, model="folo-s"),
            result("ultralytics", "bccd", "pretrained", 0, 0.7, model="yolo26s"),
        ]
        text, summary = render_table.render(rs, self.META, epochs=100, seeds=3)
        self.assertIn("**folo-n against yolo26n, partial**: 1 of the 7 runs", text)
        self.assertIn("**folo-s against yolo26s, partial**: 2 of the 7 runs", text)
        self.assertIn("| folo-s fine-tuned AP | AP50 | yolo26s fine-tuned AP |", text)
        self.assertIn("-0.100", text)  # folo-s 0.6 against yolo26s 0.7, not against folo-n's row
        self.assertLess(text.index("folo-n against"), text.index("folo-s against"))
        self.assertEqual(len(summary["groups"]), 3)

    def test_update_replaces_between_markers_only(self):
        with tempfile.TemporaryDirectory() as t:
            doc = Path(t) / "B.md"
            doc.write_text(f"head\n{render_table.BEGIN}\nold\n{render_table.END}\ntail\n")
            render_table.update(doc, "new\n")
            self.assertEqual(doc.read_text(), f"head\n{render_table.BEGIN}\nnew\n{render_table.END}\ntail\n")
            doc.write_text("no markers")
            with self.assertRaises(SystemExit):
                render_table.update(doc, "x")


class Runners(unittest.TestCase):
    def test_parse_compute_keeps_colons_in_the_offer(self):
        c = run_folo.parse_compute("cc_1f2e:NVIDIA GeForce RTX 4090@community", 0.5, None)
        self.assertEqual(c, {"connection_id": "cc_1f2e", "offer_id": "NVIDIA GeForce RTX 4090@community", "max_hours": 0.5})
        self.assertEqual(run_folo.parse_compute("cc:a:b", 1, 20)["offer_id"], "a:b")
        self.assertIsNone(run_folo.parse_compute(None, 0.5, None))
        with self.assertRaises(SystemExit):
            run_folo.parse_compute("cc_only", 0.5, None)

    def test_train_config(self):
        class A:
            epochs, batch, imgsz, weights, device = 100, 16, 640, "folo-n-coco-v2", "cpu"

        pre = run_folo.train_config(common.RunKey("folo", "folo-n", "bccd", "pretrained", 1), "ds_1", A, None, "ft-x")
        self.assertEqual(pre["weights"], "folo-n-coco-v2")
        # A weights file on the server is named by its model id, not by the name --weights found it under.
        key = common.RunKey("folo", "folo-n", "bccd", "pretrained", 1)
        self.assertEqual(run_folo.train_config(key, "ds_1", A, None, "ft-x", "m_ef99259f")["weights"], "m_ef99259f")
        self.assertEqual((pre["seed"], pre["epochs"], pre["batch"], pre["imgsz"]), (1, 100, 16, 640))
        self.assertEqual((pre["arch"], pre["device"]), ("folo-n", "cpu"))
        self.assertEqual(pre["name"], "ft-x")
        self.assertNotIn("compute", pre)
        self.assertNotIn("optimizer", pre)  # FOLO's default (auto)
        scr = run_folo.train_config(
            common.RunKey("folo", "folo-s", "bccd", "scratch", 0), "ds_1", A, {"connection_id": "c"}, "ft-y"
        )
        self.assertNotIn("weights", scr)
        self.assertEqual(scr["compute"], {"connection_id": "c"})
        self.assertEqual((scr["arch"], scr["device"]), ("folo-s", "auto"))  # a cloud run trains on its pod's GPU

    def test_reap_touches_only_its_own_runs(self):
        class FakeApi:
            def __init__(self):
                self.deleted = []

            def get(self, path):
                return {"instances": [
                    {"id": "ci_mine", "name": "a", "run_id": "run_mine", "status": "running"},
                    {"id": "ci_other", "name": "b", "run_id": "run_other", "status": "running"},
                    {"id": "ci_old", "name": "c", "run_id": "run_mine", "status": "terminated"},
                ]}

            def call(self, method, path):
                self.deleted.append(path)
                return {"status": "terminated"}

        m = run_folo.Matrix.__new__(run_folo.Matrix)
        m.api = FakeApi()
        m.log = lambda msg: None
        self.assertEqual(m.reap(set()), [])  # nothing of its own: nothing is touched (it once meant "every live one")
        self.assertEqual(m.api.deleted, [])
        self.assertEqual(m.reap({"run_mine"}), [])
        self.assertEqual(m.api.deleted, ["/api/cloud/instances/ci_mine"])

    def test_scorer_imports_are_checked_up_front(self):
        old = scorer.NEEDS
        scorer.NEEDS = ("folo_no_such_module",)
        try:
            self.assertIn("folo_no_such_module", scorer.missing())
            with self.assertRaises(SystemExit) as e:
                scorer.require()
            self.assertIn(sys.executable, str(e.exception))
            self.assertIn("Nothing was started", str(e.exception))
        finally:
            scorer.NEEDS = old

    def test_wall_seconds(self):
        run = {"started": "2026-11-10T10:00:00Z", "finished": "2026-11-10T10:02:30.5Z"}
        self.assertAlmostEqual(run_folo.wall_seconds(run), 150.5)
        self.assertIsNone(run_folo.wall_seconds({"started": None, "finished": None}))

    def test_ultralytics_weights(self):
        self.assertEqual(run_ultralytics.model_name(Path("w/yolo26s.pt")), "yolo26s")
        with tempfile.TemporaryDirectory() as t:
            fake = Path(t) / "yolo26n.pt"
            fake.write_bytes(b"not the published file")
            with self.assertRaises(SystemExit):
                run_ultralytics.ensure_weights(fake, False)  # a known name must have its known sha256
            other = Path(t) / "yolo26s.pt"
            other.write_bytes(b"x")
            self.assertEqual(run_ultralytics.ensure_weights(other, False), datasets.sha256_file(other))
            with self.assertRaises(SystemExit):
                run_ultralytics.ensure_weights(Path(t) / "missing.pt", True)  # only yolo26<size>.pt is downloaded

    def test_run_order(self):
        names = ["brain-tumor", "bccd"]
        self.assertEqual(
            run_ultralytics.run_order([0, 1], names, "seed"),
            [(0, "brain-tumor"), (0, "bccd"), (1, "brain-tumor"), (1, "bccd")],
        )
        self.assertEqual(
            run_ultralytics.run_order([0, 1], names, "dataset"),
            [(0, "brain-tumor"), (1, "brain-tumor"), (0, "bccd"), (1, "bccd")],
        )

    def test_epochs_in_results_csv(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "results.csv"
            self.assertIsNone(run_ultralytics.epochs_in(p))
            p.write_text("epoch,loss\n1,2\n2,1\n\n")
            self.assertEqual(run_ultralytics.epochs_in(p), 2)


if __name__ == "__main__":
    unittest.main()
