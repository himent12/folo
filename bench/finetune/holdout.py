"""A selection split carved from a dataset's train split, so a run picks its best checkpoint on images that are
neither the ones it trains on nor the ones it is scored on.

    python bench/finetune/holdout.py --root D:/folo-data/finetune --dest D:/folo-data/finetune-ho --seed 0

For each dataset, a seeded `fraction` (default 0.2) of its train images becomes `val` and the rest `train`, in
``<dest>/<name>-ho<seed>/``: images and labels hard-linked from the source (copied where a link is refused),
`data.yaml` with train and val, and `annotations/instances_val.json` built exactly as datasets.py builds the
evaluation split's. The evaluation split (BCCD test; the others' val) is not copied, so nothing that selects on
this folder can see it. `meta.json` records the source's layout digest, the seed, the fraction and the hold-out's
file names; the folder's own digest (datasets.layout_digest) is the same on every machine for the same source.
The files are links to the source's: editing one in place edits the source too (remove the folder and carve again).

The order is `random.Random(f"{name}-{seed}")` shuffling the sorted train file names, and the hold-out is the first
`max(1, round(n * fraction))` of them: the splits of docs/notes/hill/r6-finetune.md (wave 6, lane finetune).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets  # noqa: E402

DEFAULT_FRACTION = 0.2


def holdout_names(names: list[str], dataset: str, seed: int, fraction: float) -> set[str]:
    """The train file names that become the hold-out: a seeded shuffle of the sorted names, its first
    `max(1, round(len * fraction))`."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    order = sorted(names)
    random.Random(f"{dataset}-{seed}").shuffle(order)
    return set(order[: max(1, round(len(order) * fraction))])


def folder_name(dataset: str, seed: int) -> str:
    return f"{dataset}-ho{seed}"


def split_label(seed: int, fraction: float = DEFAULT_FRACTION) -> str:
    """The name of a hold-out split in result file names: `ho<seed>`, with `-f<fraction>` when not the default."""
    return f"ho{seed}" if fraction == DEFAULT_FRACTION else f"ho{seed}-f{fraction:g}"


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def carve(spec: datasets.Spec, src: Path, dest_root: Path, seed: int, fraction: float = DEFAULT_FRACTION) -> dict:
    """Lay out `spec`'s hold-out split from the dataset in `src` (checked against its recorded layout digest first)
    under `dest_root`; returns its meta. A folder already there with the same source digest, seed, fraction and
    files is kept as it is; any other folder there (another carve, changed files, or no `meta.json`) is refused.
    The folder is built in a staging folder of this call's own next to its place (`.<name>.partial-<pid>-<random>`)
    and renamed into it once complete, so a carve cut short leaves nothing that looks like one (only its staging
    folder, which nothing reads), and of two carves at once the second to finish finds the first's folder and checks
    it as above instead of replacing it."""
    source_digest = datasets.verify_layout(spec, src)
    dest = dest_root / folder_name(spec.name, seed)
    meta_path = dest / "meta.json"
    if dest.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            raise SystemExit(f"{dest} is not a complete hold-out carve (no readable meta.json): remove it") from None
        same = (meta.get("source_layout_sha256"), meta.get("seed"), meta.get("fraction")) == (
            source_digest, seed, fraction
        )
        if same and datasets.layout_digest(dest) == meta.get("layout_sha256"):
            return meta
        raise SystemExit(f"{dest} holds another hold-out (or its files changed): remove it and carve it again")
    imgs = sorted(p for p in (src / "images" / "train").iterdir() if p.suffix.lower() in datasets.IMAGE_EXTS)
    ho = holdout_names([p.name for p in imgs], spec.name, seed, fraction)
    if len(ho) >= len(imgs):
        raise SystemExit(
            f"{spec.name}: a hold-out of {fraction} takes all {len(imgs)} train images, leaving none to train on"
        )
    final, dest = dest, dest_root / f".{folder_name(spec.name, seed)}.partial-{os.getpid()}-{secrets.token_hex(4)}"
    classes = list(spec.classes)
    val, n_train = [], 0
    for img in imgs:
        split = "val" if img.name in ho else "train"
        label = src / "labels" / "train" / (img.stem + ".txt")
        _link(img, dest / "images" / split / img.name)
        if label.is_file():
            _link(label, dest / "labels" / split / label.name)
        if split == "val":
            text = label.read_text() if label.is_file() else ""
            rows, _ = datasets.parse_yolo_label(text, len(classes))
            w, h = datasets.image_size(img)
            val.append((img.name, w, h, rows))
        else:
            n_train += 1
    (dest / "data.yaml").write_text(datasets.data_yaml(classes, ["train", "val"]), newline="\n")
    gt = datasets.coco_ground_truth(val, classes)
    (dest / "annotations").mkdir(parents=True, exist_ok=True)
    (dest / "annotations" / "instances_val.json").write_text(json.dumps(gt), newline="\n")
    meta = {
        "name": folder_name(spec.name, seed),
        "source": spec.name,
        "source_layout_sha256": source_digest,
        "seed": seed,
        "fraction": fraction,
        "splits": {"train": {"images": n_train}, "val": {"images": len(val), "boxes": len(gt["annotations"])}},
        "holdout": sorted(ho),
        "layout_sha256": datasets.layout_digest(dest),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", newline="\n")
    try:
        os.rename(dest, final)  # refused when `final` exists: another carve finished first
    except OSError:
        if not final.exists():
            raise
        shutil.rmtree(dest)
        return carve(spec, src, dest_root, seed, fraction)  # checks the other carve's folder
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="folder of datasets.py fetch")
    ap.add_argument("--dest", required=True, type=Path, help="where the <name>-ho<seed> folders go")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--only", help="comma-separated dataset names (default: all)")
    args = ap.parse_args(argv)
    for spec in datasets.pick(args.only):
        m = carve(spec, args.root / spec.name, args.dest, args.seed, args.fraction)
        s = m["splits"]
        print(f"{m['name']}: train {s['train']['images']}, hold-out {s['val']['images']} images / "
              f"{s['val']['boxes']} boxes; digest {m['layout_sha256']}")


if __name__ == "__main__":
    main()
