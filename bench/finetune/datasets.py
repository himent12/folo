"""Fetch, verify and lay out the fine-tuning datasets in one canonical YOLO layout (plus the COCO ground truth).

    python bench/finetune/datasets.py fetch --root D:/folo-data/finetune            # all datasets
    python bench/finetune/datasets.py fetch --root D:/folo-data/finetune --only bccd,signature
    python bench/finetune/datasets.py meta --root D:/folo-data/finetune --out bench/finetune/datasets.json

Each dataset ends up in ``<root>/<name>/``:

    data.yaml                         train / val / test / names (no `path:` key: both tools then read the folder
                                      the YAML is in, wherever it lies)
    images/<split>/<file>             the source images, byte for byte
    labels/<split>/<stem>.txt         YOLO labels rewritten from the parsed boxes (6 decimals; boxes clipped to the
                                      image, empty and exactly repeated boxes dropped, counted in meta.json)
    annotations/instances_<split>.json  COCO ground truth of the evaluation split, built from those labels: the one
                                      ground truth both tools are scored against (scorer.py). FOLO's `val
                                      --save-json` reads it too, so its image and category ids are these.
    meta.json                         source, licence, counts per split, the fixes, and the layout's digest

Needs Python 3.10+ and Pillow (image sizes). Downloads are cached in ``<root>/_downloads`` and checked against the
sha256 below; the finished layout is checked against its content digest, so a re-packed archive with the same files
still passes and a changed file does not. Nothing here is committed but this code and datasets.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
USER_AGENT = "folo-bench-finetune/1"


@dataclass(frozen=True)
class Spec:
    name: str
    title: str
    domain: str
    url: str
    sha256: str
    licence: str
    licence_url: str
    source: str
    kind: str  # "yolo_zip" (Ultralytics-hosted, already YOLO) or "bccd_voc" (PASCAL VOC XML)
    classes: tuple[str, ...]
    # Split the scores are reported on: the dataset's test split when it has one, else val.
    eval_split: str
    # sha256 of the finished layout (layout_digest); empty until measured once.
    layout_sha256: str = ""
    notes: str = ""
    archive_root: str = ""  # folder inside the archive that holds the dataset


ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
BCCD_COMMIT = "d272fb14cdff6e473fafeeeba32aba5f560e9e43"

SPECS: dict[str, Spec] = {
    s.name: s
    for s in [
        Spec(
            name="bccd",
            title="BCCD blood cells",
            domain="microscopy",
            url=f"https://github.com/Shenggan/BCCD_Dataset/archive/{BCCD_COMMIT}.zip",
            sha256="e2ed144a0a2c6dbe351f3ddb09fb8ff2c9ad4f947bdf42b6a52dd4f38a996ba9",
            licence="MIT",
            licence_url=f"https://github.com/Shenggan/BCCD_Dataset/blob/{BCCD_COMMIT}/LICENSE",
            source="https://github.com/Shenggan/BCCD_Dataset",
            kind="bccd_voc",
            classes=("RBC", "WBC", "Platelets"),
            eval_split="test",
            layout_sha256="67ab1b141b038099bc75408b737638e8e1cda0812611208e76acd90e364e0d1a",
            notes="PASCAL VOC XML converted to YOLO; the repository's own ImageSets/Main train / val / test split",
            archive_root=f"BCCD_Dataset-{BCCD_COMMIT}/BCCD",
        ),
        Spec(
            name="brain-tumor",
            title="Brain tumor (MRI)",
            domain="medical imaging",
            url=f"{ULTRALYTICS_ASSETS}/brain-tumor.zip",
            sha256="0710c520add47c5de7d0da78bc206662a0a0b4b26777b582edcb4e6f7660cb0d",
            licence="AGPL-3.0",
            licence_url="https://docs.ultralytics.com/datasets/detect/brain-tumor/",
            source="https://docs.ultralytics.com/datasets/detect/brain-tumor/",
            kind="yolo_zip",
            classes=("negative", "positive"),
            eval_split="val",
            layout_sha256="908edcb35303362c5b77b267b7d43171b9c9c989fd5741242d1ca038f183bf26",
            notes="hosted by Ultralytics; train / val only, so scores are on val",
        ),
        Spec(
            name="signature",
            title="Signatures on documents",
            domain="documents",
            url=f"{ULTRALYTICS_ASSETS}/signature.zip",
            sha256="6d7bfafd89bcfffb2fdf21f79df16e1988e30f5aa006b0a27d39b51f3bcae171",
            licence="AGPL-3.0",
            licence_url="https://docs.ultralytics.com/datasets/detect/signature/",
            source="https://docs.ultralytics.com/datasets/detect/signature/",
            kind="yolo_zip",
            classes=("signature",),
            eval_split="val",
            layout_sha256="0754c64b4ae38a405dbcdbf5c8e7ce337c9fc65407183b6768a9d107d91e7bde",
            notes="hosted by Ultralytics; train / val only, so scores are on val",
        ),
        Spec(
            name="medical-pills",
            title="Medical pills",
            domain="industrial / product photos",
            url=f"{ULTRALYTICS_ASSETS}/medical-pills.zip",
            sha256="3a15abc491af0e91a835a01bca1e437e3818bf957aab26aba19176db6495ceb5",
            licence="AGPL-3.0",
            licence_url="https://docs.ultralytics.com/datasets/detect/medical-pills/",
            source="https://docs.ultralytics.com/datasets/detect/medical-pills/",
            kind="yolo_zip",
            classes=("pill",),
            eval_split="val",
            layout_sha256="696f575fc1d34c95f77ae19447594af14a756ae09e97296e1b370005a4b0c1b3",
            notes="hosted by Ultralytics; train / val only, so scores are on val",
        ),
    ]
}


# --------------------------------------------------------------------------------------------------------------------
# boxes and labels (pure functions: test_finetune.py covers them)


def parse_yolo_label(text: str, nc: int) -> tuple[list[tuple[int, float, float, float, float]], dict]:
    """Rows of a YOLO label file as (class, cx, cy, w, h), normalised, with the problems counted:
    `bad_rows` (not 5 numbers, or a class outside 0..nc-1) are dropped, boxes are clipped to the image
    (`clipped`), boxes with no area left are dropped (`empty`), and exact repeats of an earlier box are dropped
    (`duplicates`), as Ultralytics' label cache does."""
    fixes = {"bad_rows": 0, "clipped": 0, "empty": 0, "duplicates": 0}
    out: list[tuple[int, float, float, float, float]] = []
    seen: set[tuple] = set()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError
            c = int(float(parts[0]))
            if c != float(parts[0]) or not 0 <= c < nc:
                raise ValueError
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            fixes["bad_rows"] += 1
            continue
        box = clip_box(cx, cy, w, h)
        if box is None:
            fixes["empty"] += 1
            continue
        if box != (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)):
            fixes["clipped"] += 1
        row = (c, *box)
        if row in seen:
            fixes["duplicates"] += 1
            continue
        seen.add(row)
        out.append(row)
    return out, fixes


def clip_box(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    """A normalised box clipped to [0, 1] and rounded to the 6 decimals the label files keep; None if empty."""
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    box = (round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), round(x2 - x1, 6), round(y2 - y1, 6))
    if box[2] <= 0 or box[3] <= 0:
        return None
    # Keep the source numbers when clipping moved nothing by more than the 6-decimal rounding (an edge box written
    # as 0.060937 / 0.121875 starts 5e-7 left of the image), so labels read back identically.
    orig = (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
    return orig if all(abs(a - b) <= 2e-6 for a, b in zip(box, orig)) else box


def voc_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int):
    """A PASCAL VOC box (1-based pixel corners) as a normalised YOLO box, with the conversion Ultralytics' VOC.yaml
    uses (centre minus one pixel, size xmax - xmin). None for an inverted or empty box."""
    if xmax <= xmin or ymax <= ymin:
        return None
    cx, cy = (xmin + xmax) / 2 - 1, (ymin + ymax) / 2 - 1
    return cx / width, cy / height, (xmax - xmin) / width, (ymax - ymin) / height


def format_label(rows) -> str:
    return "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows)


def coco_ground_truth(images: list[tuple[str, int, int, list]], classes: list[str]) -> dict:
    """COCO instances JSON of one split from YOLO rows: `images` is [(file_name, width, height, rows)] in dataset
    order, ids from 1; category ids are class index + 1; boxes are in original-image pixels, area = w * h."""
    out = {
        "info": {"description": "FOLO fine-tuning benchmark ground truth, built from the YOLO labels"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)],
    }
    ann_id = 1
    for img_id, (file_name, width, height, rows) in enumerate(images, start=1):
        out["images"].append({"id": img_id, "file_name": file_name, "width": width, "height": height})
        for c, cx, cy, w, h in rows:
            bw, bh = w * width, h * height
            out["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c + 1,
                    "bbox": [(cx - w / 2) * width, (cy - h / 2) * height, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return out


def data_yaml(classes: list[str], splits: list[str]) -> str:
    """data.yaml without `path:` (both tools then use the YAML's folder), names as a quoted list."""
    lines = [f"{s}: images/{s}" for s in splits]
    lines.append(f"nc: {len(classes)}")
    lines.append("names: [" + ", ".join(json.dumps(c) for c in classes) + "]")
    return "\n".join(lines) + "\n"


def layout_digest(root: Path) -> str:
    """sha256 over the sorted relative paths and sha256 of every file the tools read (data.yaml, images, labels,
    annotations), so the same dataset laid out on two machines has the same digest. The label caches Ultralytics
    writes next to the labels (`labels/<split>.cache`, derived from them) are not part of it."""
    h = hashlib.sha256()
    files = [root / "data.yaml"] + sorted(
        p
        for d in ("images", "labels", "annotations")
        for p in (root / d).rglob("*")
        if p.is_file() and p.suffix != ".cache"
    )
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# fetching and laying out


def download(url: str, dest: Path, sha256: str) -> None:
    if dest.is_file() and sha256_file(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256_file(tmp)
    if got != sha256:
        # Kept for inspection; the layout digest decides whether the files themselves changed.
        print(f"warning: {dest.name}: sha256 {got} differs from the recorded {sha256}", file=sys.stderr)
    tmp.replace(dest)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) not in (1, None):
            # Both tools would rotate it, but by different code: refuse rather than guess.
            raise SystemExit(f"{path}: has an EXIF orientation; re-encode it upright first")
        return im.size


def _sources_yolo(src: Path, spec: Spec):
    """(split, image path, label text) of an Ultralytics-hosted YOLO zip."""
    for split in SPLITS:
        d = src / "images" / split
        if not d.is_dir():
            continue
        for img in sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            lab = src / "labels" / split / (img.stem + ".txt")
            yield split, img, lab.read_text() if lab.is_file() else ""


def _sources_bccd(src: Path, spec: Spec, fixes: dict):
    """(split, image path, label text) of BCCD's VOC layout, labels converted to YOLO here."""
    names = {n: i for i, n in enumerate(spec.classes)}
    for split in SPLITS:
        ids = (src / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
        for stem in sorted(ids):
            img = src / "JPEGImages" / f"{stem}.jpg"
            width, height = image_size(img)
            root = ET.parse(src / "Annotations" / f"{stem}.xml").getroot()
            rows = []
            for obj in root.iter("object"):
                name = obj.findtext("name", "").strip()
                if name not in names:
                    fixes["unknown_class"] += 1
                    continue
                b = obj.find("bndbox")
                box = voc_to_yolo(*(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")), width, height)
                if box is None:
                    fixes["inverted"] += 1
                    continue
                rows.append(f"{names[name]} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            yield split, img, "\n".join(rows)


def lay_out(spec: Spec, src: Path, dest: Path) -> dict:
    """Write the canonical layout of `spec` from the extracted archive at `src` into `dest` (replaced). Returns meta."""
    if dest.exists():
        shutil.rmtree(dest)
    classes = list(spec.classes)
    fixes = {"bad_rows": 0, "clipped": 0, "empty": 0, "duplicates": 0, "unknown_class": 0, "inverted": 0}
    per_split: dict[str, list] = {s: [] for s in SPLITS}
    sources = _sources_bccd(src, spec, fixes) if spec.kind == "bccd_voc" else _sources_yolo(src, spec)
    for split, img, text in sources:
        rows, f = parse_yolo_label(text, len(classes))
        for k, v in f.items():
            fixes[k] += v
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img, dest / "images" / split / img.name)
        (dest / "labels" / split / (img.stem + ".txt")).write_text(format_label(rows), newline="\n")
        width, height = image_size(img)
        # Read the rows back from the file as written, so the ground truth is exactly what the tools train on.
        written, _ = parse_yolo_label((dest / "labels" / split / (img.stem + ".txt")).read_text(), len(classes))
        per_split[split].append((img.name, width, height, written))
    splits = [s for s in SPLITS if per_split[s]]
    if spec.eval_split not in splits:
        raise SystemExit(f"{spec.name}: no {spec.eval_split} split")
    (dest / "data.yaml").write_text(data_yaml(classes, splits), newline="\n")
    (dest / "annotations").mkdir()
    gt = coco_ground_truth(per_split[spec.eval_split], classes)
    (dest / "annotations" / f"instances_{spec.eval_split}.json").write_text(json.dumps(gt), newline="\n")
    counts = {}
    for s in splits:
        items = per_split[s]
        per_class = [0] * len(classes)
        for _, _, _, rows in items:
            for r in rows:
                per_class[r[0]] += 1
        counts[s] = {
            "images": len(items),
            "boxes": sum(per_class),
            "background_images": sum(1 for it in items if not it[3]),
            "boxes_per_class": dict(zip(classes, per_class)),
        }
    meta = {
        "name": spec.name,
        "title": spec.title,
        "domain": spec.domain,
        "classes": classes,
        "source": spec.source,
        "url": spec.url,
        "archive_sha256": spec.sha256,
        "licence": spec.licence,
        "licence_url": spec.licence_url,
        "eval_split": spec.eval_split,
        "splits": counts,
        "image_sizes": sorted({f"{w}x{h}" for s in splits for _, w, h, _ in per_split[s]}),
        "fixes": {k: v for k, v in fixes.items() if v},
        "notes": spec.notes,
        "layout_sha256": layout_digest(dest),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", newline="\n")
    return meta


def verify_layout(spec: Spec, dest: Path) -> str:
    """The layout digest of the dataset in `dest`, recomputed from its files (never taken from meta.json), checked
    against the recorded one: what a runner trains on and scores against, and records as provenance."""
    if not (dest / "data.yaml").is_file():
        raise SystemExit(f"{dest}: no dataset there (python bench/finetune/datasets.py fetch --only {spec.name})")
    got = layout_digest(dest)
    if spec.layout_sha256 and got != spec.layout_sha256:
        raise SystemExit(
            f"{spec.name}: the files in {dest} have layout digest {got}, not the recorded {spec.layout_sha256}: they "
            f"changed since they were laid out; lay them out again (datasets.py fetch --only {spec.name})"
        )
    return got


def fetch(spec: Spec, root: Path, check_digest: bool = True) -> dict:
    """Download (cached), verify and lay out one dataset under `root`; returns its meta. Skips the work when the
    layout is already there and its files (hashed again, not meta.json's word for it) have the recorded digest."""
    dest = root / spec.name
    meta_path = dest / "meta.json"
    if meta_path.is_file() and (dest / "data.yaml").is_file() and spec.layout_sha256:
        meta = json.loads(meta_path.read_text())
        if meta.get("layout_sha256") == spec.layout_sha256:
            if layout_digest(dest) == spec.layout_sha256:
                return meta
            print(f"{spec.name}: files in {dest} changed since they were laid out: laying it out again", file=sys.stderr)
    archive = root / "_downloads" / Path(spec.url).name
    download(spec.url, archive, spec.sha256)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{spec.name}-") as tmp:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (Path(tmp) / info.filename).resolve()
                if not str(target).startswith(str(Path(tmp).resolve())):
                    raise SystemExit(f"{archive}: unsafe path {info.filename}")
            z.extractall(tmp)
        src = Path(tmp) / spec.archive_root if spec.archive_root else Path(tmp)
        meta = lay_out(spec, src, dest)
    if check_digest and spec.layout_sha256 and meta["layout_sha256"] != spec.layout_sha256:
        raise SystemExit(
            f"{spec.name}: layout digest {meta['layout_sha256']} differs from the recorded {spec.layout_sha256}: "
            "the source files changed"
        )
    return meta


def pick(only: str | None) -> list[Spec]:
    if not only:
        return list(SPECS.values())
    names = [n.strip() for n in only.split(",") if n.strip()]
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(SPECS)}")
    return [SPECS[n] for n in names]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download, verify and lay out datasets")
    f.add_argument("--root", required=True, type=Path)
    f.add_argument("--only", help="comma-separated dataset names (default: all)")
    f.add_argument("--no-digest-check", action="store_true", help="lay out even if the files changed")
    m = sub.add_parser("meta", help="collect every dataset's meta.json into one file (bench/finetune/datasets.json)")
    m.add_argument("--root", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        args.root.mkdir(parents=True, exist_ok=True)
        for spec in pick(args.only):
            meta = fetch(spec, args.root, check_digest=not args.no_digest_check)
            s = meta["splits"]
            print(
                f"{spec.name}: " + ", ".join(f"{k} {v['images']} images / {v['boxes']} boxes" for k, v in s.items())
                + f"; digest {meta['layout_sha256']}"
            )
    else:
        metas = [json.loads((args.root / s.name / "meta.json").read_text()) for s in SPECS.values()]
        args.out.write_text(json.dumps(metas, indent=2) + "\n", newline="\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
