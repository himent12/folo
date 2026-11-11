"""Pixel parity between two capture trees (FOLO Desktop baseline vs. a port, or run a vs. run b).

    venv/Scripts/python tools/parity.py --a shots/a --b shots/b --out parity/a-vs-b [--tol 8]

Walks both trees for *.png (matched by relative path) and, for each pair, measures with the per-pixel
difference d = max(|A-B|) over the R, G, B channels (alpha ignored):

  * same_size, size_a, size_b (images of different sizes are compared on their common top-left area);
  * pct_any / pct_tol / pct_32: percent of pixels with d > 0, d > tol and d > 32;
  * mad: mean absolute difference over all pixels and channels (0-255 scale), and max_diff;
  * regions: bounding boxes of connected areas of d > tol, areas closer than --merge px joined, the
    largest --max-regions kept, each with its pixel count and max difference (region_count is the total);
  * a triptych PNG (A | B | difference amplified x--amp over a dimmed A, regions boxed) at full
    resolution for every pair with pct_tol >= --triptych-pct or a size mismatch.

Writes <out>/summary.json (every number) and <out>/summary.md (a table, worst first, and the missing
files). --fail-pct P exits 1 when some pair has pct_tol > P or a file is missing (for scripted gates).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def regions(mask: np.ndarray, d: np.ndarray, merge: int, cap: int) -> tuple[list[dict], int]:
    """Connected areas of `mask`, joined when closer than about `merge` px.

    The mask is reduced to a grid of merge x merge cells; 8-connected marked cells form one region, so
    areas within a cell of each other merge. Exact boxes, counts and maxima come from the pixels.
    """
    if not mask.any():
        return [], 0
    h, w = mask.shape
    c = max(1, merge)
    gh, gw = -(-h // c), -(-w // c)
    pad = np.zeros((gh * c, gw * c), bool)
    pad[:h, :w] = mask
    cells = pad.reshape(gh, c, gw, c).any(axis=(1, 3))
    # Label the (small) cell grid with a flood fill over marked cells only.
    labels = np.zeros(cells.shape, np.int32)
    marked = np.argwhere(cells)
    n = 0
    for y0, x0 in marked:
        if labels[y0, x0]:
            continue
        n += 1
        labels[y0, x0] = n
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for yy in (y - 1, y, y + 1):
                if yy < 0 or yy >= gh:
                    continue
                for xx in (x - 1, x, x + 1):
                    if 0 <= xx < gw and cells[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = n
                        stack.append((yy, xx))
    # Per-pixel labels of the differing pixels, then per-label stats, vectorised.
    ys, xs = np.nonzero(mask)
    lab = labels[ys // c, xs // c]
    order = np.argsort(lab, kind="stable")
    lab, ys, xs = lab[order], ys[order], xs[order]
    vals = d[ys, xs]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    counts = np.diff(np.r_[starts, lab.size])
    out = []
    for s, k in zip(starts, counts):
        e = s + k
        out.append({
            "x": int(xs[s:e].min()), "y": int(ys[s:e].min()),
            "w": int(xs[s:e].max() - xs[s:e].min() + 1), "h": int(ys[s:e].max() - ys[s:e].min() + 1),
            "pixels": int(k), "max_diff": int(vals[s:e].max()),
        })
    out.sort(key=lambda r: -r["pixels"])
    return out[:cap], len(out)


def triptych(a: np.ndarray, b: np.ndarray, d: np.ndarray, boxes: list[dict], amp: int, path: Path) -> None:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    canvas = np.zeros((h, 3 * w, 3), np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], w : w + b.shape[1]] = b
    ch, cw = d.shape
    base = (a[:ch, :cw].astype(np.uint16) // 4).astype(np.uint8)
    heat = np.clip(d.astype(np.uint32) * amp, 0, 255).astype(np.uint8)
    panel = base.copy()
    panel[..., 0] = np.maximum(base[..., 0], heat)
    panel[..., 1] = np.where(heat > 0, base[..., 1] // 2, base[..., 1])
    panel[..., 2] = np.where(heat > 0, base[..., 2] // 2, base[..., 2])
    canvas[:ch, 2 * w : 2 * w + cw] = panel
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    for r in boxes:
        x0, y0 = 2 * w + r["x"] - 2, r["y"] - 2
        draw.rectangle([x0, y0, x0 + r["w"] + 3, y0 + r["h"] + 3], outline=(255, 220, 0), width=2)
    draw.line([(w, 0), (w, h)], fill=(255, 0, 255), width=1)
    draw.line([(2 * w, 0), (2 * w, h)], fill=(255, 0, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, compress_level=1)


def compare(rel: str, pa: Path, pb: Path, args, out: Path) -> dict:
    a, b = load_rgb(pa), load_rgb(pb)
    same = a.shape == b.shape
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    ca, cb = a[:h, :w], b[:h, :w]
    diff = np.abs(ca.astype(np.int16) - cb.astype(np.int16)).astype(np.uint8)
    d = diff.max(axis=2)
    total = d.size
    over_tol = d > args.tol
    boxes, nreg = regions(over_tol, d, args.merge, args.max_regions)
    res = {
        "path": rel,
        "same_size": same,
        "size_a": [int(a.shape[1]), int(a.shape[0])],
        "size_b": [int(b.shape[1]), int(b.shape[0])],
        "pct_any": 100.0 * np.count_nonzero(d) / total,
        "pct_tol": 100.0 * np.count_nonzero(over_tol) / total,
        "pct_32": 100.0 * np.count_nonzero(d > 32) / total,
        "mad": float(diff.mean()),
        "max_diff": int(d.max()),
        "region_count": nreg,
        "regions": boxes,
        "triptych": None,
    }
    if not same or res["pct_tol"] >= args.triptych_pct and res["pct_tol"] > 0:
        tp = out / "triptych" / (rel[:-4] + ".png")
        triptych(a, b, d, boxes, args.amp, tp)
        res["triptych"] = tp.relative_to(out).as_posix()
    return res


def pngs(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*.png")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tol", type=int, default=8, help="per-pixel difference that counts (0-255) [8]")
    ap.add_argument("--merge", type=int, default=24, help="join regions closer than about this many px [24]")
    ap.add_argument("--max-regions", type=int, default=20)
    ap.add_argument("--triptych-pct", type=float, default=0.001,
                    help="write a triptych when pct_tol is at least this (percent) [0.001]")
    ap.add_argument("--amp", type=int, default=8, help="difference amplification in the triptych [8]")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--fail-pct", type=float, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    sa, sb = pngs(args.a), pngs(args.b)
    both = sorted(sa & sb)
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda r: compare(r, args.a / r, args.b / r, args, args.out), both))
    results.sort(key=lambda r: (r["same_size"], -r["pct_tol"], -r["mad"]))
    summary = {
        "a": str(args.a), "b": str(args.b), "tol": args.tol, "merge": args.merge,
        "compared": len(results),
        "identical": sum(1 for r in results if r["same_size"] and r["pct_any"] == 0),
        "differ_above_tol": sum(1 for r in results if not r["same_size"] or r["pct_tol"] > 0),
        "only_in_a": sorted(sa - sb), "only_in_b": sorted(sb - sa),
        "seconds": round(time.perf_counter() - t0, 2),
        "results": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    md = [f"# Parity: `{args.a}` vs `{args.b}`", "",
          f"{len(results)} compared, {summary['identical']} identical, {summary['differ_above_tol']} differ above "
          f"tol {args.tol}; {len(summary['only_in_a'])} only in A, {len(summary['only_in_b'])} only in B; "
          f"{summary['seconds']} s.", "",
          "| image | size | % >0 | % >tol | % >32 | MAD | max | regions (largest: x,y wxh px) | triptych |",
          "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        size = f"{r['size_a'][0]}x{r['size_a'][1]}" + ("" if r["same_size"] else f" vs {r['size_b'][0]}x{r['size_b'][1]}")
        reg = ""
        if r["region_count"]:
            g = r["regions"][0]
            reg = f"{r['region_count']} ({g['x']},{g['y']} {g['w']}x{g['h']} {g['pixels']})"
        tp = f"[png]({r['triptych']})" if r["triptych"] else ""
        md.append(f"| {r['path']} | {size} | {r['pct_any']:.4f} | {r['pct_tol']:.4f} | {r['pct_32']:.4f} | "
                  f"{r['mad']:.4f} | {r['max_diff']} | {reg} | {tp} |")
    for side in ("a", "b"):
        miss = summary[f"only_in_{side}"]
        if miss:
            md += ["", f"## Only in {side.upper()} ({len(miss)})", ""] + [f"- {m}" for m in miss]
    (args.out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(results)} compared, {summary['identical']} identical, {summary['differ_above_tol']} above tol, "
          f"{len(summary['only_in_a'])}+{len(summary['only_in_b'])} missing, {summary['seconds']} s -> {args.out}")
    if args.fail_pct is not None:
        bad = [r for r in results if not r["same_size"] or r["pct_tol"] > args.fail_pct]
        if bad or summary["only_in_a"] or summary["only_in_b"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
