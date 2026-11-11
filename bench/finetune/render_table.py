"""Render the "Fine-tuning on small datasets" tables of docs/BENCHMARKS.md from the run results.

    python bench/finetune/render_table.py bench/finetune/results                     # print
    python bench/finetune/render_table.py bench/finetune/results --update docs/BENCHMARKS.md

Reads every run result in the folder (common.load_results), keeps those trained for --epochs (the protocol's 100;
a smoke run with fewer epochs is counted and left out), writes summary.json next to them, and prints the dataset
table and the results table. --update replaces the text between `<!-- finetune:begin -->` and
`<!-- finetune:end -->`. Nothing is invented: a group without results is shown as "not run", a group with fewer
seeds than the protocol's as partial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

BEGIN, END = "<!-- finetune:begin -->", "<!-- finetune:end -->"


def table(head: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" * len(head)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cell(stat: dict | None, want: int) -> str:
    """`0.412 (0.405–0.420)` over seeds; `0.412` for one; `not run` without; `, 2 of 3 seeds` when partial."""
    if not stat:
        return "not run"
    if stat["n"] == 1:
        s = f"{stat['mean']:.3f}"
    else:
        s = f"{stat['mean']:.3f} ({stat['min']:.3f}–{stat['max']:.3f})"
    if stat["n"] < want:
        s += f", {stat['n']} of {want} seeds"
    return s


def minutes(stat: dict | None) -> str:
    return "" if not stat else f"{stat['mean'] / 60:.1f}"


def render(results: list[dict], metas: list[dict], epochs: int, seeds: int) -> tuple[str, dict]:
    kept = [r for r in results if r.get("epochs_planned") == epochs]
    ignored = len(results) - len(kept)
    summary = common.summarize(kept, expected_seeds=seeds)
    summary["epochs"] = epochs
    summary["ignored_other_epochs"] = ignored
    groups = {(g["dataset"], g["tool"], g["variant"]): g for g in summary["groups"]}
    lines = []

    names = [m["name"] for m in metas] or sorted({g[0] for g in groups})
    by_name = {m["name"]: m for m in metas}
    rows = []
    for n in names:
        m = by_name.get(n)
        if not m:
            continue
        s = m["splits"]
        ev = m["eval_split"]
        rows.append(
            [
                f"[{m['title']}]({m['source']})",
                m["domain"],
                m["licence"],
                len(m["classes"]),
                s["train"]["images"],
                s["train"]["boxes"],
                f"{ev}: {s[ev]['images']} / {s[ev]['boxes']}",
            ]
        )
    if rows:
        lines.append(
            table(
                ["dataset", "domain", "licence", "classes", "train images", "train boxes", "scored on: images / boxes"],
                rows,
                "lllrrrl",
            )
        )
        lines.append("")

    n_runs = len(kept)
    planned = len(names) * (2 * seeds + 1)
    if n_runs == 0:
        lines.append(
            f"**No {epochs}-epoch run has been measured yet**: the table below is empty until the matrix "
            f"({planned} runs) has run."
        )
    elif n_runs < planned:
        lines.append(f"**Partial**: {n_runs} of the {planned} runs of the matrix have been measured.")
    else:
        lines.append(f"All {planned} runs of the matrix measured.")
    if ignored:
        lines.append(f"({ignored} result files of runs with another epoch count, such as smoke runs, are left out.)")
    lines.append("")

    rows = []
    for n in names:
        fp = groups.get((n, "folo", "pretrained"))
        up = groups.get((n, "ultralytics", "pretrained"))
        fs = groups.get((n, "folo", "scratch"))
        g = lambda grp, k: grp.get(k) if grp else None
        delta = ""
        if fp and up and fp["ap"] and up["ap"]:
            delta = f"{fp['ap']['mean'] - up['ap']['mean']:+.3f}"
        rows.append(
            [
                n,
                cell(g(fp, "ap"), seeds),
                cell(g(fp, "ap50"), seeds),
                cell(g(up, "ap"), seeds),
                cell(g(up, "ap50"), seeds),
                delta,
                cell(g(fs, "ap"), 1),
                cell(g(fs, "ap50"), 1),
                minutes(g(fp, "train_wall_s")),
                minutes(g(up, "train_wall_s")),
            ]
        )
    lines.append(
        table(
            [
                "dataset",
                "folo-n fine-tuned AP",
                "AP50",
                "yolo26n fine-tuned AP",
                "AP50",
                "FOLO − Ultralytics (AP)",
                "folo-n from scratch AP",
                "AP50",
                "folo-n min / run",
                "yolo26n min / run",
            ],
            rows,
            "lllllrllrr",
        )
    )
    gpus = sorted({gpu for grp in summary["groups"] for gpu in grp["gpus"]})
    if gpus:
        lines.append("")
        lines.append("Trained on: " + "; ".join(gpus) + ".")
    return "\n".join(lines) + "\n", summary


def update(doc: Path, text: str) -> None:
    s = doc.read_text(encoding="utf-8")
    a, b = s.find(BEGIN), s.find(END)
    if a < 0 or b < a:
        raise SystemExit(f"{doc}: markers {BEGIN} / {END} not found")
    s = s[: a + len(BEGIN)] + "\n" + text + s[b:]
    doc.write_text(s, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the fine-tuning tables")
    ap.add_argument("results", type=Path)
    ap.add_argument("--datasets-meta", type=Path, default=HERE / "datasets.json")
    ap.add_argument("--epochs", type=int, default=100, help="the protocol's epoch count; other runs are left out")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per fine-tuned group in the protocol")
    ap.add_argument("--update", type=Path, default=None, help="docs/BENCHMARKS.md")
    ap.add_argument("--no-summary", action="store_true", help="do not write summary.json")
    args = ap.parse_args(argv)
    metas = json.loads(args.datasets_meta.read_text()) if args.datasets_meta.is_file() else []
    text, summary = render(common.load_results(args.results), metas, args.epochs, args.seeds)
    if not args.no_summary:
        common.write_json_atomic(args.results / "summary.json", summary)
    if args.update:
        update(args.update, text)
    else:
        sys.stdout.reconfigure(encoding="utf-8")  # en dashes and minus signs on a Windows console
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
