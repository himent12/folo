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
    # The holdout variant chooses its best checkpoint by another protocol: summary.json has it, the tables (and what
    # they say about the runs: the omitted count, the GPUs) do not.
    ignored = sum(1 for r in results if r.get("variant") != "holdout" and r.get("epochs_planned") != epochs)
    summary = common.summarize(kept, expected_seeds=seeds)
    summary["epochs"] = epochs
    summary["ignored_other_epochs"] = ignored
    groups = {
        (g["dataset"], g["tool"], g["model"], g["variant"]): g for g in summary["groups"] if g["variant"] != "holdout"
    }
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

    # One table per size letter: folo-n against yolo26n (always shown), folo-s against yolo26s once it has results.
    sizes = sorted({common.size_of(g[2]) for g in groups} | {"n"}, key=lambda z: ("nsmlx".find(z), z))
    for size in sizes:
        models = {t: sorted({g[2] for g in groups if g[1] == t and common.size_of(g[2]) == size}) for t in common.TOOLS}
        fm = (models["folo"] or [f"folo-{size}"])[0]
        um = (models["ultralytics"] or [f"yolo26{size}"])[0]
        lines.extend(results_table(groups, names, fm, um, seeds, epochs, ignored if size == sizes[0] else 0))
    gpus = sorted({gpu for grp in groups.values() for gpu in grp["gpus"]})
    if gpus:
        lines.append("Trained on: " + "; ".join(gpus) + ".")
    return "\n".join(lines).rstrip("\n") + "\n", summary


def results_table(groups: dict, names: list[str], fm: str, um: str, seeds: int, epochs: int, ignored: int) -> list[str]:
    """The status line and the results table of the FOLO model `fm` against the Ultralytics model `um`."""
    lines = []
    n_runs = sum(len(g["seeds"]) for k, g in groups.items() if k[2] in (fm, um))
    planned = len(names) * (2 * seeds + 1)
    title = f"{fm} against {um}"
    if n_runs == 0:
        lines.append(
            f"**{title}: no {epochs}-epoch run has been measured yet**: the table below is empty until the matrix "
            f"({planned} runs) has run."
        )
    elif n_runs < planned:
        lines.append(f"**{title}, partial**: {n_runs} of the {planned} runs of the matrix have been measured.")
    else:
        lines.append(f"**{title}**: all {planned} runs of the matrix measured.")
    if ignored:
        lines.append(f"({ignored} result files of runs with another epoch count, such as smoke runs, are left out.)")
    lines.append("")
    rows = []
    for n in names:
        fp = groups.get((n, "folo", fm, "pretrained"))
        up = groups.get((n, "ultralytics", um, "pretrained"))
        fs = groups.get((n, "folo", fm, "scratch"))
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
    head = [
        "dataset",
        f"{fm} fine-tuned AP",
        "AP50",
        f"{um} fine-tuned AP",
        "AP50",
        "FOLO − Ultralytics (AP)",
        f"{fm} from scratch AP",
        "AP50",
        f"{fm} min / run",
        f"{um} min / run",
    ]
    lines.append(table(head, rows, "lllllrllrr"))
    shared = sorted(
        {k[2] for k, g in groups.items() if k[2] in (fm, um) and g.get("train_wall_shared_s")}
    )
    for model in shared:
        alone = any(k[2] == model and g.get("train_wall_s") for k, g in groups.items())
        lines.append("")
        if alone:
            lines.append(
                f"Minutes per run of {model}: only the runs that had the GPU to themselves; the others trained side "
                "by side with other runs on one GPU (their accuracy counts, their time does not)."
            )
        else:
            lines.append(
                f"Minutes per run of {model}: blank, because every one of its runs trained side by side with other "
                "runs on one GPU (their accuracy counts, their time does not)."
            )
    lines.append("")
    return lines


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
