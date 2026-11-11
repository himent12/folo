"""Bookkeeping shared by the runners and the renderer: run keys, result files, the summary.

A run is one (tool, model, dataset, variant, seed): tool `folo` or `ultralytics`, model the architecture
(`folo-n`, `folo-s`, ... for FOLO; the checkpoint's stem, `yolo26n`, `yolo26s`, ... for Ultralytics), variant
`pretrained` (fine-tuned from COCO weights) or `scratch` (no weights: the control). Its result is one JSON file,
`<model>-<dataset>-<variant>-s<seed>.json`, written atomically once the run is scored; a runner skips every run whose
file exists, so a matrix can be stopped and started again. Models of the same size letter (folo-n and yolo26n,
folo-s and yolo26s) are compared in one table.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

TOOLS = ("folo", "ultralytics")
VARIANTS = ("pretrained", "scratch")
RESULT_VERSION = 1


@dataclass(frozen=True)
class RunKey:
    tool: str
    model: str
    dataset: str
    variant: str
    seed: int

    def __post_init__(self):
        if self.tool not in TOOLS:
            raise ValueError(f"tool {self.tool!r} is not one of {TOOLS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant {self.variant!r} is not one of {VARIANTS}")
        if (self.tool == "folo") != self.model.startswith("folo-") or not self.model:
            raise ValueError(f"model {self.model!r} does not belong to tool {self.tool!r}")

    @property
    def stem(self) -> str:
        return f"{self.model}-{self.dataset}-{self.variant}-s{self.seed}"

    def result_path(self, results_dir: Path) -> Path:
        return Path(results_dir) / f"{self.stem}.json"


def size_of(model: str) -> str:
    """The size letter that pairs models across tools: `folo-n` and `yolo26n` are `n`; anything else is itself."""
    return model[-1] if model and model[-1] in "nsmlx" else model


def parse_seeds(text: str) -> list[int]:
    """`0,1,2` or `0-2` (or a mix: `0-2,5`) as a sorted list without repeats."""
    seeds: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise ValueError(f"bad seed range {part!r}")
            seeds.update(range(a, b + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        raise ValueError("no seeds given")
    return sorted(seeds)


def write_json_atomic(path: Path, obj) -> None:
    """`obj` as indented JSON at `path`, through a temporary file of its own in the same folder and `os.replace`, so
    a reader sees the old file or the new one, and two writers never share a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, indent=2) + "\n")
        os.chmod(tmp, 0o644)  # mkstemp's 0600 would make result files private on a pod
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def host_info() -> dict:
    return {"os": platform.platform(), "python": platform.python_version(), "machine": platform.node()}


def result_record(key: RunKey, **fields) -> dict:
    """The shape every result file has (fields not known for a tool are None)."""
    rec = {
        "version": RESULT_VERSION,
        "tool": key.tool,
        "model": key.model,
        "dataset": key.dataset,
        "variant": key.variant,
        "seed": key.seed,
        "epochs_planned": None,
        "epochs_run": None,
        "imgsz": None,
        "batch": None,
        "weights": None,  # what the run started from: a name / file and its sha256, or None from scratch
        "eval_split": None,
        "metrics": None,  # scorer.score(): the common pycocotools numbers
        "tool_reported": None,  # the tool's own validation of its best checkpoint, for reference only
        "train_wall_s": None,
        "gpu": None,
        "versions": None,
        "compute": None,  # where it trained: local, or the cloud offer / instance / cost
        "created": now_iso(),
    }
    unknown = set(fields) - set(rec)
    if unknown:
        raise ValueError(f"unknown result fields {sorted(unknown)}")
    rec.update(fields)
    return rec


def load_results(results_dir: Path) -> list[dict]:
    """Every run result in `results_dir` (summary.json and other files without the run shape are skipped)."""
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("version") == RESULT_VERSION and r.get("tool") in TOOLS:
            out.append(r)
    return out


def _stat(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def shared(r: dict) -> bool:
    """The run trained side by side with others on one GPU."""
    return ((r.get("compute") or {}).get("concurrent_jobs") or 1) > 1


def summarize(results: list[dict], expected_seeds: int = 3) -> dict:
    """Group runs by (dataset, tool, model, variant): AP / AP50 / AP75 / training time over seeds, with which seeds
    are present, and `complete` once `expected_seeds` seeds are (1 for the scratch control)."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["dataset"], r["tool"], r["model"], r["variant"]), []).append(r)
    rows = []
    for (dataset, tool, model, variant), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        want = 1 if variant == "scratch" else expected_seeds
        metric = lambda k: [r["metrics"][k] for r in rs if r.get("metrics") and r["metrics"].get(k) is not None]
        rows.append(
            {
                "dataset": dataset,
                "tool": tool,
                "model": model,
                "variant": variant,
                "seeds": [r["seed"] for r in rs],
                "complete": len(rs) >= want,
                "eval_split": rs[0].get("eval_split"),
                "ap": _stat(metric("ap")),
                "ap50": _stat(metric("ap50")),
                "ap75": _stat(metric("ap75")),
                # Training time of runs that had their GPU to themselves; runs trained side by side on one GPU
                # (`compute.concurrent_jobs` > 1) are counted apart, since they share it.
                "train_wall_s": _stat([r["train_wall_s"] for r in rs if r.get("train_wall_s") is not None and not shared(r)]),
                "train_wall_shared_s": _stat([r["train_wall_s"] for r in rs if r.get("train_wall_s") is not None and shared(r)]),
                "epochs_run": sorted({r.get("epochs_run") for r in rs if r.get("epochs_run") is not None}),
                "epochs_planned": sorted({r.get("epochs_planned") for r in rs if r.get("epochs_planned") is not None}),
                "gpus": sorted({r.get("gpu") for r in rs if r.get("gpu")}),
            }
        )
    return {"version": RESULT_VERSION, "created": now_iso(), "runs": len(results), "groups": rows}
