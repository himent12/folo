"""Time `folo train` / `folo val` runs, alternating binaries back to back (an A/B on a machine that is never quiet).

The trainer protocol of docs/BENCHMARKS.md ("Trainer (`train`)", row 10 of the targets table): the incumbent's
synthetic set (512 train / 64 val images, 640x640 JPEG), batch 16, 8 workers, 2 epochs with mosaic on in both, a
fresh process per run; **sustained img/s = 512 / epoch-2 training wall time** (the last `step` record of epoch 2 in
the run's `metrics.jsonl`: the epoch's images over the time since the epoch started, loader start-up included,
validation excluded), best of the rounds.

    python bench/trainer/ab.py train --bin base=D:/.../base/folo.exe --bin new=D:/.../new/folo.exe --rounds 3 \
        --out D:/folo-data/runs-hill/trainer-data/row10 -- \
        --data D:/folo-bench/data --model folo-n --epochs 2 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 \
        --device gpu:0 [--amp]

    python bench/trainer/ab.py val --bin base=... --bin new=... --rounds 3 -- \
        --model D:/folo-weights/folo-n-coco.folo --data D:/folo-data/coco --device gpu:0

Everything after `--` goes to `folo train` / `folo val` unchanged. Per run it prints the per-epoch training img/s,
each epoch's whole duration (training + validation + checkpoints), the process wall time, and how many CPU cores
the run itself and every other process used (system busy time minus the run's own, over the wall time); then the
median, best and spread per binary. Windows only for the CPU accounting (elsewhere those columns are empty).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _filetime(ft) -> float:
    return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7


def system_busy_s() -> float | None:
    """CPU-seconds the whole system has spent busy (all cores)."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    # Kernel time includes idle time.
    return _filetime(kernel) + _filetime(user) - _filetime(idle)


def process_cpu_s(proc: subprocess.Popen) -> float | None:
    """CPU-seconds `proc` has used (it may have exited; the handle is still open)."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    c, e, k, u = (wintypes.FILETIME() for _ in range(4))
    ok = ctypes.windll.kernel32.GetProcessTimes(
        wintypes.HANDLE(int(proc._handle)), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)
    )
    return _filetime(k) + _filetime(u) if ok else None


def run(cmd: list[str], log: Path) -> dict:
    busy0, t0 = system_busy_s(), time.perf_counter()
    with log.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        code = proc.wait()
        ours = process_cpu_s(proc)
    wall = time.perf_counter() - t0
    busy1 = system_busy_s()
    row = {"exit": code, "wall_s": round(wall, 2)}
    if ours is not None and busy0 is not None and busy1 is not None:
        row["cores_ours"] = round(ours / wall, 1)
        row["cores_others"] = round(max(0.0, busy1 - busy0 - ours) / wall, 1)
    return row


def train_row(run_dir: Path) -> dict:
    """Per-epoch training img/s (the epoch's last step record) and whole-epoch durations from metrics.jsonl."""
    last_step: dict[int, float] = {}
    durations: dict[int, float] = {}
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "step":
            last_step[r["epoch"]] = r["imgs_per_s"]
        elif r.get("event") == "epoch":
            durations[r["epoch"]] = r["duration_s"]
    epochs = sorted(last_step)
    return {
        "train_img_s": [round(last_step[e], 1) for e in epochs],
        "epoch_s": [round(durations.get(e, float("nan")), 2) for e in epochs],
    }


def val_row(log: Path) -> dict:
    """The last JSON record `folo val --json` printed (metrics and its own timing, whatever it reports)."""
    out = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in ("seconds", "images", "imgs_per_s", "map50", "map50_95"):
                if k in r:
                    out[k] = r[k]
            m = r.get("metrics")
            if isinstance(m, dict):
                for k in ("map50", "map50_95"):
                    if k in m:
                        out[k] = m[k]
    return out


def summarize(name: str, values: list[float], unit: str) -> str:
    if not values:
        return f"{name}: no data"
    return (
        f"{name}: median {statistics.median(values):.1f} {unit}, best {max(values):.1f}, "
        f"range {min(values):.1f}-{max(values):.1f} (n={len(values)})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["train", "val"])
    ap.add_argument("--bin", action="append", required=True, metavar="NAME=PATH", help="a folo binary (repeat)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs-bench"), help="scratch folder for runs and logs")
    ap.add_argument("--keep", action="store_true", help="keep the run folders (default: only logs and metrics)")
    ap.add_argument("--json", type=Path, help="also write every row to this JSON file")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    args, rest = ap.parse_args(argv[:cut]), argv[cut + 1 :]
    bins = [b.split("=", 1) for b in args.bin]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(args.rounds):
        for name, path in bins:
            tag = f"{name}-r{r}"
            log = args.out / f"{tag}.log"
            if args.kind == "train":
                run_dir = args.out / tag
                shutil.rmtree(run_dir, ignore_errors=True)
                cmd = [path, "train", *rest, "--project", str(args.out), "--name", tag, "--json", "--json-events", "epoch"]
                row = {"bin": name, "round": r, **run(cmd, log)}
                if row["exit"] == 0:
                    row.update(train_row(run_dir))
                    shutil.copy(run_dir / "metrics.jsonl", args.out / f"{tag}.metrics.jsonl")
                if not args.keep:
                    shutil.rmtree(run_dir, ignore_errors=True)
            else:
                cmd = [path, "val", *rest, "--json"]
                row = {"bin": name, "round": r, **run(cmd, log)}
                row.update(val_row(log))
            rows.append(row)
            print(json.dumps(row), flush=True)
    print()
    for name, _ in bins:
        mine = [r for r in rows if r["bin"] == name and r["exit"] == 0]
        if args.kind == "train":
            n_epochs = max((len(r["train_img_s"]) for r in mine), default=0)
            for e in range(n_epochs):
                print(summarize(f"{name} epoch {e + 1} train", [r["train_img_s"][e] for r in mine], "img/s"))
                print(summarize(f"{name} epoch {e + 1} whole", [r["epoch_s"][e] for r in mine], "s"))
        print(summarize(f"{name} process wall", [r["wall_s"] for r in mine], "s"))
        others = [r["cores_others"] for r in mine if "cores_others" in r]
        if others:
            print(summarize(f"{name} cores used by other processes", others, "cores"))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0 if all(r["exit"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
