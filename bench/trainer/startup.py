"""Time `folo train` start-up: process start to a given training step, alternating binaries back to back.

A fresh process per run reads `folo train --json` from a pipe and stops the run (by PID) at the first `step`
record with `step >= --until-step` (default 1). `step` counts batches (1-based); the trainer emits a record after
every batch (crates/folo-vision/src/train/epoch.rs), and the optimizer updates the weights once per
`accumulate = max(round(nbs / batch), 1)` batches (nbs 64 by default), so at `--batch 16` the first optimizer step is batch 4: `--until-step 4`
times process start to the first optimizer step, the default `1` to the end of the first batch. metrics.jsonl has
every step but stdout is throttled to about ten a second, hence `>=` (the stop may come up to ~0.1 s after step N).
Every record before it is printed with its arrival time, so the phases (dataset open, model build, first batch)
can be read off the log.

    python bench/trainer/startup.py --bin base=D:/.../base/folo.exe --bin new=D:/.../new/folo.exe --rounds 3 \
        --until-step 4 --out D:/folo-data/runs-hill2/trainer/startup -- \
        --data D:/folo-data/coco --model folo-n --epochs 1 --batch 16 --imgsz 640 --device gpu:0

Everything after `--` goes to `folo train` unchanged. `--warm 1` runs every binary once before the timed rounds
(a second start, with whatever caches the first one wrote); with `--warm`, every run keeps the process alive 3 s
after the target step before stopping it (its time is taken at the step), so the GPU pipeline cache (written after
a quiet second) is on disk for the next run. `--clear PATH` deletes PATH before every run (a cold start for caches kept there) and fails if it cannot.
`--timeout` stops a run (killing the process) that has not reached the step by then.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

# How long a `--warm` run stays alive after the target step: the pipeline-cache saver writes after a quiet second.
SETTLE_S = 3.0


def until_step(cmd: list[str], log: Path, step: int, timeout_s: float, settle_s: float) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    # A blocked read cannot see a deadline: a timer kills the process, which ends the read.
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, expire)
    timer.daemon = True
    timer.start()
    row: dict = {"until_step": step, "step_s": None}
    try:
        with log.open("w", encoding="utf-8") as f:
            assert proc.stdout is not None
            for line in proc.stdout:
                t = time.perf_counter() - t0
                f.write(f"{t:8.3f} {line}")
                s = line.strip()
                if not s.startswith("{"):
                    continue
                try:
                    r = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "step" and int(r.get("step", 0)) >= step:
                    row["step_s"] = round(t, 3)
                    row["step"] = r.get("step")
                    break
        if row["step_s"] is not None and settle_s > 0 and proc.poll() is None:
            time.sleep(settle_s)
    finally:
        timer.cancel()
        proc.kill()
        proc.wait()
    if timed_out.is_set() and row["step_s"] is None:
        row["timed_out"] = True
    return row


def clear(path: Path) -> None:
    """Deletes `path` (a file or a folder); raises when something is left."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()
    if path.exists():
        raise RuntimeError(f"--clear: {path} could not be deleted")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", action="append", required=True, metavar="NAME=PATH", help="a folo binary (repeat)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--until-step", type=int, default=1, help="stop at the first step record >= N (4: first optimizer step at batch 16)")
    ap.add_argument("--warm", type=int, default=0, help="untimed runs of every binary before the rounds")
    ap.add_argument("--clear", action="append", default=[], type=Path, help="delete this path before every run")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds before a run is killed")
    ap.add_argument("--out", type=Path, default=Path("runs-bench"), help="scratch folder for runs and logs")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    args, rest = ap.parse_args(argv[:cut]), argv[cut + 1 :]
    bins = [b.split("=", 1) for b in args.bin]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(-args.warm, args.rounds):
        for name, path in bins:
            for c in args.clear:
                clear(c)
            tag = f"{name}-{'w' if r < 0 else 'r'}{abs(r)}"
            cmd = [path, "train", *rest, "--project", str(args.out), "--name", tag, "--json"]
            settle = SETTLE_S if args.warm > 0 else 0.0
            row = {"bin": name, "round": r, **until_step(cmd, args.out / f"{tag}.log", args.until_step, args.timeout, settle)}
            shutil.rmtree(args.out / tag, ignore_errors=True)
            print(json.dumps(row), flush=True)
            if r >= 0:
                rows.append(row)
    print()
    for name, _ in bins:
        v = [r["step_s"] for r in rows if r["bin"] == name and r["step_s"] is not None]
        what = f"step {args.until_step}"
        if v:
            print(f"{name} {what}: median {statistics.median(v):.2f} s, range {min(v):.2f}-{max(v):.2f} (n={len(v)})")
        else:
            print(f"{name} {what}: no data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
