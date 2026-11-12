"""Time `folo train` start-up: process start to the first optimizer step, alternating binaries back to back.

A fresh process per run reads `folo train --json` from a pipe and stops the run (by PID) at the first `step`
record, which the trainer emits after the first optimizer step. Every record before it is printed with its arrival
time, so the phases (dataset open, model build, first batch) can be read off the log.

    python bench/trainer/startup.py --bin base=D:/.../base/folo.exe --bin new=D:/.../new/folo.exe --rounds 3 \
        --out D:/folo-data/runs-hill2/trainer/startup -- \
        --data D:/folo-data/coco --model folo-n --epochs 1 --batch 16 --imgsz 640 --device gpu:0

Everything after `--` goes to `folo train` unchanged. `--warm 1` runs every binary once before the timed rounds
(a second start, with whatever caches the first one wrote); `--clear PATH` deletes PATH before every run (a cold
start for caches kept there).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def first_step(cmd: list[str], log: Path, timeout_s: float) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    row: dict = {"first_step_s": None}
    try:
        with log.open("w", encoding="utf-8") as f:
            assert proc.stdout is not None
            for line in proc.stdout:
                t = time.perf_counter() - t0
                f.write(f"{t:8.3f} {line}")
                if t > timeout_s:
                    break
                s = line.strip()
                if not s.startswith("{"):
                    continue
                try:
                    r = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "step":
                    row["first_step_s"] = round(t, 3)
                    break
    finally:
        proc.kill()
        proc.wait()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", action="append", required=True, metavar="NAME=PATH", help="a folo binary (repeat)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warm", type=int, default=0, help="untimed runs of every binary before the rounds")
    ap.add_argument("--clear", action="append", default=[], type=Path, help="delete this path before every run")
    ap.add_argument("--timeout", type=float, default=600.0)
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
                shutil.rmtree(c, ignore_errors=True) if c.is_dir() else c.unlink(missing_ok=True)
            tag = f"{name}-{'w' if r < 0 else 'r'}{abs(r)}"
            cmd = [path, "train", *rest, "--project", str(args.out), "--name", tag, "--json"]
            row = {"bin": name, "round": r, **first_step(cmd, args.out / f"{tag}.log", args.timeout)}
            shutil.rmtree(args.out / tag, ignore_errors=True)
            print(json.dumps(row), flush=True)
            if r >= 0:
                rows.append(row)
    print()
    for name, _ in bins:
        v = [r["first_step_s"] for r in rows if r["bin"] == name and r["first_step_s"] is not None]
        if v:
            print(f"{name} first step: median {statistics.median(v):.2f} s, range {min(v):.2f}-{max(v):.2f} (n={len(v)})")
        else:
            print(f"{name} first step: no data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
