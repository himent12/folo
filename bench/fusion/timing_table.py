"""Summarise bin/timing.sh's logs: per device, conf and configuration, the median and min-max over rounds of
folo predict's mean post-process and total ms per image, and folo val's seconds."""

import glob
import json
import os
import statistics
import sys


def last_done(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "done":
                return r
    return None


def fmt(v):
    return f"{statistics.median(v):.2f} ({min(v):.2f}-{max(v):.2f})" if v else "-"


def main(d):
    rows = {}
    for p in glob.glob(os.path.join(d, "predict-*.jsonl")):
        dev, conf, c, rnd = os.path.basename(p)[len("predict-"):-len(".jsonl")].split("-")
        r = last_done(p)
        if r is None:
            continue
        t = r["mean_timing"]
        rows.setdefault(("predict", dev, conf, c), []).append((t["postprocess_ms"], t["total_ms"]))
    for p in glob.glob(os.path.join(d, "val-*.json")):
        dev, c, rnd = os.path.basename(p)[len("val-"):-len(".json")].split("-")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        rows.setdefault(("val", dev, "0.001", c), []).append((r["seconds"], r["map50_95"]))
    order = {"base": 0, "off": 1, "dfl": 2, "bstar": 3}
    for k in sorted(rows, key=lambda k: (k[0], k[1], k[2], order[k[3]])):
        v = rows[k]
        a, b = [x[0] for x in v], [x[1] for x in v]
        if k[0] == "predict":
            print(f"{k[0]:<8}{k[1]:<4} conf {k[2]:<6}{k[3]:<6} post ms {fmt(a)}  total ms {fmt(b)}  (n={len(v)})")
        else:
            print(f"{k[0]:<8}{k[1]:<4} conf {k[2]:<6}{k[3]:<6} seconds {fmt(a)}  mAP50-95 {b[0]:.4f}  (n={len(v)})")


if __name__ == "__main__":
    main(sys.argv[1])
