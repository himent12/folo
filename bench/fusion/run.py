"""Score post-processing configurations on a candidate dump.

    python run.py --dump <dir> --gt <instances json> --configs <json file or inline json list> --out <json>
                  [--workers 8] [--records <dir>] [--save-json <file>]

Each configuration is {"name": ..., **post.postprocess keys}. Prints and writes the 12 COCO stats per
configuration; --records keeps each configuration's match records (coco.match_records) as a pickle for the
bootstrap (bootstrap.py); --save-json writes the first configuration's detections as a COCO results file.
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coco  # noqa: E402
from dump import Dump  # noqa: E402
from post import postprocess  # noqa: E402

_dump = None


def _init(path):
    global _dump
    _dump = Dump(path)


def _work(args):
    cfgs, lo, hi = args
    out = []
    for cfg in cfgs:
        res = []
        for i in range(lo, hi):
            res.append(postprocess(_dump.image(i), cfg))
        out.append(res)
    return out


STAT_NAMES = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L"]


def run(dump_path, gt, cfgs, workers=8, chunk=125):
    d = Dump(dump_path)
    n = len(d)
    files = [m["file"] for m in d.images]
    jobs = [(cfgs, lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    per_cfg = [[] for _ in cfgs]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dump_path,)) as ex:
        for part in ex.map(_work, jobs):
            for k, r in enumerate(part):
                per_cfg[k].extend(r)
    return files, per_cfg


_truths = {}


def _eval(args):
    gt, dets, want = args
    if gt not in _truths:
        _truths[gt] = coco.Truth(gt)
    ev = coco.evaluate(_truths[gt], dets)
    return [float(x) for x in ev.stats], (coco.match_records(ev) if want else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-workers", type=int, default=6)
    ap.add_argument("--records")
    ap.add_argument("--save-json")
    a = ap.parse_args()
    if os.path.exists(a.configs):
        with open(a.configs, encoding="utf-8") as f:
            cfgs = json.load(f)
    else:
        cfgs = json.loads(a.configs)
    t0 = time.time()
    truth = coco.Truth(a.gt)
    files, per_cfg = run(a.dump, truth, cfgs, a.workers)
    t_post = time.time() - t0
    rows = []
    all_dets = [coco.detections_array(truth, files, res) for res in per_cfg]
    del per_cfg
    if a.save_json:
        with open(a.save_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"image_id": int(r[0]), "category_id": int(r[6]), "bbox": [r[1], r[2], r[3], r[4]],
                        "score": r[5]} for r in all_dets[0]], f)
    # COCOeval is single-threaded: one process per configuration.
    jobs = [(a.gt, d, bool(a.records)) for d in all_dets]
    with ProcessPoolExecutor(max(1, min(a.eval_workers, len(jobs)))) as ex:
        evals = list(ex.map(_eval, jobs))
    for cfg, dets, (st, recs) in zip(cfgs, all_dets, evals):
        stats = dict(zip(STAT_NAMES, st))
        row = {"name": cfg.get("name", ""), "cfg": cfg, "stats": stats, "detections": int(len(dets))}
        rows.append(row)
        print(f"{row['name']:<40} AP {stats['AP']:.4f} AP50 {stats['AP50']:.4f} AP75 {stats['AP75']:.4f} "
              f"S {stats['AP_S']:.4f} M {stats['AP_M']:.4f} L {stats['AP_L']:.4f} AR100 {stats['AR100']:.4f}",
              flush=True)
        if a.records:
            os.makedirs(a.records, exist_ok=True)
            with open(os.path.join(a.records, f"{row['name']}.pkl"), "wb") as f:
                pickle.dump(recs, f)
    print(f"post-process {t_post:.0f} s, total {time.time() - t0:.0f} s", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"dump": a.dump, "gt": a.gt, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
