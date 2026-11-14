"""Score a trained refiner (or B* alone) on a feature dump: COCO AP with pycocotools, match records for the paired
bootstrap (bench/fusion/bootstrap.py), and edge-error diagnostics.

    python evaluate.py --dump <dir> --boxes <npz> --gt <instances json> --out <json> [--records <dir>]
                       [--model <run>/refiner.pt ...] [--name <label> ...] [--baseline]

With --baseline, B* itself (the table's boxes) is scored as `Bstar`. Each --model is scored under its --name.
Diagnostics are over the B* boxes whose same-class truth overlaps at IoU >= 0.5: mean absolute edge error in
network pixels and in units of s_e, before and after, the share of it closed, by COCO size (truth area in original
pixels: S < 32^2 <= M < 96^2 <= L) and by the kept candidate's level, and the mean IoU before and after.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fusion"))

import coco  # noqa: E402
from data import FeatureDump, loader, to_device  # noqa: E402
from dump import unmap  # noqa: E402
from refiner import box_iou  # noqa: E402
from stem import Stem  # noqa: E402
from train import WEIGHTS, build, fine_inputs  # noqa: E402

STAT_NAMES = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L"]


@torch.no_grad()
def refine_all(dump, model, dev, batch=16, workers=6):
    """Refined network boxes and probe scales for every row of the table (all kept boxes)."""
    n_rows = len(dump.t["cls"])
    out = np.zeros((n_rows, 4), dtype=np.float32)
    scale = np.zeros((n_rows, 4), dtype=np.float32)
    qual = np.full(n_rows, np.nan, dtype=np.float32)
    mean = np.zeros((n_rows, 4), dtype=np.float32)
    std = np.zeros((n_rows, 4), dtype=np.float32)
    model.eval()
    stem = Stem(WEIGHTS).to(dev)
    for idx, pyr, rows, inp in loader(dump, list(range(dump.n)), batch, None, workers):
        b = to_device(rows, dev)
        if b["box"].shape[0] == 0:
            continue
        pyr = [torch.as_tensor(p).to(dev) for p in pyr]
        proj = model.project(pyr) if model.features else None
        logits, s, q = model(proj, b, fine_inputs(model, stem, inp, dev))
        ref = model.refine(b["box"], logits, s)
        r = np.asarray(rows["row"])
        out[r] = ref.cpu().numpy()
        scale[r] = s.cpu().numpy()
        p = logits.float().softmax(-1)
        m1 = (p * model.bins).sum(-1)
        mean[r] = m1.cpu().numpy()
        std[r] = ((p * model.bins ** 2).sum(-1) - m1 ** 2).clamp(min=0).sqrt().cpu().numpy()
        if q is not None:
            qual[r] = torch.sigmoid(q).cpu().numpy()
    return out, scale, qual, mean, std


def detections(dump, boxes, scores=None, renms=None):
    """Per image (orig boxes, scores, classes) for coco.detections_array."""
    t = dump.t
    res = []
    sc_all = t["score"] if scores is None else scores
    for i in range(dump.n):
        s0, n = t["img_start"][i], t["img_count"][i]
        b = boxes[s0:s0 + n]
        sc = sc_all[s0:s0 + n].astype(np.float32)
        cl = t["cls"][s0:s0 + n].astype(np.int64)
        if renms is not None and n:
            import post
            k = post.batched_nms(b, sc, cl, renms, 1000)
            b, sc, cl = b[k], sc[k], cl[k]
        orig = unmap(b, t["ratio"][i], t["pad"][i], t["orig"][i])
        res.append((orig, sc, cl))
    return res


def diagnostics(dump, before, after, scale):
    t = dump.t
    m = t["gt_iou"] >= 0.5
    gt = t["gt"][m]
    b0, b1, s = before[m], after[m], scale[m]
    e0, e1 = np.abs(b0 - gt), np.abs(b1 - gt)
    img = np.repeat(np.arange(dump.n), t["img_count"])[m]
    ratio = t["ratio"][img]
    area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) / (ratio[:, 0] * ratio[:, 1])
    size = np.where(area < 32 ** 2, "S", np.where(area < 96 ** 2, "M", "L"))
    lvl = t["level"][m]
    iou0 = box_iou(torch.from_numpy(b0), torch.from_numpy(gt)).numpy()
    iou1 = box_iou(torch.from_numpy(b1), torch.from_numpy(gt)).numpy()

    def row(sel):
        if sel.sum() == 0:
            return None
        a, c = e0[sel].mean(), e1[sel].mean()
        return {"boxes": int(sel.sum()), "err_px": [float(a), float(c)],
                "err_s": [float((e0[sel] / s[sel]).mean()), float((e1[sel] / s[sel]).mean())],
                "closed": float(1 - c / a), "iou": [float(iou0[sel].mean()), float(iou1[sel].mean())],
                "improved": float((iou1[sel] > iou0[sel]).mean())}

    out = {"all": row(np.ones(len(gt), dtype=bool))}
    for z in "SML":
        out[z] = row(size == z)
    for l in range(3):
        out[f"P{l + 3}"] = row(lvl == l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--inputs", action="store_true", help="read <dump>-in/input.bin (experiment H refiners)")
    ap.add_argument("--renms", type=float, help="NMS again after refinement at this IoU")
    ap.add_argument("--quality-gamma", type=float, help="score * q^gamma with the model's quality output")
    ap.add_argument("--save-boxes", help="write the refined network boxes of the first model here (npy)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    dump = FeatureDump(a.dump, a.boxes, a.inputs)
    truth = coco.Truth(a.gt)
    files = [m["file"] for m in dump.images]
    results = {}
    if a.baseline:
        results["Bstar"] = (detections(dump, dump.t["box"]), None)
    for k, (path, name) in enumerate(zip(a.model, a.name)):
        st = torch.load(path, map_location=dev)
        model = build(st["config"]).to(dev)
        model.load_state_dict(st["model"])
        ref, scale, qual, mean, std = refine_all(dump, model, dev)
        if k == 0 and a.save_boxes:
            np.save(a.save_boxes, ref)
            np.savez(a.save_boxes[:-4] + "-dist.npz", scale=scale, mean=mean, std=std, qual=qual)
        diag = diagnostics(dump, dump.t["box"], ref, scale)
        results[name] = (detections(dump, ref, renms=a.renms), diag)
        if a.quality_gamma is not None and not np.isnan(qual).all():
            sc = dump.t["score"] * np.power(np.clip(qual, 1e-6, 1), a.quality_gamma)
            results[name + f"-q{a.quality_gamma}"] = (detections(dump, ref, scores=sc), None)
    rows = {}
    for name, (res, diag) in results.items():
        dets = coco.detections_array(truth, files, res)
        ev = coco.evaluate(truth, dets)
        stats = dict(zip(STAT_NAMES, [float(x) for x in ev.stats]))
        rows[name] = {"stats": stats, "detections": int(len(dets)), "diag": diag}
        print(f"{name:<28} AP {stats['AP']:.4f} AP50 {stats['AP50']:.4f} AP75 {stats['AP75']:.4f} "
              f"S {stats['AP_S']:.4f} M {stats['AP_M']:.4f} L {stats['AP_L']:.4f}", flush=True)
        if diag:
            for k, v in diag.items():
                if v:
                    print(f"    {k:<4} n {v['boxes']:>7}  err px {v['err_px'][0]:6.2f} -> {v['err_px'][1]:6.2f}  "
                          f"err/s {v['err_s'][0]:.3f} -> {v['err_s'][1]:.3f}  closed {v['closed']:+.3f}  "
                          f"IoU {v['iou'][0]:.4f} -> {v['iou'][1]:.4f}  improved {v['improved']:.3f}", flush=True)
        if a.records:
            os.makedirs(a.records, exist_ok=True)
            with open(os.path.join(a.records, f"{name}.pkl"), "wb") as f:
                pickle.dump(coco.match_records(ev), f)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
