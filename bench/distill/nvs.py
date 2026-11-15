"""counsel-b: is folo-s a better LOCALISER than folo-n on the same objects? CPU only, existing dumps."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fusion"))
import coco
from dump import Dump
from post import iou_matrix, postprocess
GT = "D:/folo-data/coco/annotations/instances_val2017.json"
DN = "D:/folo-data/runs-hill3/fusion/dumps/land-val"
DS = "D:/folo-data/runs-hill3/fusion/dumps/s-val"
CFG = {"conf": 0.001, "iou": 0.7, "max_det": 300}
_g = None
def _init():
    global _g
    g = json.load(open(GT, encoding="utf-8"))
    cats = sorted(c["id"] for c in g["categories"]); cls_of = {c: i for i, c in enumerate(cats)}
    id_of = {im["file_name"]: im["id"] for im in g["images"]}
    by = {}
    for a in g["annotations"]:
        if a.get("iscrowd", 0): continue
        x, y, w, h = a["bbox"]
        by.setdefault(a["image_id"], []).append((x, y, x + w, y + h, cls_of[a["category_id"]], a["area"]))
    dn, ds = Dump(DN), Dump(DS)
    sidx = {m["file"]: i for i, m in enumerate(ds.images)}
    _g = (id_of, by, dn, ds, sidx)
def prim(boxes, scores, classes, tb, tc):
    """per truth: index of the top-scoring same-class detection at IoU>=0.5, or -1; and its iou"""
    T = len(tb); out = -np.ones(T, np.int64); io = np.zeros(T)
    if len(boxes) == 0 or T == 0: return out, io
    m = iou_matrix(tb, boxes); ok = (tc[:, None] == classes[None, :]) & (m >= 0.5)
    sc = np.where(ok, scores[None, :], -1.0); j = sc.argmax(1); has = sc[np.arange(T), j] > 0
    out[has] = j[has]; io[has] = m[np.arange(T), j][has]
    return out, io
def _work(r):
    lo, hi = r; id_of, by, dn, ds, sidx = _g
    rows = []; res = {k: [] for k in ("n", "s", "n_sbox", "s_nbox", "n_sscore", "s_nscore")}; files = []
    for i in range(lo, hi):
        a = dn.image(i); b = ds.image(sidx[a.file]); files.append(a.file)
        t = np.array(by.get(id_of[a.file], []), dtype=np.float64).reshape(-1, 6)
        tb, tc, ta = t[:, :4].astype(np.float32), t[:, 4].astype(np.int64), t[:, 5]
        nb, ns, nc = postprocess(a, CFG); sb, ss, sc = postprocess(b, CFG)
        pn, ion = prim(nb, ns, nc, tb, tc); ps, ios = prim(sb, ss, sc, tb, tc)
        nb2, sb2, ns2, ss2 = nb.copy(), sb.copy(), ns.copy(), ss.copy()
        for g in range(len(tb)):
            en = np.abs(nb[pn[g]] - tb[g]) if pn[g] >= 0 else np.full(4, np.nan)
            es = np.abs(sb[ps[g]] - tb[g]) if ps[g] >= 0 else np.full(4, np.nan)
            side = np.sqrt(max((tb[g, 2] - tb[g, 0]) * (tb[g, 3] - tb[g, 1]), 1e-6))
            rows.append((ta[g], side, pn[g] >= 0, ps[g] >= 0, ion[g], ios[g], np.mean(en), np.mean(es),
                         ns[pn[g]] if pn[g] >= 0 else 0, ss[ps[g]] if ps[g] >= 0 else 0, np.median(en), np.median(es)))
            if pn[g] >= 0 and ps[g] >= 0:
                nb2[pn[g]] = sb[ps[g]]; sb2[ps[g]] = nb[pn[g]]; ns2[pn[g]] = ss[ps[g]]; ss2[ps[g]] = ns[pn[g]]
        res["n"].append((nb, ns, nc)); res["s"].append((sb, ss, sc))
        res["n_sbox"].append((nb2, ns, nc)); res["s_nbox"].append((sb2, ss, sc))
        res["n_sscore"].append((nb, ns2, nc)); res["s_nscore"].append((sb, ss2, sc))
    return rows, res, files
def main():
    n = len(Dump(DN)); jobs = [(lo, min(lo + 125, n)) for lo in range(0, n, 125)]
    rows = []; res = {}; files = []
    with ProcessPoolExecutor(10, initializer=_init) as ex:
        for r, rs, f in ex.map(_work, jobs):
            rows += r; files += f
            for k, v in rs.items(): res.setdefault(k, []).extend(v)
    R = np.array(rows, dtype=np.float64); np.save("nvs_rows.npy", R)
    truth = coco.Truth(GT)
    for k, v in res.items():
        s = coco.evaluate(truth, coco.detections_array(truth, files, v)).stats
        print(f"{k:<9} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S {s[3]:.4f} M {s[4]:.4f} L {s[5]:.4f}", flush=True)
    area = R[:, 0]
    for name, m in (("S", area < 32**2), ("M", (area >= 32**2) & (area < 96**2)), ("L", area >= 96**2), ("all", area > 0)):
        hn, hs = R[:, 2] > 0, R[:, 3] > 0; both = m & hn & hs
        print(f"{name}: truths {m.sum()} n-found {(m&hn).mean()/m.mean():.3f} s-found {(m&hs).mean()/m.mean():.3f} both {both.sum()} "
              f"only-n {(m&hn&~hs).sum()} only-s {(m&~hn&hs).sum()} | on both: IoU n {R[both,4].mean():.4f} s {R[both,5].mean():.4f} "
              f"IoU>=.75 n {(R[both,4]>=.75).mean():.3f} s {(R[both,5]>=.75).mean():.3f} IoU>=.9 n {(R[both,4]>=.9).mean():.3f} s {(R[both,5]>=.9).mean():.3f} "
              f"| mean|edge| n {R[both,6].mean():.2f} s {R[both,7].mean():.2f} median n {np.median(R[both,6]):.2f} s {np.median(R[both,7]):.2f} "
              f"rel(edge/side) n {np.median(R[both,6]/R[both,1]):.4f} s {np.median(R[both,7]/R[both,1]):.4f} "
              f"| score n {R[both,8].mean():.3f} s {R[both,9].mean():.3f} median n {np.median(R[both,8]):.3f} s {np.median(R[both,9]):.3f} "
              f"| s closer {(R[both,7]<R[both,6]).mean():.3f}")
if __name__ == "__main__":
    main()
