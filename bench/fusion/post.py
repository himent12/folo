"""FOLO's post-process re-implemented on a candidate dump (dump.py), and the variants of the fusion lane's ladder.

`postprocess(im, cfg)` returns (boxes in original-image pixels (n, 4) f32, scores (n,) f32, classes (n,) int) for
one image. With the default config it is MODEL.md §9.3's validation protocol, bit for bit what `folo val --coco`
scores (nms.rs: score order with ties by (anchor, class) position, strict `> iou`, per-class buckets stopping at
`max_det` survivors, the 30000-candidate cap, f32 IoU in the same arithmetic order).

Config keys (all optional):
    conf 0.001, iou 0.7, max_det 300, max_cand 30000
    nms: "hard" | "soft" | "matrix"; soft_sigma 0.5; matrix_sigma 2.0, matrix_kernel "gaussian" | "linear"
    level_offset: [d3, d4, d5] logit offsets per level (score' = sigmoid(logit(score) + d_level))
    dfl_T: softmax temperature of the DFL decode (1 = today); dfl_window: expectation within +-w bins of the argmax
    fuse: None, or {"kind": "vote" | "var" | "lattice", "tau": cluster IoU (>=), "a": score power,
          "sigma": IoU kernel width (None = flat), "b": variance power, "excess": bool, "eps": bins^2,
          "censor_c": power of (1 - p_end), "censor_thr": drop sides with p_end above, "levels": "all" | "same"}
"""

import numpy as np

from dump import unmap

F32 = np.float32


def _areas(b):
    return np.maximum(b[:, 2] - b[:, 0], F32(0)) * np.maximum(b[:, 3] - b[:, 1], F32(0))


def iou_one(b, area_b, c, area_c):
    """IoU of box b with boxes c, f32, the arithmetic of nms_sorted / box_iou."""
    w = np.maximum(np.minimum(b[2], c[:, 2]) - np.maximum(b[0], c[:, 0]), F32(0))
    h = np.maximum(np.minimum(b[3], c[:, 3]) - np.maximum(b[1], c[:, 1]), F32(0))
    inter = w * h
    union = area_b + area_c - inter
    out = np.zeros(len(c), dtype=F32)
    pos = union > 0
    out[pos] = inter[pos] / union[pos]
    return out


def iou_matrix(a, b):
    w = np.maximum(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), F32(0))
    h = np.maximum(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), F32(0))
    inter = w * h
    union = _areas(a)[:, None] + _areas(b)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter / union, F32(0))
    return out.astype(F32)


def score_order(scores):
    """Descending score, ties by ascending position (nms.rs score_order)."""
    return np.lexsort((np.arange(len(scores)), -scores.astype(np.float64)))


def nms_sorted(boxes, thr, limit):
    n = len(boxes)
    keep = []
    if n == 0 or limit == 0:
        return keep
    area = _areas(boxes)
    sup = np.zeros(n, dtype=bool)
    thr = F32(thr)
    for i in range(n):
        if sup[i]:
            continue
        keep.append(i)
        if len(keep) == limit:
            break
        if i + 1 < n:
            sup[i + 1:] |= iou_one(boxes[i], area[i], boxes[i + 1:], area[i + 1:]) > thr
    return keep


def batched_nms(boxes, scores, classes, thr, max_det):
    """Indices of the kept candidates, by descending score (ties: lower index), at most max_det."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = score_order(scores)
    kept = []
    oc = classes[order]
    for c in np.unique(oc):
        ids = order[oc == c]  # score order inside the bucket (stable)
        k = nms_sorted(boxes[ids], thr, max_det)
        kept.extend(ids[k].tolist())
    kept = np.array(kept, dtype=np.int64)
    kept = kept[score_order(scores[kept])]
    # score_order ranks by position in `kept`; ties must rank by candidate index: re-sort stably.
    kept = kept[np.lexsort((kept, -scores[kept].astype(np.float64)))]
    return kept[:max_det]


def soft_nms(boxes, scores, classes, sigma, conf, max_det):
    """Gaussian soft-NMS per class: returns (indices, new scores)."""
    idx_all, sc_all = [], []
    for c in np.unique(classes):
        ids = np.nonzero(classes == c)[0]
        s = scores[ids].astype(np.float64).copy()
        b = boxes[ids]
        area = _areas(b)
        alive = np.ones(len(ids), dtype=bool)
        while alive.any() and len(idx_all) < 10 ** 9:
            cand = np.nonzero(alive)[0]
            j = cand[np.argmax(s[cand])]  # first maximum: lower position on ties
            idx_all.append(ids[j])
            sc_all.append(s[j])
            alive[j] = False
            rest = np.nonzero(alive)[0]
            if len(rest) == 0:
                break
            iou = iou_one(b[j], area[j], b[rest], area[rest]).astype(np.float64)
            s[rest] *= np.exp(-(iou * iou) / sigma)
            alive[rest[s[rest] <= conf]] = False
    idx_all = np.array(idx_all, dtype=np.int64)
    sc_all = np.array(sc_all, dtype=np.float64)
    o = np.lexsort((idx_all, -sc_all))[:max_det]
    return idx_all[o], sc_all[o].astype(F32)


def matrix_nms(boxes, scores, classes, sigma, kernel, conf, max_det):
    idx_all, sc_all = [], []
    for c in np.unique(classes):
        ids = np.nonzero(classes == c)[0]
        ids = ids[score_order(scores[ids])]
        b = boxes[ids]
        iou = np.triu(iou_matrix(b, b).astype(np.float64), k=1)
        comp = iou.max(axis=0)  # each box's largest IoU with a higher-scored one
        if kernel == "gaussian":
            decay = np.exp(-sigma * (iou ** 2 - comp[:, None] ** 2))
        else:
            decay = (1 - iou) / (1 - comp[:, None])
        decay = np.where(np.triu(np.ones_like(iou, dtype=bool), k=1), decay, np.inf).min(axis=0)
        decay = np.minimum(decay, 1.0)
        s = scores[ids].astype(np.float64) * decay
        ok = s > conf
        idx_all.extend(ids[ok].tolist())
        sc_all.extend(s[ok].tolist())
    idx_all = np.array(idx_all, dtype=np.int64)
    sc_all = np.array(sc_all, dtype=np.float64)
    o = np.lexsort((idx_all, -sc_all))[:max_det]
    return idx_all[o], sc_all[o].astype(F32)


def side_stats(logits, T=1.0):
    """Per anchor and side: probabilities (n, 4, R), expectation and variance in bins."""
    z = logits.astype(np.float64) / T
    z = z - z.max(axis=2, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=2, keepdims=True)
    k = np.arange(logits.shape[2], dtype=np.float64)
    mean = (p * k).sum(axis=2)
    var = np.maximum((p * k * k).sum(axis=2) - mean * mean, 0.0)
    return p, mean, var


def window_expectation(logits, w):
    """Expectation within +-w bins of each side's argmax (renormalised)."""
    p, _, _ = side_stats(logits)
    R = logits.shape[2]
    am = p.argmax(axis=2)
    k = np.arange(R)
    mask = np.abs(k[None, None, :] - am[:, :, None]) <= w
    q = np.where(mask, p, 0.0)
    q /= q.sum(axis=2, keepdims=True)
    return (q * k).sum(axis=2)


def boxes_from_dist(im, d):
    """Network boxes from ltrb distances in bins (decode::dist2bbox then scale by the stride), f32."""
    d = d.astype(F32)
    s = im.stride
    px, py = im.point[:, 0], im.point[:, 1]
    return np.stack([(px - d[:, 0]) * s, (py - d[:, 1]) * s, (px + d[:, 2]) * s, (py + d[:, 3]) * s], axis=1)


def _edges(box):
    return box  # x1 (left), y1 (top), x2 (right), y2 (bottom): the four sides in (l, t, r, b) order


def fuse(im, box_net, kept, cand_anchor, cand_class, cand_score, f):
    """The fused network boxes of the kept candidates (indices into the candidate arrays)."""
    kind = f["kind"]
    tau = F32(f.get("tau", 0.7))
    a = f.get("a", 1.0)
    sigma = f.get("sigma")
    out = box_net[cand_anchor[kept]].copy()
    need_p = kind in ("var", "lattice") or f.get("censor_c") or f.get("censor_thr") is not None
    if need_p:
        p, mean, var = side_stats(im.logits)
    for c in np.unique(cand_class[kept]):
        kk = np.nonzero(cand_class[kept] == c)[0]  # positions in `kept`
        mem = np.nonzero(cand_class == c)[0]
        ka = cand_anchor[kept[kk]]
        ma = cand_anchor[mem]
        iou = iou_matrix(box_net[ka], box_net[ma]).astype(np.float64)  # K x M
        inside = iou >= tau
        if f.get("levels", "all") == "same":
            inside &= im.level[ka][:, None] == im.level[ma][None, :]
        base = np.where(inside, cand_score[mem][None, :].astype(np.float64) ** a, 0.0)
        if sigma:
            base = base * np.exp(-((1 - iou) ** 2) / sigma)
        x = box_net[ma].astype(np.float64)  # M x 4
        if kind == "lattice":
            for r, k in enumerate(kk):
                out[k] = _lattice(im, p, ka[r], ma, base[r])
            continue
        w = np.repeat(base[:, :, None], 4, axis=2)  # K x M x 4
        if kind == "var":
            v = var[ma]  # M x 4, bins^2
            if f.get("excess"):
                fr = mean[ma] - np.floor(mean[ma])
                v = np.maximum(v - fr * (1 - fr), 0.0)
            s2 = (im.stride[ma].astype(np.float64) ** 2)[:, None]
            v = s2 * (v + f.get("eps", 0.01))
            w = w / (v[None, :, :] ** f.get("b", 1.0))
        if f.get("censor_c") or f.get("censor_thr") is not None:
            pend = np.maximum(p[ma][:, :, -1], p[ma][:, :, 0])  # M x 4
            if f.get("censor_c"):
                w = w * ((1 - pend) ** f["censor_c"])[None, :, :]
            if f.get("censor_thr") is not None:
                w = w * (pend <= f["censor_thr"])[None, :, :]
        sw = w.sum(axis=1)  # K x 4
        fused = np.einsum("kme,me->ke", w, x) / np.where(sw > 0, sw, 1.0)
        keep_old = sw <= 0
        old = box_net[ka].astype(np.float64)
        fused = np.where(keep_old, old, fused)
        out[kk] = fused.astype(F32)
    return out


def _lattice(im, p, k_anchor, ma, base):
    """Pool the members' side distributions as prod p_i^w_i (sum w = 1) on the kept anchor's bin lattice."""
    sel = base > 0
    ma, w = ma[sel], base[sel]
    w = w / w.sum()
    R = p.shape[2]
    sk = float(im.stride[k_anchor])
    ak = im.point[k_anchor].astype(np.float64) * sk
    res = np.empty(4)
    for e in range(4):
        axis = e % 2
        sign = -1.0 if e < 2 else 1.0
        # Grid: the kept anchor's lattice, extended to cover every member's range.
        lo, hi = 0.0, R - 1.0
        for m in ma:
            am = float(im.point[m][axis]) * float(im.stride[m])
            sm = float(im.stride[m])
            # member positions am + sign * j * sm, j = 0..R-1, in kept-lattice units
            j0 = sign * (am - ak[axis]) / sk
            j1 = sign * (am + sign * (R - 1) * sm - ak[axis]) / sk
            lo, hi = min(lo, j0, j1), max(hi, j0, j1)
        grid = np.arange(np.floor(lo), np.ceil(hi) + 1)
        pos = ak[axis] + sign * grid * sk
        logq = np.zeros(len(grid))
        for m, wm in zip(ma, w):
            am = float(im.point[m][axis]) * float(im.stride[m])
            sm = float(im.stride[m])
            mpos = am + sign * np.arange(R) * sm
            o = np.argsort(mpos)
            pm = np.interp(pos, mpos[o], p[m, e][o], left=0.0, right=0.0)
            logq += wm * np.log(np.maximum(pm, 1e-12))
        q = np.exp(logq - logq.max())
        q /= q.sum()
        res[e] = (q * pos).sum()
    return res.astype(F32)


def postprocess(im, cfg):
    conf = cfg.get("conf", 0.001)
    iou = cfg.get("iou", 0.7)
    max_det = cfg.get("max_det", 300)
    max_cand = cfg.get("max_cand", 30000)

    box_net = im.box_net
    box_orig = im.box_orig
    T, win = cfg.get("dfl_T", 1.0), cfg.get("dfl_window")
    if T != 1.0 or win:
        if win:
            d = window_expectation(im.logits, win)
        else:
            _, d, _ = side_stats(im.logits, T)
        box_net = boxes_from_dist(im, d)
        box_orig = None

    ca, cc, cs = im.pair_anchor, im.pair_class, im.pair_score
    if cfg.get("single_label"):
        # predict's candidates: each anchor's best class (first maximum), from its pairs (every class > 0.001, so
        # the best of them is the anchor's best whenever it can pass a predict conf).
        order = np.lexsort((cc, -cs.astype(np.float64), ca))
        first = np.ones(len(order), dtype=bool)
        first[1:] = ca[order][1:] != ca[order][:-1]
        sel = np.sort(order[first])
        ca, cc, cs = ca[sel], cc[sel], cs[sel]
    off = cfg.get("level_offset")
    if off:
        lg = np.log(cs.astype(np.float64)) - np.log1p(-cs.astype(np.float64))
        lg = lg + np.asarray(off, dtype=np.float64)[im.level[ca]]
        cs = (1.0 / (1.0 + np.exp(-lg))).astype(F32)
    ok = cs > F32(conf)
    ca, cc, cs = ca[ok], cc[ok], cs[ok]
    if len(cs) > max_cand:
        pos = np.arange(len(cs))
        top = np.lexsort((cc, ca, -cs.astype(np.float64)))[:max_cand]
        top = np.sort(top)  # back to (anchor, class) order
        ca, cc, cs = ca[top], cc[top], cs[top]
        del pos
    boxes = box_net[ca]
    kind = cfg.get("nms", "hard")
    if kind == "hard":
        kept = batched_nms(boxes, cs, cc, iou, max_det)
        scores = cs[kept]
    elif kind == "soft":
        kept, scores = soft_nms(boxes, cs, cc, cfg.get("soft_sigma", 0.5), conf, max_det)
    elif kind == "matrix":
        kept, scores = matrix_nms(boxes, cs, cc, cfg.get("matrix_sigma", 2.0), cfg.get("matrix_kernel", "gaussian"),
                                  conf, max_det)
    else:
        raise ValueError(kind)
    f = cfg.get("fuse")
    if f:
        net = fuse(im, box_net, kept, ca, cc, cs, f)
        orig = unmap(net, im.ratio, im.pad, im.orig)
    elif box_orig is None:
        orig = unmap(boxes[kept], im.ratio, im.pad, im.orig)
    else:
        orig = box_orig[ca[kept]]
    return orig, scores.astype(F32), cc[kept]
