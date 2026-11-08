"""Numpy port of Ultralytics' detection AP, used by coco_reference.py for the comparison in docs/METRICS.md.

Two versions:

- ultralytics 8.4.198 (`utils/metrics.py` `box_iou` / `ap_per_class` / `compute_ap`, `engine/validator.py`
  `match_predictions`): greedy matching in confidence order, and a zero-precision point at the last recall;
- "legacy", the algorithm of earlier 8.x releases: matching with two `np.unique` passes, and the PR curve extended
  straight from the last recall to (1, 0).

Both work in float32 like the torch originals. `check_ultralytics_port.py` checks the 8.4.198 port against a real
install.
"""

import numpy as np


ULTRALYTICS_VERSION = "8.4.198"


def ul_box_iou(a, b, eps=1e-7):
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    inter = wh[..., 0] * wh[..., 1]
    wh1, wh2 = a[:, 2:] - a[:, :2], b[:, 2:] - b[:, :2]
    return inter / (wh1[:, None, 0] * wh1[:, None, 1] + wh2[None, :, 0] * wh2[None, :, 1] - inter + np.float32(eps))


def ul_match(iou, thresholds):
    """8.4.198 greedy matching: detections (columns, descending confidence) each take the best unclaimed target."""
    correct = np.zeros((iou.shape[1], len(thresholds)), bool)
    matched = np.zeros((iou.shape[0], len(thresholds)), dtype=bool)
    for j in np.flatnonzero((iou >= thresholds.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, range(len(thresholds))] >= thresholds
        matched[k, range(len(thresholds))] |= correct[j]
    return correct


def ul_match_legacy(iou, thresholds):
    """Earlier 8.x matching: all pairs above the threshold in descending IoU order, de-duplicated with np.unique."""
    correct = np.zeros((iou.shape[1], len(thresholds)), bool)
    for i, thr in enumerate(thresholds.tolist()):
        matches = np.array(np.nonzero(iou >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def ul_compute_ap(recall, precision, legacy=False):
    if legacy:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x)


def ultralytics_stats(images, legacy=False):
    """Per-image matching as DetectionValidator does it, giving (tp [N, 10], conf, pred_cls, target_cls) over all
    images. Each image's predictions are put in descending confidence order first, as the NMS output is."""
    thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)  # torch.linspace(0.5, 0.95, 10)
    match = ul_match_legacy if legacy else ul_match
    tps, confs, pcls, tcls = [], [], [], []
    for im in images:
        preds = sorted(im["preds"], key=lambda x: -x[4])
        t = np.array([x[4] for x in im["truths"]], np.int64)
        p = np.array([x[5] for x in preds], np.int64)
        tcls.append(t)
        if not len(p):
            continue
        if len(t):
            iou = ul_box_iou([x[:4] for x in im["truths"]], [x[:4] for x in preds])
            tps.append(match(iou * (t[:, None] == p), thresholds))
        else:
            tps.append(np.zeros((len(p), 10), bool))
        confs.append(np.array([x[4] for x in preds], np.float32))
        pcls.append(p)
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0, np.float32)
    pred_cls = np.concatenate(pcls) if pcls else np.zeros(0, np.int64)
    return tp, conf, pred_cls, np.concatenate(tcls)


def run_ultralytics(images, legacy=False):
    tp, conf, pred_cls, target_cls = ultralytics_stats(images, legacy)
    if not len(target_cls):
        return None
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((len(unique_classes), 10))
    for ci, c in enumerate(unique_classes):
        sel = pred_cls == c
        if sel.sum() == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (nt[ci] + 1e-16)
        precision = tpc / (tpc + fpc)
        for j in range(10):
            ap[ci, j] = ul_compute_ap(recall[:, j], precision[:, j], legacy)
    return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean())}
