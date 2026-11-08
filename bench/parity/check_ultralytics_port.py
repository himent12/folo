"""Check coco_reference.py's numpy port of Ultralytics' AP against a real ultralytics install.

Run it with a Python that has `ultralytics` (and torch) installed. The script only imports it, it installs nothing:

    <python-with-ultralytics> bench/parity/check_ultralytics_port.py

For every fixture in crates/folo-vision/tests/fixtures/coco it runs ultralytics' own `box_iou`,
`DetectionValidator.match_predictions` and `ap_per_class` on the fixture's images. It then compares the TP matrix
and mAP50 / mAP50-95 with the port, and with the `ultralytics` numbers stored in the fixture.
"""

import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics_port import ULTRALYTICS_VERSION, run_ultralytics, ultralytics_stats  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "folo-vision", "tests", "fixtures", "coco")


def real_stats(images):
    v = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    tps, confs, pcls, tcls = [], [], [], []
    for im in images:
        preds = sorted(im["preds"], key=lambda x: -x[4])
        t = torch.tensor([x[4] for x in im["truths"]], dtype=torch.int64)
        p = torch.tensor([x[5] for x in preds], dtype=torch.int64)
        tcls.append(t.numpy())
        if not len(p):
            continue
        if len(t):
            iou = box_iou(torch.tensor([x[:4] for x in im["truths"]]), torch.tensor([x[:4] for x in preds]))
            tps.append(BaseValidator.match_predictions(v, p, t, iou).numpy())
        else:
            tps.append(np.zeros((len(p), 10), bool))
        confs.append(np.array([x[4] for x in preds], np.float32))
        pcls.append(p.numpy())
    return np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(tcls)


def main():
    assert ultralytics.__version__ == ULTRALYTICS_VERSION, f"ported {ULTRALYTICS_VERSION}, found {ultralytics.__version__}"
    worst = 0.0
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
        fx = json.load(open(path))
        images = fx["images"]
        tp, conf, pred_cls, target_cls = real_stats(images)
        if not len(target_cls):
            continue
        tp_port = ultralytics_stats(images)[0]
        assert (tp == tp_port).all(), f"{fx['name']}: TP matrix differs"
        ap = ap_per_class(tp, conf, pred_cls, target_cls)[5]
        real = {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean())}
        port = run_ultralytics(images)
        d = max(abs(real[k] - port[k]) for k in real)
        d_fixture = max(abs(real[k] - fx["ultralytics"][k]) for k in real)
        worst = max(worst, d, d_fixture)
        print(f"{fx['name']:22} real AP {real['map50_95']:.6f} AP50 {real['map50']:.6f}  |port - real| {d:.1e}  "
              f"|fixture - real| {d_fixture:.1e}")
    print(f"worst difference {worst:.1e}")
    assert worst < 1e-12


if __name__ == "__main__":
    main()
