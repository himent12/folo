"""pycocotools bbox evaluation of a detections file (FOLO's `folo val --coco --save-json`, or ultra_train.py's
dets.json) against a COCO ground-truth file. Prints the 12 COCO stats as one JSON line.

    python bench/learn/coco_eval.py --gt <set>/annotations/instances_val2017.json --dt dets.json
"""

import argparse
import contextlib
import io
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    a = ap.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
        dt = gt.loadRes(a.dt)
        e = COCOeval(gt, dt, "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    names = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L"]
    print(json.dumps({k: round(float(v), 4) for k, v in zip(names, e.stats)}))


if __name__ == "__main__":
    main()
