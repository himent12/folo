#!/usr/bin/env python3
"""Run any vision model on a FOLO benchmark and submit the entry (docs/BENCH.md, "Who can run").

Fill in `predict` below (an Ultralytics example is in the comments), then:

    python examples/bench_external.py --bench bn_3f9a --name my-detector
    python examples/bench_external.py --server http://127.0.0.1:7878 --bench bn_3f9a --name fast --no-conditions

The script fetches every image of the benchmark (and, with --conditions, the default, every image under every
condition of its plan, exactly the bytes every contestant gets), runs `predict` on each, maps the class names
to the benchmark's and posts the predictions as an external entry, which the server scores in the request; the
script prints the entry's score. Only the standard library is needed (plus whatever your model needs).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# The class names of your model, when `predict` returns class indices of the model rather than of the benchmark
# (for Ultralytics: list(model.names.values())). Leave it None when `predict` returns the benchmark's class
# indices or class names.
MODEL_CLASSES = None


def predict(image_bytes):
    """The one function to fill in: an image file's bytes (JPEG, PNG, ...) in, the detections out, as a list of
    (class, score, x1, y1, x2, y2): `class` a class name or index (see MODEL_CLASSES), `score` in 0..1, the box
    in pixels of the image. Decode the image upright (apply its EXIF orientation): the true boxes are in those
    coordinates. This default finds nothing, which is a valid (if poor) entry."""
    return []


# An Ultralytics model, for example:
#
# import io
# from PIL import Image, ImageOps
# from ultralytics import YOLO
#
# _model = YOLO("yolo11n.pt")
# MODEL_CLASSES = list(_model.names.values())
#
# def predict(image_bytes):
#     image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
#     r = _model.predict(image, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
#     return [
#         (int(c), float(s), *map(float, b))
#         for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
#     ]


class Api:
    """A tiny client of docs/API.md."""

    def __init__(self, server, token):
        self.base = server.rstrip("/") + "/api"
        self.token = token

    def request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req) as res:
                return res.read()
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())["error"]
                message = f"{err['code']}: {err['message']}" + (f" ({err['hint']})" if err.get("hint") else "")
            except (ValueError, KeyError):
                message = str(e)
            sys.exit(f"{req.method} {path}: {e.code} {message}")
        except urllib.error.URLError as e:
            sys.exit(f"cannot reach {self.base}: {e.reason}")

    def json(self, path, body=None):
        return json.loads(self.request(path, body))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--server", default=os.environ.get("FOLO_SERVER", "http://127.0.0.1:7878"))
    p.add_argument("--token", default=os.environ.get("FOLO_TOKEN"), help="the server's token, when it has one")
    p.add_argument("--bench", required=True, help="the benchmark's id (GET /api/bench/benchmarks)")
    p.add_argument("--name", required=True, help="the entry's name on the leaderboard")
    p.add_argument("--conditions", action=argparse.BooleanOptionalAction, default=True,
                   help="also run every condition of the benchmark's plan (default: yes)")
    p.add_argument("--unscored", action="store_true", help="the model gives no confidences (scores are ignored)")
    args = p.parse_args()

    api = Api(args.server, args.token)
    bench = api.json(f"/bench/benchmarks/{args.bench}")
    classes = {c.lower(): i for i, c in enumerate(bench["classes"])}
    images, offset = [], 0
    while True:
        page = api.json(f"/bench/benchmarks/{args.bench}/images?offset={offset}&limit=1000")
        images += [im["id"] for im in page["images"]]
        offset += len(page["images"])
        if offset >= page["total"] or not page["images"]:
            break
    variants = [None]
    if args.conditions:
        variants += [(c["condition"], s) for c in bench["plan"]["conditions"] for s in c["severities"]]

    unknown = {}

    def to_class(c):
        """The benchmark's class index of `c` (a name, or an index of MODEL_CLASSES or of the benchmark)."""
        if isinstance(c, str):
            name = c
        elif MODEL_CLASSES is not None:
            name = MODEL_CLASSES[int(c)]
        else:
            return int(c) if 0 <= int(c) < len(bench["classes"]) else None
        i = classes.get(name.strip().lower())
        if i is None:
            unknown[name] = unknown.get(name, 0) + 1
        return i

    predictions = []
    total = len(images) * len(variants)
    for v in variants:
        query = "" if v is None else f"?condition={v[0]}&severity={v[1]}"
        for image in images:
            data = api.request(f"/bench/benchmarks/{args.bench}/images/{image}{query}")
            boxes = []
            for c, score, x1, y1, x2, y2 in predict(data):
                k = to_class(c)
                if k is not None:
                    boxes.append({"class": k, "score": min(max(float(score), 0.0), 1.0),
                                  "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
            record = {"image": image, "boxes": boxes}
            if v is not None:
                record["condition"], record["severity"] = v
            predictions.append(record)
            print(f"\r{len(predictions)}/{total} images", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if unknown:
        print(f"dropped detections of classes the benchmark does not have: {unknown}", file=sys.stderr)

    body = {"name": args.name,
            "contestant": {"kind": "external", "scored": not args.unscored, "predictions": predictions}}
    entry = api.json(f"/bench/benchmarks/{args.bench}/entries", body)
    score = entry["score"]
    print(json.dumps({"entry": entry["id"], "score": score["total"], "partial": score["partial"],
                      "map50_95": entry["map50_95"],
                      "axes": {a["axis"]: a["value"] for a in score["axes"]}}, indent=2))


if __name__ == "__main__":
    main()
