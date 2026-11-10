#!/usr/bin/env python3
"""Run any vision model on a FOLO benchmark and submit the entry (docs/BENCH.md, "Who can run").

Fill in `predict` below (an Ultralytics example is in the comments), then:

    python examples/bench_external.py --bench bn_3f9a --name my-detector
    python examples/bench_external.py --server http://127.0.0.1:7878 --bench bn_3f9a --name fast --no-conditions

The script opens an external entry, then goes pass by pass: the unchanged images, then (with --conditions, the
default) every condition and severity of the benchmark's plan. For each pass it fetches every image (exactly the
bytes every contestant gets), runs `predict`, maps the class names to the benchmark's and sends the pass as one
part of the entry (split further when its JSON would be over 32 MiB), so the server holds about one pass at a time
whatever the benchmark's size. Then it finishes the entry, which the server scores, and prints the score. If
anything fails after the entry was opened, the entry is stopped, so no half-filled entry is left behind. Only the
standard library is needed (plus whatever your model needs).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# The class names of your model, when `predict` returns class indices of the model rather than of the benchmark
# (for Ultralytics: list(model.names.values())). Leave it None when `predict` returns the benchmark's class
# indices or class names.
MODEL_CLASSES = None

# A part's JSON is kept under this (the server takes up to 64 MiB per request).
MAX_PART_BYTES = 32 << 20
# A record carries at most this many boxes (the server's limit); the best-scored ones are kept.
MAX_BOXES = 1000


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


class ApiError(Exception):
    pass


class Api:
    """A tiny client of docs/API.md."""

    def __init__(self, server, token):
        self.base = server.rstrip("/") + "/api"
        self.token = token
        self.requests = 0
        self.largest = 0

    def request(self, path, body=None):
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.largest = max(self.largest, len(data))
        req = urllib.request.Request(self.base + path, data=data, method="GET" if data is None else "POST")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        self.requests += 1
        try:
            with urllib.request.urlopen(req) as res:
                return res.read()
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())["error"]
                message = f"{err['code']}: {err['message']}" + (f" ({err['hint']})" if err.get("hint") else "")
            except (ValueError, KeyError):
                message = str(e)
            raise ApiError(f"{req.method} {path}: {e.code} {message}") from None
        except urllib.error.URLError as e:
            raise ApiError(f"cannot reach {self.base}: {e.reason}") from None

    def json(self, path, body=None):
        return json.loads(self.request(path, body))


def number(v, digits):
    """`v` as a short JSON number: rounded to `digits` decimals, without a trailing `.0`."""
    v = round(float(v), digits)
    return int(v) if v == int(v) else v


def significant(v, digits=4):
    """`v` with `digits` significant digits."""
    return float(f"{float(v):.{digits}g}")


def encode(records):
    """The JSON of a part, compact."""
    return json.dumps({"predictions": records}, separators=(",", ":")).encode()


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
    try:
        bench = api.json(f"/bench/benchmarks/{args.bench}")
        images, offset = [], 0
        while True:
            page = api.json(f"/bench/benchmarks/{args.bench}/images?offset={offset}&limit=1000")
            images += [im["id"] for im in page["images"]]
            offset += len(page["images"])
            if offset >= page["total"] or not page["images"]:
                break
    except ApiError as e:
        sys.exit(str(e))
    classes = {c.lower(): i for i, c in enumerate(bench["classes"])}
    passes = [None]
    if args.conditions:
        passes += [(c["condition"], s) for c in bench["plan"]["conditions"] for s in c["severities"]]

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

    def record(image, data, v):
        boxes = []
        for c, score, x1, y1, x2, y2 in predict(data):
            k = to_class(c)
            if k is not None:
                boxes.append((min(max(float(score), 0.0), 1.0), k, x1, y1, x2, y2))
        boxes.sort(key=lambda b: -b[0])
        r = {"image": image, "boxes": [
            {"class": k, "score": significant(s), "x1": number(x1, 2), "y1": number(y1, 2),
             "x2": number(x2, 2), "y2": number(y2, 2)}
            for s, k, x1, y1, x2, y2 in boxes[:MAX_BOXES]]}
        if v is not None:
            r["condition"], r["severity"] = v
        return r

    contestant = {"kind": "external", "scored": not args.unscored, "open": True, "predictions": []}
    try:
        entry = api.json(f"/bench/benchmarks/{args.bench}/entries", {"name": args.name, "contestant": contestant})
    except ApiError as e:
        sys.exit(str(e))
    uri = f"/bench/benchmarks/{args.bench}/entries/{entry['id']}"
    t0, parts_sent = time.time(), 0
    try:
        for n, v in enumerate(passes, 1):
            query = "" if v is None else f"?condition={v[0]}&severity={v[1]}"
            part, size, sent = [], 0, 0
            for image in images:
                r = record(image, api.request(f"/bench/benchmarks/{args.bench}/images/{image}{query}"), v)
                grown = len(json.dumps(r, separators=(",", ":"))) + 1
                if part and size + grown > MAX_PART_BYTES:
                    api.request(f"{uri}/predictions", encode(part))
                    part, size, sent = [], 0, sent + 1
                part.append(r)
                size += grown
            api.request(f"{uri}/predictions", encode(part))
            parts_sent += sent + 1
            what = "unchanged" if v is None else f"{v[0]} {v[1]}"
            print(f"pass {n}/{len(passes)} ({what}): {len(images)} images in {sent + 1} "
                  f"part{'s' if sent else ''}, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        entry = api.json(f"{uri}/finish", {})
    except BaseException as e:
        # No half-filled entry is left behind.
        try:
            api.request(f"{uri}/stop", {})
        except ApiError:
            pass
        if isinstance(e, ApiError):
            sys.exit(f"{e} (the entry was stopped)")
        raise
    if unknown:
        print(f"dropped detections of classes the benchmark does not have: {unknown}", file=sys.stderr)
    score = entry["score"]
    print(json.dumps({"entry": entry["id"], "score": score["total"], "partial": score["partial"],
                      "map50_95": entry["map50_95"], "ci95": entry.get("ci95"),
                      "axes": {a["axis"]: a["value"] for a in score["axes"]},
                      "parts": parts_sent, "requests": api.requests, "largest_body_bytes": api.largest}, indent=2))


if __name__ == "__main__":
    main()
