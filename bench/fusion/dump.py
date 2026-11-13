"""Reader of the pre-NMS candidate dump written by `cargo run -p folo-vision --example candidate_dump`.

Layout (crates/folo-vision/examples/candidate_dump.rs): dump.json, images.jsonl, anchors.bin (per anchor:
anchor index u32, class-max score f32, 2 unused words, the network box x1 y1 x2 y2 f32, the original-image box f32,
then 4 * reg_max DFL logits f32), pairs.bin (per pair: the anchor's position in its image's list u32, class u32,
score f32).
"""

import json
import os

import numpy as np


class Image:
    """One image of a dump: its letterbox, its candidate anchors and its (anchor, class, score) pairs."""

    __slots__ = ("index", "file", "orig", "ratio", "pad", "anchor", "level", "stride", "point", "box_net",
                 "box_orig", "logits", "pair_anchor", "pair_class", "pair_score", "image_id")


class Dump:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "dump.json"), encoding="utf-8") as f:
            self.header = json.load(f)
        h = self.header
        self.reg_max = h["reg_max"]
        self.num_classes = h["num_classes"]
        self.words = h["anchor_words"]
        with open(os.path.join(path, "images.jsonl"), encoding="utf-8") as f:
            self.images = [json.loads(l) for l in f if l.strip()]
        self.anchors = np.memmap(os.path.join(path, "anchors.bin"), dtype="<u4", mode="r").reshape(-1, self.words)
        self.pairs = np.memmap(os.path.join(path, "pairs.bin"), dtype="<u4", mode="r").reshape(-1, 3)
        lv = h["levels"]
        self.level_start = np.array([l["start"] for l in lv], dtype=np.int64)
        self.level_width = np.array([l["width"] for l in lv], dtype=np.int64)
        self.level_stride = np.array([l["stride"] for l in lv], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def image(self, i):
        m = self.images[i]
        a0, na = m["anchor_start"], m["anchors"]
        p0, np_ = m["pair_start"], m["pairs"]
        rec = np.asarray(self.anchors[a0:a0 + na])
        pr = np.asarray(self.pairs[p0:p0 + np_])
        im = Image()
        im.index, im.file = m["index"], m["file"]
        im.orig = np.array(m["orig"], dtype=np.float32)
        im.ratio = np.array(m["ratio"], dtype=np.float32)
        im.pad = np.array(m["pad"], dtype=np.float32)
        im.anchor = rec[:, 0].astype(np.int64)
        im.level = np.searchsorted(self.level_start, im.anchor, side="right") - 1
        im.stride = self.level_stride[im.level]
        local = im.anchor - self.level_start[im.level]
        w = self.level_width[im.level]
        # Grid units (cell centres), as decode::make_anchors.
        im.point = np.stack([(local % w).astype(np.float32) + np.float32(0.5),
                             (local // w).astype(np.float32) + np.float32(0.5)], axis=1)
        im.box_net = rec[:, 4:8].view(np.float32)
        im.box_orig = rec[:, 8:12].view(np.float32)
        im.logits = rec[:, 12:].view(np.float32).reshape(-1, 4, self.reg_max)
        im.pair_anchor = pr[:, 0].astype(np.int64)
        im.pair_class = pr[:, 1].astype(np.int64)
        im.pair_score = pr[:, 2].view(np.float32)
        return im


def unmap(box, ratio, pad, orig):
    """ImageMeta::unmap_xyxy in f32: network pixels to original pixels, clipped."""
    box = box.astype(np.float32)
    out = np.empty_like(box)
    out[:, 0] = np.clip((box[:, 0] - pad[0]) / ratio[0], np.float32(0), orig[0])
    out[:, 1] = np.clip((box[:, 1] - pad[1]) / ratio[1], np.float32(0), orig[1])
    out[:, 2] = np.clip((box[:, 2] - pad[0]) / ratio[0], np.float32(0), orig[0])
    out[:, 3] = np.clip((box[:, 3] - pad[1]) / ratio[1], np.float32(0), orig[1])
    return out
