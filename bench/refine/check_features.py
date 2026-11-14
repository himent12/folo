"""Checks a feature dump against the detector itself: the head's box branch (MODEL.md §6.1, fused weights) run in
PyTorch on the dumped P3 / P4 / P5 features must give the dumped DFL logits of the image's candidate anchors (up
to the f16 storage of the features: measured on landed folo-n with its EMA weights, max |logit diff| 0.006,
decoded distances 0.0002 bins on average and 0.004 at worst). A wrong level order, layout, image offset or a
one-cell shift fails it (whole bins on most anchors).

    python check_features.py <weights .folo> <dump dir> [--images 0 1 2]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fusion"))

from dump import Dump  # noqa: E402

DT = {"F32": np.float32, "F16": np.float16}


def read_folo(path, prefix):
    """The tensors under `prefix`, from the `ema.` copy when the file has one (what FOLO infers with)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        src = "ema." + prefix if any(k.startswith("ema." + prefix) for k in head) else prefix
        out = {}
        for k, v in head.items():
            if k == "__metadata__" or not k.startswith(src):
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            t = np.frombuffer(f.read(b - a), dtype=DT[v["dtype"]]).reshape(v["shape"]).copy()
            out[prefix + k[len(src):]] = torch.from_numpy(t)
    return out


def conv(w, p):
    """(weight, bias) of a Conv, fused (`conv.bias`) or with its BatchNorm folded in (MODEL.md §9.1, eps 1e-3)."""
    k = w[p + ".conv.weight"].float()
    if p + ".conv.bias" in w:
        return k, w[p + ".conv.bias"].float()
    g, b = w[p + ".bn.weight"].float(), w[p + ".bn.bias"].float()
    m, v = w[p + ".bn.running_mean"].float(), w[p + ".bn.running_var"].float()
    t = g / torch.sqrt(v + 1e-3)
    return k * t[:, None, None, None], b - m * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("dump")
    ap.add_argument("--images", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    w = read_folo(a.weights, "head.box.")
    d = Dump(a.dump)
    raw = np.memmap(os.path.join(a.dump, "feats.bin"), dtype="<f2", mode="r")
    per = raw.size // len(d)
    c = per // 11200
    sizes = [(c, 80, 80), (2 * c, 40, 40), (4 * c, 20, 20)]
    worst = 0.0
    for i in a.images:
        x = torch.from_numpy(np.asarray(raw[i * per:(i + 1) * per]).astype(np.float32))
        im = d.image(i)
        o = 0
        for l, (ch, h, wd) in enumerate(sizes):
            f = x[o:o + ch * h * wd].reshape(1, ch, h, wd)
            o += ch * h * wd
            p = f"head.box.p{l + 3}."
            y = F.silu(F.conv2d(f, *conv(w, p + "conv0"), padding=1))
            y = F.silu(F.conv2d(y, *conv(w, p + "conv1"), padding=1))
            y = F.conv2d(y, w[p + "out.weight"].float(), w[p + "out.bias"].float())[0]  # [64, h, w]
            sel = im.level == l
            if not sel.any():
                continue
            local = im.anchor[sel] - d.level_start[l]
            got = y.reshape(y.shape[0], -1)[:, torch.from_numpy(local)].T.reshape(-1, 4, d.reg_max).numpy()
            want = im.logits[sel]
            err = np.abs(got - want).max()
            rel = err / np.abs(want).max()
            k = np.arange(d.reg_max)
            dist = lambda z: (np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(  # noqa: E731
                -1, keepdims=True) * k).sum(-1)
            dd = np.abs(dist(got) - dist(want))
            worst = max(worst, dd.mean())
            print(f"image {i} P{l + 3}: {sel.sum()} anchors, max |diff| {err:.4f} (relative {rel:.2e}), mean |diff| "
                  f"{np.abs(got - want).mean():.4f}; decoded distance diff in bins: mean {dd.mean():.4f} max {dd.max():.4f}")
    print(f"worst mean decoded difference {worst:.4f} bins:", "ok" if worst < 0.1 else "MISMATCH")


if __name__ == "__main__":
    main()
