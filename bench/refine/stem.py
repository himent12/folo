"""The detector's stem (MODEL.md §4 rows 0 and 1: Conv 3->16 k3 s2, Conv 16->32 k3 s2, BN folded, SiLU) in
PyTorch, for experiment H's high-resolution probes, and its check against FOLO's own trace.

    python stem.py <weights .folo> <dump dir written with --input --stem>
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_features import conv, read_folo  # noqa: E402


class Stem(torch.nn.Module):
    """u8 [B, 3, H, W] -> (x / 255 [B, 3, H, W], layer 0 [B, 16, H/2, W/2], layer 1 [B, 32, H/4, W/4]), frozen."""

    def __init__(self, weights):
        super().__init__()
        w = read_folo(weights, "backbone.")
        for name, key in (("c0", "backbone.stem"), ("c1", "backbone.down2")):
            k, b = conv(w, key)
            self.register_buffer(name + "_w", k)
            self.register_buffer(name + "_b", b)

    @torch.no_grad()
    def forward(self, u8):
        x = u8.float() / 255.0
        y0 = F.silu(F.conv2d(x, self.c0_w, self.c0_b, stride=2, padding=1))
        y1 = F.silu(F.conv2d(y0, self.c1_w, self.c1_b, stride=2, padding=1))
        return x, y0, y1


def main():
    weights, dump = sys.argv[1], sys.argv[2]
    stem = Stem(weights)
    n = len(open(os.path.join(dump, "images.jsonl"), encoding="utf-8").read().split("\n")) - 1
    inp = np.fromfile(os.path.join(dump, "input.bin"), dtype=np.uint8).reshape(n, 3, 640, 640)
    ref = np.fromfile(os.path.join(dump, "stem.bin"), dtype="<f4").reshape(n, -1)
    worst = 0.0
    for i in range(n):
        _, y0, y1 = stem(torch.from_numpy(inp[i:i + 1].copy()))
        got = torch.cat([y0.flatten(), y1.flatten()]).numpy()
        d = np.abs(got - ref[i])
        worst = max(worst, d.max())
        print(f"image {i}: layer 0+1 max |diff| {d.max():.2e}, mean {d.mean():.2e}, max |value| {np.abs(ref[i]).max():.2f}")
    print(f"worst {worst:.2e}:", "ok" if worst < 1e-3 else "MISMATCH")


if __name__ == "__main__":
    main()
