"""Checks refiner.sample (gather-based bilinear on the projected maps) against grid_sample (border padding,
align_corners=False) on random maps and points, and the refiner's forward and identity start on random inputs."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refiner import STRIDES, Refiner, sample  # noqa: E402

torch.manual_seed(0)
B, d = 2, 8
sizes = [(80, 80), (40, 40), (20, 20)]
maps = [torch.randn(B, d, h, w, dtype=torch.float64) for h, w in sizes]
offsets, o = [], 0
for h, w in sizes:
    offsets.append(o)
    o += h * w
tab = torch.cat([m.flatten(2) for m in maps], dim=2).transpose(1, 2).reshape(-1, d)
N = 500
pts = torch.rand(N, 7, 2, dtype=torch.float64) * 760 - 60  # some outside the 640 image
img = torch.randint(0, B, (N,))
lvl = torch.randint(0, 3, (N, 7))
got = sample(tab, sizes, offsets, img, pts, lvl)
err = 0.0
for l in range(3):
    s = STRIDES[l]
    g = pts / (sizes[l][1] * s) * 2 - 1  # normalised, align_corners=False
    ref = F.grid_sample(maps[l][img], g[:, :, None, :], mode="bilinear", padding_mode="border",
                        align_corners=False)[..., 0].transpose(1, 2)
    m = lvl == l
    err = max(err, (got[m] - ref[m]).abs().max().item())
print(f"sample vs grid_sample: max abs diff {err:.3g}")
assert err < 1e-9

model = Refiner(channels=(64, 128, 256))
pyr = [torch.randn(B, c, h, w) for c, (h, w) in zip((64, 128, 256), sizes)]
x1 = torch.rand(N) * 500
y1 = torch.rand(N) * 500
box = torch.stack([x1, y1, x1 + 5 + torch.rand(N) * 130, y1 + 5 + torch.rand(N) * 130], 1)
b = {"img": img, "box": box, "sigma": torch.rand(N, 4) * 10 + 1, "sigma_fused": torch.rand(N, 4) * 10 + 1,
     "score": torch.rand(N), "level": torch.randint(0, 3, (N,)), "probs": torch.softmax(torch.randn(N, 4, 16), -1),
     "spread": torch.rand(N, 4), "members": torch.randint(1, 9, (N,)),
     "valid": torch.tensor([0.0, 80.0, 640.0, 560.0]).expand(N, 4)}
logits, s, q = model(model.project(pyr), b)
ref = model.refine(box, logits, s)
print("identity at init:", (ref - box).abs().max().item())
assert (ref - box).abs().max().item() == 0
print("ok")
