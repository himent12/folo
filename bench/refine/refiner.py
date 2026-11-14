"""The box refiner of lane refine (docs/notes/hill/r3-refine.md), in PyTorch: probe geometry, point sampling of the
pyramid, the side network, its loss.

Sides are handled in one canonical frame so one network serves all four: side e in (left, top, right, bottom) has
the outward normal n_e = (-x, -y, +x, +y), probes sit at `x_e + t * s_e * n_e` for t in {-2, -1, 0, 1, 2} (t > 0 is
outside the box) and at `u` in {1/8, 3/8, 5/8, 7/8} along the edge (left to right, top to bottom), and the network
predicts the outward residual `r = (x_gt - x_e) * n_e / s_e` as a softmax expectation over 17 bins on [-2, 2].
Refined edge: `x_e + n_e * s_e * E[r]`.

Sampling is bilinear on the feature grid (the value at pixel (i, j) of a stride-s level sits at ((j + 0.5) s,
(i + 0.5) s)), clamped to the border, and done as index selects plus a weighted sum on the projected maps (a linear
projection commutes with bilinear interpolation, so projecting the maps first is the same as projecting samples).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = (8, 16, 32)
NBINS = 17
T_PROBE = (-2.0, -1.0, 0.0, 1.0, 2.0)
U_PROBE = (0.125, 0.375, 0.625, 0.875)
NORMAL = ((-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0))


class Geometry:
    """Probe placement. `kappa`, `alpha`: s_e = max(kappa * sigma_e, alpha * size across the edge); `fixed`: s_e =
    fixed * size instead (the fixed-grid control). `levels`: 'closest' (the level whose stride is closest to s_e),
    'all' (all three, concatenated), 'anchor' (the kept candidate's level). `sigma`: 'kept' or 'fused'."""

    def __init__(self, kappa=1.5, alpha=0.04, fixed=None, levels="closest", sigma="kept", t=T_PROBE, u=U_PROBE):
        self.kappa, self.alpha, self.fixed, self.levels, self.sigma = kappa, alpha, fixed, levels, sigma
        self.t, self.u = tuple(t), tuple(u)

    def as_dict(self):
        return {"kappa": self.kappa, "alpha": self.alpha, "fixed": self.fixed, "levels": self.levels,
                "sigma": self.sigma, "t": list(self.t), "u": list(self.u)}

    def scale(self, box, sigma):
        """s_e per side, [N, 4] pixels."""
        w = (box[:, 2] - box[:, 0]).clamp(min=1.0)
        h = (box[:, 3] - box[:, 1]).clamp(min=1.0)
        size = torch.stack([w, h, w, h], dim=1)
        if self.fixed is not None:
            s = self.fixed * size
        else:
            s = torch.maximum(self.kappa * sigma, self.alpha * size)
        return s.clamp(min=0.5)

    def points(self, box, s):
        """Probe points [N, 4, T, U, 2] (network pixels) and the 3 x 3 context grid [N, 9, 2]."""
        dev = box.device
        t = torch.tensor(self.t, device=dev)
        u = torch.tensor(self.u, device=dev)
        x1, y1, x2, y2 = box.unbind(1)
        w, h = x2 - x1, y2 - y1
        pts = []
        for e in range(4):
            nx, ny = NORMAL[e]
            edge = box[:, e]
            off = edge[:, None] + t[None, :] * s[:, e:e + 1] * (nx + ny)  # [N, T] the coordinate across the edge
            if e % 2 == 0:  # left / right: across is x, along is y
                along = y1[:, None] + u[None, :] * h[:, None]  # [N, U]
                px = off[:, :, None].expand(-1, -1, len(self.u))
                py = along[:, None, :].expand(-1, len(self.t), -1)
            else:
                along = x1[:, None] + u[None, :] * w[:, None]
                py = off[:, :, None].expand(-1, -1, len(self.u))
                px = along[:, None, :].expand(-1, len(self.t), -1)
            pts.append(torch.stack([px, py], dim=-1))
        probes = torch.stack(pts, dim=1)
        g = (torch.arange(3, device=dev, dtype=box.dtype) + 0.5) / 3
        cx = x1[:, None] + g[None, :] * w[:, None]
        cy = y1[:, None] + g[None, :] * h[:, None]
        ctx = torch.stack([cx[:, None, :].expand(-1, 3, -1), cy[:, :, None].expand(-1, -1, 3)], dim=-1)
        return probes, ctx.reshape(-1, 9, 2)

    def side_level(self, s, level):
        """The pyramid level each side samples, [N, 4] (for 'closest' and 'anchor')."""
        if self.levels == "anchor":
            return level[:, None].expand(-1, 4)
        return torch.round(torch.log2(s / STRIDES[0])).clamp(0, 2).long()

    @staticmethod
    def context_level(box):
        w = (box[:, 2] - box[:, 0]).clamp(min=1.0)
        h = (box[:, 3] - box[:, 1]).clamp(min=1.0)
        return torch.round(torch.log2(torch.minimum(w, h) / 3 / STRIDES[0])).clamp(0, 2).long()


def sample(table, sizes, offsets, img, pts, lvl):
    """Bilinear samples of the projected maps.

    table: [B * A, d] the projected maps of a batch (image-major, level-major inside an image, row-major in a level);
    sizes: [(H, W)] per level; offsets: per level start row inside an image; img: [N] batch position per box;
    pts: [N, ..., 2] points; lvl: [N, ...] levels (broadcast to pts[..., 0]). Returns [N, ..., d].
    """
    A = sum(h * w for h, w in sizes)
    dev = pts.device
    st = torch.tensor(STRIDES, device=dev, dtype=pts.dtype)[lvl]
    H = torch.tensor([h for h, _ in sizes], device=dev)[lvl]
    W = torch.tensor([w for _, w in sizes], device=dev)[lvl]
    off = torch.tensor(offsets, device=dev)[lvl]
    u = (pts[..., 0] / st - 0.5).clamp(min=0).minimum((W - 1).to(pts.dtype))
    v = (pts[..., 1] / st - 0.5).clamp(min=0).minimum((H - 1).to(pts.dtype))
    x0 = torch.floor(u).long().minimum(W - 2).clamp(min=0)
    y0 = torch.floor(v).long().minimum(H - 2).clamp(min=0)
    fx = (u - x0).unsqueeze(-1)
    fy = (v - y0).unsqueeze(-1)
    base = img.view(-1, *([1] * (pts.dim() - 2))) * A + off
    i00 = base + y0 * W + x0
    g = lambda i: table[i]  # noqa: E731
    return ((1 - fy) * ((1 - fx) * g(i00) + fx * g(i00 + 1)) + fy * ((1 - fx) * g(i00 + W) + fx * g(i00 + W + 1)))


def sample_map(tab, H, W, stride, img, pts):
    """Bilinear samples (border clamped) of one dense map flattened to tab [B * H * W, C] (pixel (i, j) at
    ((j + 0.5) stride, (i + 0.5) stride)); img [N], pts [N, ..., 2] -> [N, ..., C]."""
    u = (pts[..., 0] / stride - 0.5).clamp(0, W - 1)
    v = (pts[..., 1] / stride - 0.5).clamp(0, H - 1)
    x0 = torch.floor(u).long().clamp(max=W - 2)
    y0 = torch.floor(v).long().clamp(max=H - 2)
    fx = (u - x0).unsqueeze(-1)
    fy = (v - y0).unsqueeze(-1)
    i00 = img.view(-1, *([1] * (pts.dim() - 2))) * (H * W) + y0 * W + x0
    return ((1 - fy) * ((1 - fx) * tab[i00] + fx * tab[i00 + 1])
            + fy * ((1 - fx) * tab[i00 + W] + fx * tab[i00 + W + 1]))


FINE_STRIDES = (1, 2, 4)  # experiment H: the input (x / 255), layer 0, layer 1
STRIP_ACROSS, STRIP_ALONG = 32, 8


def fine_tables(maps):
    """[(table [B * H * W, C], H, W, stride)] of experiment H's fine sources."""
    return [(m.permute(0, 2, 3, 1).reshape(-1, m.shape[1]).float(), m.shape[2], m.shape[3], st)
            for m, st in zip(maps, FINE_STRIDES)]


def strip_points(box, s, px=None):
    """Experiment H2's per-side strips [N, 4, ALONG, ACROSS, 2]: across the edge at spacing s / 4, or `px` input
    pixels when given (outward increasing, centred on the edge), along it at (j + 0.5) / ALONG of the edge (left to
    right, top to bottom)."""
    dev = box.device
    k = torch.arange(STRIP_ACROSS, device=dev, dtype=box.dtype) - (STRIP_ACROSS - 1) / 2
    step = s / 4 if px is None else torch.full_like(s, px)
    u = (torch.arange(STRIP_ALONG, device=dev, dtype=box.dtype) + 0.5) / STRIP_ALONG
    x1, y1, x2, y2 = box.unbind(1)
    w, h = x2 - x1, y2 - y1
    out = []
    for e in range(4):
        sign = NORMAL[e][0] + NORMAL[e][1]
        across = box[:, e, None] + sign * k[None, :] * step[:, e, None]  # [N, ACROSS]
        if e % 2 == 0:
            along = y1[:, None] + u[None, :] * h[:, None]
            px = across[:, None, :].expand(-1, STRIP_ALONG, -1)
            py = along[:, :, None].expand(-1, -1, STRIP_ACROSS)
        else:
            along = x1[:, None] + u[None, :] * w[:, None]
            py = across[:, None, :].expand(-1, STRIP_ALONG, -1)
            px = along[:, :, None].expand(-1, -1, STRIP_ACROSS)
        out.append(torch.stack([px, py], dim=-1))
    return torch.stack(out, dim=1)


class StripNet(nn.Module):
    """Experiment H2's eyes: [M, 3, ALONG, ACROSS] RGB strips -> [M, out]."""

    def __init__(self, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=(1, 2), padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=(2, 2), padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
        )
        self.fc = nn.Linear(32 * (STRIP_ACROSS // 4), out)

    def forward(self, x):
        y = self.net(x - 0.5).mean(dim=2)  # pool along the edge, keep the across profile
        return F.silu(self.fc(y.flatten(1)))


SCALARS = 6 + 3 + 3 + 16 + 4  # see side_scalars


def side_scalars(box, s, sigma, sigma_f, score, level, side_lvl, probs, spread, members, valid):
    """Per side, [N, 4, SCALARS]: log(sigma / size), log(sigma_fused / sigma), log(s / size), log w, log h,
    score, the kept anchor's level (one hot), the side's probe level (one hot), the side's 16 DFL probabilities,
    spread / s, log(1 + members), the outward distance from the edge to the image's border / s (clamped to
    [-4, 4]), and whether that distance is under one pixel."""
    N = box.shape[0]
    w = (box[:, 2] - box[:, 0]).clamp(min=1.0)
    h = (box[:, 3] - box[:, 1]).clamp(min=1.0)
    size = torch.stack([w, h, w, h], dim=1)
    n = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=box.device)
    border = (valid - box) * n  # outward distance to the image's border
    f = [
        torch.log(sigma / size), torch.log(sigma_f / sigma), torch.log(s / size),
        torch.log(w / 640)[:, None].expand(-1, 4), torch.log(h / 640)[:, None].expand(-1, 4),
        score[:, None].expand(-1, 4),
    ]
    lv = F.one_hot(level.long(), 3).float()[:, None, :].expand(-1, 4, -1)
    sl = F.one_hot(side_lvl, 3).float()
    tail = [
        (spread / s).clamp(max=4.0), torch.log1p(members.float())[:, None].expand(-1, 4),
        (border / s).clamp(-4, 4), (border.abs() < 1).float(),
    ]
    out = torch.cat([torch.stack(f, dim=-1), lv, sl, probs.float(), torch.stack(tail, dim=-1)], dim=-1)
    return out.reshape(N, 4, SCALARS)


class Refiner(nn.Module):
    def __init__(self, channels=(64, 128, 256), d=64, hidden=256, layers=2, geometry=None, features=True,
                 shared=True, quality=False, fine=None, strip_px=None):
        super().__init__()
        self.geom = geometry or Geometry()
        self.d, self.features, self.shared, self.quality = d, features, shared, quality
        self.fine = fine  # None (probes on P3-P5), "h1" (probes on the input, layer 0, layer 1), "h2" (strip CNN)
        self.strip_px = strip_px  # h2: strip spacing in input pixels (None: s_e / 4)
        self.channels = tuple(channels)
        self.proj = nn.ModuleList([nn.Conv2d(c, d, 1) for c in channels])
        nl = 3 if self.geom.levels == "all" else 1
        np_ = len(self.geom.t) * len(self.geom.u)
        self.ctx = nn.Linear(9 * d, d)
        self.side_emb = nn.Embedding(4, 16)
        if fine == "h1":
            self.fine_proj = nn.Linear(3 + 16 + 32, d)
            nl = 1
        elif fine == "h2":
            self.strip = StripNet(128)
        edge = 128 if fine == "h2" else np_ * nl * d
        fin = (edge + d if features else 0) + SCALARS + 16
        heads = 1 if shared else 4

        def mlp():
            mods, k = [], fin
            for _ in range(layers):
                mods += [nn.Linear(k, hidden), nn.SiLU()]
                k = hidden
            out = nn.Linear(k, NBINS + (1 if quality else 0))
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            return nn.Sequential(*mods, out)

        self.mlps = nn.ModuleList([mlp() for _ in range(heads)])
        self.register_buffer("bins", torch.linspace(-2, 2, NBINS))

    def config(self):
        return {"channels": list(self.channels), "d": self.d, "hidden": self.mlps[0][0].out_features,
                "layers": sum(isinstance(m, nn.Linear) for m in self.mlps[0]) - 1, "geometry": self.geom.as_dict(),
                "features": self.features, "shared": self.shared, "quality": self.quality, "fine": self.fine,
                "strip_px": self.strip_px}

    def project(self, pyramid):
        """pyramid: list of [B, C_l, H_l, W_l] -> table [B * A, d], sizes, offsets."""
        maps = [p(f.float()) for p, f in zip(self.proj, pyramid)]
        sizes = [(m.shape[2], m.shape[3]) for m in maps]
        offsets, o = [], 0
        for h, w in sizes:
            offsets.append(o)
            o += h * w
        tab = torch.cat([m.flatten(2) for m in maps], dim=2).transpose(1, 2).reshape(-1, self.d)
        return tab, sizes, offsets

    def forward(self, proj, b, fine=None):
        """proj: (table, sizes, offsets) of the batch's pyramid; b: dict of per-box tensors (img, box, sigma,
        sigma_fused, score, level, probs, spread, members, valid). Returns (logits [N, 4, NBINS], s [N, 4],
        quality logit or None). fine: fine_tables(...) of the batch for experiment H."""
        box = b["box"]
        sig = b["sigma"] if self.geom.sigma == "kept" else b["sigma_fused"]
        s = self.geom.scale(box, sig)
        side_lvl = self.geom.side_level(s, b["level"])
        N = box.shape[0]
        parts = []
        if self.features:
            tab, sizes, offsets = proj
            probes, ctx = self.geom.points(box, s)
            if self.fine == "h1":
                pf = torch.cat([sample_map(t, H, W, st, b["img"], probes) for t, H, W, st in fine], dim=-1)
                pf = self.fine_proj(pf)
            elif self.fine == "h2":
                t, H, W, st = fine[0]
                st_ = sample_map(t, H, W, st, b["img"], strip_points(box, s, self.strip_px))  # [N, 4, ALONG, ACROSS, 3]
                pf = self.strip(st_.reshape(N * 4, STRIP_ALONG, STRIP_ACROSS, 3).permute(0, 3, 1, 2))
            elif self.geom.levels == "all":
                feats = [sample(tab, sizes, offsets, b["img"], probes, torch.full_like(side_lvl, l)[:, :, None, None]
                                .expand(-1, -1, probes.shape[2], probes.shape[3])) for l in range(3)]
                pf = torch.cat(feats, dim=-1)
            else:
                lv = side_lvl[:, :, None, None].expand(-1, -1, probes.shape[2], probes.shape[3])
                pf = sample(tab, sizes, offsets, b["img"], probes, lv)
            cl = self.geom.context_level(box)[:, None].expand(-1, 9)
            cf = sample(tab, sizes, offsets, b["img"], ctx, cl)
            c = F.silu(self.ctx(cf.reshape(N, -1)))
            parts += [pf.reshape(N, 4, -1), c[:, None, :].expand(-1, 4, -1)]
        sc = side_scalars(box, s, sig, b["sigma_fused"], b["score"], b["level"], side_lvl, b["probs"], b["spread"],
                          b["members"], b["valid"])
        emb = self.side_emb.weight[None, :, :].expand(N, -1, -1)
        x = torch.cat(parts + [sc, emb], dim=-1)
        if self.shared:
            out = self.mlps[0](x)
        else:
            out = torch.stack([self.mlps[e](x[:, e]) for e in range(4)], dim=1)
        q = out[..., NBINS:].mean(dim=1).squeeze(-1) if self.quality else None
        return out[..., :NBINS], s, q

    def refine(self, box, logits, s):
        r = (logits.softmax(-1) * self.bins).sum(-1)
        n = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=box.device)
        return box + n * s * r


def dfl_loss(logits, target):
    """Two-bin cross-entropy on a target in [-2, 2] (DFL's, on the 17-bin lattice of step 0.25)."""
    pos = (target.clamp(-2, 2 - 1e-4) + 2) / 0.25
    left = pos.floor().long()
    wr = pos - left
    lp = logits.log_softmax(-1)
    return -((1 - wr) * lp.gather(-1, left[..., None])[..., 0] + wr * lp.gather(-1, (left + 1)[..., None])[..., 0])


def ciou(a, b, eps=1e-7):
    """CIoU of xyxy boxes (MODEL.md §7.2)."""
    w1, h1 = (a[:, 2] - a[:, 0]).clamp(min=eps), (a[:, 3] - a[:, 1]).clamp(min=eps)
    w2, h2 = (b[:, 2] - b[:, 0]).clamp(min=eps), (b[:, 3] - b[:, 1]).clamp(min=eps)
    iw = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    ih = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.maximum(a[:, 2], b[:, 2]) - torch.minimum(a[:, 0], b[:, 0])
    ch = torch.maximum(a[:, 3], b[:, 3]) - torch.minimum(a[:, 1], b[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((a[:, 0] + a[:, 2] - b[:, 0] - b[:, 2]) ** 2 + (a[:, 1] + a[:, 3] - b[:, 1] - b[:, 3]) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha), iou


def box_iou(a, b):
    iw = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    ih = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = iw * ih
    ua = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    ub = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (ua + ub - inter).clamp(min=1e-9)
