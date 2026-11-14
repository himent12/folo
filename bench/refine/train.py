"""Train the box refiner on a feature dump, in resumable chunks.

    python train.py --run <dir> [--dump <dir> --boxes <npz> ...] [--budget-min 18]

The first call writes <run>/config.json from the flags; later calls (same --run) read it and continue from
<run>/ckpt.pt. Each call trains for at most --budget-min minutes, checkpoints, and exits 0; <run>/done marks the
end. Several --dump/--boxes pairs are trained on together (an epoch visits every image of every dump once).
Loss: DFL two-bin cross-entropy on the outward residual in units of s_e (clamped to [-2, 2]) plus `--w-iou` times
(1 - CIoU) of the refined box against its truth, on the B* boxes whose same-class truth overlaps at IoU >=
--min-iou.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import FeatureDump, loader, to_device  # noqa: E402
from refiner import Geometry, Refiner, ciou, dfl_loss, fine_tables  # noqa: E402
from stem import Stem  # noqa: E402

WEIGHTS = "D:/folo-data/runs-hill3/fusion/land-n-v2/weights/best.folo"

NORMAL = torch.tensor([-1.0, -1.0, 1.0, 1.0])


def build(cfg):
    g = Geometry(**cfg["geometry"])
    return Refiner(channels=cfg["channels"], d=cfg["d"], hidden=cfg["hidden"], layers=cfg["layers"], geometry=g,
                   features=cfg["features"], shared=cfg["shared"], quality=cfg.get("quality", False),
                   fine=cfg.get("fine"), strip_px=cfg.get("strip_px"))


def fine_inputs(model, stem, inp, dev):
    """Experiment H's fine sources of a batch (None for a pyramid-only refiner)."""
    if model.fine is None:
        return None
    return fine_tables(stem(torch.as_tensor(inp).to(dev)))


def losses(model, proj, b, cfg, fine=None):
    logits, s, q = model(proj, b, fine)
    n = NORMAL.to(s.device)
    target = (b["gt"] - b["box"]) * n / s
    l_dfl = dfl_loss(logits, target).mean(-1)
    ref = model.refine(b["box"], logits, s)
    c, iou = ciou(ref, b["gt"])
    l_iou = 1 - c
    w = b["score"] ** cfg.get("score_power", 0.0)
    loss = (w * (l_dfl + cfg["w_iou"] * l_iou)).sum() / w.sum()
    if q is not None:
        loss = loss + cfg.get("w_q", 1.0) * torch.nn.functional.binary_cross_entropy_with_logits(
            q, iou.detach().clamp(0, 1))
    return loss, l_dfl.mean().item(), l_iou.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dump", action="append")
    ap.add_argument("--boxes", action="append")
    ap.add_argument("--budget-min", type=float, default=18)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--img-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--primary", action="store_true", help="train on each truth's top-scoring box only")
    ap.add_argument("--w-iou", type=float, default=1.0)
    ap.add_argument("--score-power", type=float, default=0.0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--fixed", type=float)
    ap.add_argument("--levels", default="closest")
    ap.add_argument("--sigma", default="kept")
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--four-heads", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--fine", choices=["h1", "h2"], help="experiment H: probes on the input and the stem")
    ap.add_argument("--strip-px", type=float, help="h2: strip spacing in input pixels instead of s_e / 4")
    ap.add_argument("--max-images", type=int, help="use only the first N images of each dump")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.run, exist_ok=True)
    cfg_path = os.path.join(a.run, "config.json")
    if os.path.exists(os.path.join(a.run, "done")):
        print("done already")
        return
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "dumps": a.dump, "boxes": a.boxes, "epochs": a.epochs, "img_batch": a.img_batch, "lr": a.lr,
            "wd": a.wd, "min_iou": "primary" if a.primary else a.min_iou, "w_iou": a.w_iou, "score_power": a.score_power, "d": a.d,
            "hidden": a.hidden, "layers": a.layers, "features": not a.no_features, "shared": not a.four_heads,
            "quality": a.quality, "fine": a.fine, "strip_px": a.strip_px, "max_images": a.max_images, "seed": a.seed,
            "geometry": {"kappa": a.kappa, "alpha": a.alpha, "fixed": a.fixed, "levels": a.levels, "sigma": a.sigma},
        }
        with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, indent=1)
    dev = torch.device("cuda")
    dumps = [FeatureDump(d, b, cfg.get("fine") is not None) for d, b in zip(cfg["dumps"], cfg["boxes"])]
    cfg["channels"] = list(dumps[0].channels)
    model = build(cfg).to(dev)
    stem = Stem(WEIGHTS).to(dev)
    params = [p for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    # Image lists per dump; an epoch walks every dump's images in one seeded order of (dump, image) pairs, in
    # batches of one dump.
    lists = [list(range(min(d.n, cfg["max_images"] or d.n))) for d in dumps]
    batches_per_epoch = sum((len(l) + cfg["img_batch"] - 1) // cfg["img_batch"] for l in lists)
    total = batches_per_epoch * cfg["epochs"]
    warm = min(300, total // 20)

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    step, ckpt = 0, os.path.join(a.run, "ckpt.pt")
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed at step {step} / {total}", flush=True)
    t0 = time.time()
    log = open(os.path.join(a.run, "log.jsonl"), "a", encoding="utf-8", newline="\n")
    model.train()
    stop = False
    while step < total and not stop:
        epoch = step // batches_per_epoch
        rng = np.random.default_rng(cfg["seed"] * 1000 + epoch)
        plan = []  # (dump, image ids) per batch of this epoch
        for k, l in enumerate(lists):
            perm = rng.permutation(l)
            plan += [(k, perm[i:i + cfg["img_batch"]]) for i in range(0, len(perm), cfg["img_batch"])]
        order = rng.permutation(len(plan))
        plan = [plan[i] for i in order]
        pos = step - epoch * batches_per_epoch
        # One loader per dump would interleave badly; fetch batches through a small thread of per-batch reads.
        it = iter_batches(dumps, plan[pos:], cfg["min_iou"], a.workers)
        acc = []
        for idx, pyr, rows, inp in it:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            pyr = [torch.as_tensor(p).to(dev, non_blocking=True) for p in pyr]
            b = to_device(rows, dev)
            if b["box"].shape[0] == 0:
                step += 1
                continue
            proj = model.project(pyr) if model.features else None
            loss, ld, li = losses(model, proj, b, cfg, fine_inputs(model, stem, inp, dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            acc.append((loss.item(), ld, li, b["box"].shape[0]))
            step += 1
            if step % 1000 == 0:
                save(model, opt, step, cfg, ckpt)
            if step % 100 == 0:
                m = np.array(acc)
                rec = {"step": step, "total": total, "epoch": step / batches_per_epoch, "loss": float(m[:, 0].mean()),
                       "dfl": float(m[:, 1].mean()), "ciou": float(m[:, 2].mean()), "boxes": float(m[:, 3].mean()),
                       "lr": lr_at(step), "min": (time.time() - t0) / 60}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                acc = []
            if (time.time() - t0) / 60 > a.budget_min or step >= total or step % batches_per_epoch == 0:
                stop = (time.time() - t0) / 60 > a.budget_min
                break
    save(model, opt, step, cfg, ckpt)
    if step >= total:
        torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(a.run, "refiner.pt"))
        open(os.path.join(a.run, "done"), "w").close()
        print(f"finished: {step} steps", flush=True)
    else:
        print(f"chunk end at step {step} / {total} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def save(model, opt, step, cfg, ckpt):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg}, ckpt + ".tmp")
    os.replace(ckpt + ".tmp", ckpt)


class _Plan(torch.utils.data.Dataset):
    def __init__(self, dumps, plan, min_iou):
        self.dumps, self.plan, self.min_iou = dumps, plan, min_iou

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, k):
        d, idx = self.plan[k]
        dm = self.dumps[d]
        return idx, dm.pyramid(idx), dm.rows(idx, self.min_iou), dm.inputs(idx)


def iter_batches(dumps, plan, min_iou, workers):
    return torch.utils.data.DataLoader(_Plan(dumps, plan, min_iou), batch_size=None, shuffle=False,
                                       num_workers=workers, prefetch_factor=4 if workers else None)


if __name__ == "__main__":
    main()
