"""Reading a feature dump (candidate_dump --features) and its box table (boxes.py) in image batches."""

import json
import os

import numpy as np
import torch

LEVELS = ((80, 80), (40, 40), (20, 20))  # at 640: P3, P4, P5; channels c, 2c, 4c

FIELDS = ("box", "sigma", "sigma_fused", "score", "level", "probs", "spread", "members", "gt", "gt_iou", "cls")


def table(boxes_npz):
    """The box table as memory-mapped arrays: the .npz is unpacked once into a folder of .npy files beside it."""
    folder = boxes_npz[:-4] + ".npy.d"
    if not os.path.exists(os.path.join(folder, "complete")):
        os.makedirs(folder, exist_ok=True)
        z = np.load(boxes_npz)
        for k in z.files:
            np.save(os.path.join(folder, k + ".npy"), z[k])
        open(os.path.join(folder, "complete"), "w").close()
    return {f[:-4]: np.load(os.path.join(folder, f), mmap_mode="r") for f in os.listdir(folder) if f.endswith(".npy")}


class FeatureDump:
    """Pickles as its paths (DataLoader workers on Windows are spawned) and reopens the memory maps."""

    def __getstate__(self):
        return {"args": self.args}

    def __setstate__(self, st):
        self.__init__(*st["args"])

    def __init__(self, dump_dir, boxes_npz, inputs=False):
        self.args = (dump_dir, boxes_npz, inputs)
        # Experiment H: the network inputs of the same views, in the sibling dump `<dump_dir>-in` (--input).
        self.inp = None
        if inputs:
            self.inp = np.memmap(os.path.join(dump_dir + "-in", "input.bin"), dtype=np.uint8, mode="r")
            self.inp = self.inp.reshape(-1, 3, 640, 640)
        with open(os.path.join(dump_dir, "dump.json"), encoding="utf-8") as f:
            self.header = json.load(f)
        self.n = self.header["images"]
        raw = np.memmap(os.path.join(dump_dir, "feats.bin"), dtype="<f2", mode="r")
        self.per = raw.size // self.n
        assert self.per * self.n == raw.size
        c = self.per // 11200
        assert c * 11200 == self.per, self.per
        self.channels = (c, 2 * c, 4 * c)
        self.feats = raw.reshape(self.n, self.per)
        self.t = table(boxes_npz)
        assert len(self.t["img_count"]) == self.n
        with open(os.path.join(dump_dir, "images.jsonl"), encoding="utf-8") as f:
            self.images = [json.loads(l) for l in f if l.strip()]

    def inputs(self, idx):
        """Network inputs of images `idx`, u8 [B, 3, 640, 640] (empty without inputs)."""
        if self.inp is None:
            return np.zeros((0,), dtype=np.uint8)
        return np.stack([np.asarray(self.inp[i]) for i in idx])

    def pyramid(self, idx):
        """Features of images `idx` as three f16 arrays [B, C, H, W]."""
        x = np.stack([np.asarray(self.feats[i]) for i in idx])
        out, o = [], 0
        for c, (h, w) in zip(self.channels, LEVELS):
            out.append(x[:, o:o + c * h * w].reshape(len(idx), c, h, w))
            o += c * h * w
        return out

    def rows(self, idx, min_iou=None):
        """Box rows of images `idx` (optionally only those with a truth at IoU >= min_iou), with `img` = position
        in `idx` and `valid` = the image's area per box."""
        t = self.t
        sel, img = [], []
        for k, i in enumerate(idx):
            s0, n = t["img_start"][i], t["img_count"][i]
            r = np.arange(s0, s0 + n)
            if min_iou == "primary":  # each truth's top-scoring box (primary.py)
                r = r[t["primary"][r]]
            elif min_iou is not None:
                r = r[t["gt_iou"][r] >= min_iou]
            sel.append(r)
            img.append(np.full(len(r), k, dtype=np.int64))
        sel = np.concatenate(sel) if sel else np.zeros(0, dtype=np.int64)
        out = {f: np.asarray(t[f][sel]) for f in FIELDS}
        out["img"] = np.concatenate(img)
        out["valid"] = t["valid"][np.asarray(idx)[out["img"]]]
        out["row"] = sel
        return out


def to_device(rows, dev):
    out = {}
    for k, v in rows.items():
        dt = torch.int64 if k in ("img", "level", "members", "cls", "row") else torch.float32
        out[k] = torch.as_tensor(v).to(dev).to(dt)
    return out


class Batches(torch.utils.data.Dataset):
    """Item k: (pyramid, rows) of the k-th image batch of a fixed order."""

    def __init__(self, dump, order, batch, min_iou=None):
        self.dump, self.order, self.batch, self.min_iou = dump, order, batch, min_iou

    def __len__(self):
        return (len(self.order) + self.batch - 1) // self.batch

    def __getitem__(self, k):
        idx = self.order[k * self.batch:(k + 1) * self.batch]
        return idx, self.dump.pyramid(idx), self.dump.rows(idx, self.min_iou), self.dump.inputs(idx)


def loader(dump, order, batch, min_iou=None, workers=6, start=0):
    ds = Batches(dump, order, batch, min_iou)
    sub = torch.utils.data.Subset(ds, range(start, len(ds)))
    return torch.utils.data.DataLoader(sub, batch_size=None, shuffle=False, num_workers=workers,
                                       persistent_workers=False, prefetch_factor=4 if workers else None)
