"""Measure the incumbent (Ultralytics YOLO26, PyTorch eager) so FOLO has numbers to beat.

Usage (from the repo root, with any Python env that has torch + ultralytics):

    python bench/baselines/ultralytics_bench.py --weights-dir <dir with yolo26{n,s,m}.pt> \
        --work-dir D:/folo-bench --out bench/baselines/results-rtx4080s.json

Sections (``--sections``, comma separated, default all):
    params      parameters and GFLOPs as Ultralytics reports them (unfused and fused)
    gpu_fwd     model forward only, random input already on the GPU; fp32 / fp32-strict / fp16, batch 1 and 16,
                cudnn.benchmark off and on
    gpu_e2e     Ultralytics predictor end to end: uint8 HWC numpy in -> boxes on the host
    cpu         forward and end to end on the CPU, fp32, batch 1
    train_step  one training step (forward + loss + backward + optimizer) on a fixed GPU batch, no data loader
    train       Ultralytics trainer on a synthetic dataset, 2 epochs, AMP on and off (each run in a subprocess)

The script never writes outside ``--work-dir`` and ``--out``: Ultralytics settings are overridden in memory only
(downloads, runs and telemetry are redirected/disabled), and auto-install of missing packages is turned off.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# percentile and summarize are not used here: test_ultralytics_bench.py reaches them as ub.percentile and ub.summarize.
from bench_timing import (
    cpu_cores_busy,
    cpu_snapshot,
    percentile,
    summarize,
    summarize_training,
    time_cuda,
    time_host,
    wait_for_quiet_cpu,
)
from bench_workloads import IMGSZ, MODELS, TRAIN_MODELS, load_detection_model, make_dataset, predict_e2e, sample_images

os.environ["YOLO_AUTOINSTALL"] = "false"  # never pip-install into the user's env, even if the caller enabled it


# --------------------------------------------------------------------------------------------------------------------
# environment


def isolate_ultralytics(work_dir: Path) -> None:
    """Keep Ultralytics inside work_dir: its config dir (settings.json, fonts) goes there, set before the first import,
    so the user's own settings file is never read or written; downloads and runs go there too, and telemetry is off.
    Must run before anything imports ultralytics (child and data loader processes inherit the environment)."""
    assert "ultralytics" not in sys.modules, "isolate_ultralytics() must run before ultralytics is imported"
    os.environ["YOLO_CONFIG_DIR"] = str(work_dir / "ultralytics-config")
    import ultralytics.utils as uu
    from ultralytics.utils import SETTINGS

    overrides = {
        "weights_dir": str(work_dir / "weights"),
        "runs_dir": str(work_dir / "runs"),
        "datasets_dir": str(work_dir / "datasets"),
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": False,
    }
    for k, v in overrides.items():
        dict.__setitem__(SETTINGS, k, v)  # in memory only (the settings file is ours anyway, under work_dir)
    uu.WEIGHTS_DIR = work_dir / "weights"
    uu.RUNS_DIR = work_dir / "runs"
    try:
        from ultralytics.utils.events import events

        events.enabled = False
    except Exception:
        pass


def nvidia_smi() -> dict:
    q = "name,driver_version,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.max.sm,power.draw,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        vals = [v.strip() for v in out.splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:  # pragma: no cover - informational only
        return {"error": str(e)}


BUSY_PCT = 5.0  # a foreign process using more than this much of any GPU engine counts as contention


def our_pids() -> set[int]:
    """This process, its parent benchmark processes, and Windows' System process (pid 4), whose copy-engine work is
    paging done on behalf of whichever process is using the GPU."""
    return {4, os.getpid(), *(int(p) for p in os.environ.get("FOLO_BENCH_PIDS", "").split(",") if p)}


def proc_label(pid: int) -> str:
    try:
        import psutil

        return f"pid {pid} {psutil.Process(pid).name()}"
    except Exception:
        return f"pid {pid}"


def foreign_gpu_load() -> dict[str, float] | None:
    """GPU engine utilisation (%) of every other process using the GPU right now, {"pid <n> <name>": pct}, or None
    when the query failed (unknown, which is not the same as quiet).

    Windows: per-process "GPU Engine" performance counters (summed over engines), so an idle process that merely
    holds a CUDA context is not counted. Elsewhere: nvidia-smi compute processes, reported as 100 (load unknown).
    """
    ours = our_pids()
    load: dict[int, float] = {}
    if sys.platform == "win32":
        ps = (
            r"(Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples | "
            "Where-Object CookedValue -gt 0.5 | ForEach-Object { $_.InstanceName + ' ' + $_.CookedValue }"
        )
        try:
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if res.returncode != 0:
            return None
        for line in res.stdout.splitlines():
            m = re.match(r"pid_(\d+)_\S+\s+([\d.,]+)$", line.strip())  # the decimal mark follows the locale
            if m and int(m.group(1)) not in ours:
                pid = int(m.group(1))
                load[pid] = load.get(pid, 0.0) + float(m.group(2).replace(",", "."))
    else:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if res.returncode != 0:
            return None
        for line in res.stdout.split():
            if line.strip().isdigit() and int(line) not in ours:
                load[int(line)] = 100.0
    return {proc_label(pid): round(v, 1) for pid, v in load.items() if v > BUSY_PCT}


class ContentionWatch(threading.Thread):
    """Samples other processes' GPU load back to back in the background (one sample spans ~2.5 s on Windows).

    tag(row) covers the window since the previous tag()/reset(): it records the peak load of every sample whose time
    span overlaps that window (`other_gpu_load_pct`), how many successful samples that was (`contention_samples`)
    and how many queries failed (`contention_errors`). A row with 0 successful samples was not observed, which is
    different from a row observed to be quiet.
    """

    def __init__(self, gap_s: float = 0.5):
        super().__init__(daemon=True)
        self.gap_s = gap_s
        self.lock = threading.Lock()
        self.samples: list[tuple[float, float, dict[str, float] | None]] = []  # (start, end, load or None = failed)
        self.window_start = time.perf_counter()
        self.paused = threading.Event()  # set during CPU timing: sampling spawns PowerShell, which costs CPU

    def run(self) -> None:
        while True:
            if not self.paused.is_set():
                t0 = time.perf_counter()
                load = foreign_gpu_load()
                with self.lock:
                    self.samples.append((t0, time.perf_counter(), load))
            time.sleep(self.gap_s)

    def tag(self, row: dict) -> dict:
        now = time.perf_counter()
        with self.lock:
            hits = [load for t0, t1, load in self.samples if t1 > self.window_start and t0 < now]
            ok = [h for h in hits if h is not None]
            peak: dict[str, float] = {}
            for load in ok:
                for k, v in load.items():
                    peak[k] = max(peak.get(k, 0.0), v)
            row["other_gpu_load_pct"] = dict(sorted(peak.items()))
            row["contention_samples"] = len(ok)
            row["contention_errors"] = len(hits) - len(ok)
            self.samples = [s for s in self.samples if s[1] > now - 60]  # keep the list short
            self.window_start = now
        return row

    def reset(self) -> None:
        with self.lock:
            self.window_start = time.perf_counter()


WATCH = ContentionWatch()


def wait_for_quiet_gpu(max_wait_s: float, max_failures: int = 3) -> dict[str, float] | None:
    """Block until no other process loads the GPU (or max_wait_s passes). Returns the load still there, or None if
    the load could not be queried (after max_failures consecutive failures the run proceeds, recorded as unknown)."""
    t0 = time.perf_counter()
    failures = 0
    load = foreign_gpu_load()
    while load != {} and time.perf_counter() - t0 < max_wait_s:
        if load is None:
            failures += 1
            if failures >= max_failures:
                print("[wait] cannot query GPU load; proceeding with contention unknown", flush=True)
                break
            time.sleep(5)
        else:
            failures = 0
            print(f"[wait] GPU busy: {load} (% engine use); waited {time.perf_counter() - t0:.0f} s", flush=True)
            time.sleep(20)
        load = foreign_gpu_load()
    if load:
        print(f"[wait] giving up after {max_wait_s:.0f} s, measuring under contention: {load}", flush=True)
    WATCH.reset()
    return load


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor()


def environment() -> dict:
    import cv2
    import numpy
    import psutil
    import torch
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "os": platform.platform(),
        "cpu": cpu_name(),
        "cpu_logical": os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "torch_default_threads": torch.get_num_threads(),  # 1 when importing ultralytics set OMP_NUM_THREADS=1
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi(),
        "other_gpu_load_at_start": foreign_gpu_load(),
        "tf32_defaults": {
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }


# --------------------------------------------------------------------------------------------------------------------
# models


def section_params(weights_dir: Path) -> dict:
    from ultralytics import YOLO

    out = {}
    for name in MODELS:
        m = YOLO(str(weights_dir / f"{name}.pt")).model.float()
        n_l, n_p, n_g, gflops = m.info(verbose=True, imgsz=IMGSZ)  # returns None unless verbose
        m.fuse(verbose=False)
        f_l, f_p, _, f_gflops = m.info(verbose=True, imgsz=IMGSZ)  # returns None unless verbose
        out[name] = {
            "layers": n_l,
            "params": n_p,
            "gflops_640": round(gflops, 3),
            "fused_layers": f_l,
            "fused_params": f_p,
            "fused_gflops_640": round(f_gflops, 3),
            "nc": len(m.names),
        }
        print(f"[params] {name}: {n_p:,} params, {gflops:.1f} GFLOPs (fused {f_p:,}, {f_gflops:.1f})", flush=True)
    return out


def section_gpu_fwd(weights_dir: Path, quick: bool) -> list[dict]:
    import torch

    rows = []
    for name in MODELS:
        for precision in ["fp32", "fp32-strict", "fp16"]:
            half = precision == "fp16"
            m = load_detection_model(weights_dir / f"{name}.pt", "cuda", half)
            for bench_on in [False, True]:
                torch.backends.cudnn.benchmark = bench_on
                torch.backends.cudnn.allow_tf32 = precision != "fp32-strict"
                torch.backends.cuda.matmul.allow_tf32 = False  # PyTorch default; convs are what matter here
                for batch in [1, 16]:
                    x = torch.rand(batch, 3, IMGSZ, IMGSZ, device="cuda", dtype=torch.half if half else torch.float)
                    iters = (100 if batch == 1 else 30) // (4 if quick else 1)

                    def fn(m=m, x=x):
                        with torch.inference_mode():
                            return m(x)

                    r = time_cuda(fn, batch, warmup=25 if batch == 1 else 10, iters=iters, tp_iters=iters)
                    r.update(model=name, precision=precision, batch=batch, cudnn_benchmark=bench_on)
                    rows.append(WATCH.tag(r))
                    print(
                        f"[gpu_fwd] {name} {precision:11s} b{batch:<2d} cudnn.bench={int(bench_on)}: "
                        f"median {r['median_ms']:.3f} ms, p90 {r['p90_ms']:.3f} ms, "
                        f"{r['throughput_images_per_s']:.0f} img/s",
                        flush=True,
                    )
            del m
            torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    return rows


def section_gpu_e2e(weights_dir: Path, data_dir: Path, quick: bool) -> list[dict]:
    import torch

    torch.backends.cudnn.benchmark = False  # predictor default
    images = sample_images(data_dir, 16)
    rows = []
    for name in MODELS:
        for precision in ["fp32", "fp16"]:
            for batch in [1, 16]:
                iters = (100 if batch == 1 else 30) // (4 if quick else 1)
                r = predict_e2e(
                    weights_dir / f"{name}.pt", "cuda:0", precision == "fp16", batch, images, 10 if batch == 1 else 5, iters
                )
                r.update(model=name, precision=precision, batch=batch)
                rows.append(WATCH.tag(r))
                print(
                    f"[gpu_e2e] {name} {precision} b{batch:<2d}: median {r['median_ms']:.3f} ms, "
                    f"p90 {r['p90_ms']:.3f} ms, {r['throughput_images_per_s']:.0f} img/s, "
                    f"ul speed {r['ultralytics_speed_ms_per_image']}",
                    flush=True,
                )
                torch.cuda.empty_cache()
    return rows


def section_cpu(weights_dir: Path, data_dir: Path, threads: list[int], quick: bool, max_wait_s: float) -> list[dict]:
    """GPU-contention sampling is paused here; each row waits for (and records) a quiet CPU instead."""
    import torch

    images = sample_images(data_dir, 1)
    rows = []
    default_threads = torch.get_num_threads()
    WATCH.paused.set()
    time.sleep(3.0)  # let an in-flight sample finish
    for nt in threads:
        torch.set_num_threads(nt)
        for name in MODELS:
            iters = {"yolo26n": 50, "yolo26s": 30, "yolo26m": 20}[name] // (4 if quick else 1)
            m = load_detection_model(weights_dir / f"{name}.pt", "cpu", False)
            x = torch.rand(1, 3, IMGSZ, IMGSZ)

            def fn(m=m, x=x):
                with torch.inference_mode():
                    return m(x)

            load = wait_for_quiet_cpu(max_wait_s)
            c0 = cpu_snapshot()
            r = time_host(fn, 1, warmup=5, iters=iters)
            r["cpu_cores_busy"] = cpu_cores_busy(c0, cpu_snapshot())
            r.update(model=name, precision="fp32", batch=1, threads=nt, kind="forward", system_cpu_pct_before=load)
            r["memory_format"] = "channels_last" if torch.backends.mkldnn.is_available() else "contiguous"
            rows.append(r)
            print(
                f"[cpu] {name} fwd threads={nt}: median {r['median_ms']:.2f} ms, p90 {r['p90_ms']:.2f} ms "
                f"(cpu load before {load}%)",
                flush=True,
            )
            load = wait_for_quiet_cpu(max_wait_s)
            c0 = cpu_snapshot()
            r = predict_e2e(weights_dir / f"{name}.pt", "cpu", False, 1, images, 3, iters, threads=nt)
            r["cpu_cores_busy"] = cpu_cores_busy(c0, cpu_snapshot())
            r.update(model=name, precision="fp32", batch=1, threads=nt, kind="e2e", system_cpu_pct_before=load)
            rows.append(r)
            print(f"[cpu] {name} e2e threads={nt}: median {r['median_ms']:.2f} ms, p90 {r['p90_ms']:.2f} ms", flush=True)
    for name in MODELS:  # as a user gets it: Ultralytics picks the thread count itself
        iters = {"yolo26n": 50, "yolo26s": 30, "yolo26m": 20}[name] // (4 if quick else 1)
        load = wait_for_quiet_cpu(max_wait_s)
        c0 = cpu_snapshot()
        r = predict_e2e(weights_dir / f"{name}.pt", "cpu", False, 1, images, 3, iters, threads=None)
        r["cpu_cores_busy"] = cpu_cores_busy(c0, cpu_snapshot())
        r.update(model=name, precision="fp32", batch=1, threads=r["torch_threads_seen"][-1], kind="e2e-ultralytics-default")
        r["system_cpu_pct_before"] = load
        rows.append(r)
        print(f"[cpu] {name} e2e UL default ({r['torch_threads_seen']} threads): median {r['median_ms']:.2f} ms", flush=True)
    torch.set_num_threads(default_threads)
    WATCH.reset()
    WATCH.paused.clear()
    return rows


def section_train_step(weights_dir: Path, quick: bool) -> list[dict]:
    """Forward + loss + backward + SGD step on one fixed batch already on the GPU: the compute ceiling of training."""
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    torch.backends.cudnn.benchmark = False
    g = torch.Generator().manual_seed(0)
    rows = []
    batch = 16
    for name in TRAIN_MODELS:
        for amp in [True, False]:
            m = YOLO(str(weights_dir / f"{name}.pt")).model.float().cuda().train()
            for p in m.parameters():
                p.requires_grad_(True)
            m.args = get_cfg()  # loss gains (box/cls/dfl) from the default hyperparameters
            img = torch.rand(batch, 3, IMGSZ, IMGSZ, generator=g).cuda()
            per_img = 8
            n = batch * per_img
            xy = torch.rand(n, 2, generator=g) * 0.8 + 0.1
            wh = torch.rand(n, 2, generator=g) * 0.15 + 0.03
            targets = {
                "img": img,
                "batch_idx": torch.arange(batch).repeat_interleave(per_img).float().cuda(),
                "cls": torch.randint(0, len(m.names), (n, 1), generator=g).float().cuda(),
                "bboxes": torch.cat([xy, wh], 1).cuda(),
            }
            opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.937, nesterov=True, weight_decay=5e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            torch.cuda.reset_peak_memory_stats()

            def step(m=m, opt=opt, scaler=scaler, amp=amp, targets=targets):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss, _ = m.loss(targets)
                scaler.scale(loss.sum()).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            iters = 30 // (3 if quick else 1)
            r = time_cuda(step, batch, warmup=10, iters=iters, tp_iters=iters)
            r.update(
                model=name,
                amp=amp,
                batch=batch,
                max_memory_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                max_memory_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20),
            )
            rows.append(WATCH.tag(r))
            print(
                f"[train_step] {name} amp={int(amp)}: median {r['median_ms']:.2f} ms/step, "
                f"{r['throughput_images_per_s']:.0f} img/s, peak alloc {r['max_memory_allocated_mib']} MiB",
                flush=True,
            )
            del m, opt, targets
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------------------------------------------------
# synthetic dataset + trainer


def train_one(weights: Path, data_yaml: Path, work_dir: Path, amp: bool, epochs: int, out_json: Path) -> None:
    """Runs in its own process. Times every training batch (CUDA-synchronised) via trainer callbacks."""
    import torch
    from ultralytics import YOLO

    WATCH.start()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(weights))
    stamps: dict[int, list[float]] = {}
    counts: dict[int, list[int]] = {}  # images in each completed batch (the last batch of an epoch can be partial)
    epoch_wall: dict[int, list[float]] = {}
    epoch_cpu: dict[int, list[tuple[float, float, float]]] = {}  # data loading is CPU-bound: record who used the CPU
    mem = {}
    trainer_state = {"pending": 0}

    def on_train_start(tr):
        orig = tr.preprocess_batch

        def counted(b):  # a batch that hits OOM is dropped by the trainer; it only counts once it reaches batch end
            trainer_state["pending"] = int(b["img"].shape[0])
            return orig(b)

        tr.preprocess_batch = counted

    def on_epoch_start(tr):
        torch.cuda.synchronize()
        epoch_wall[tr.epoch] = [time.perf_counter()]
        epoch_cpu[tr.epoch] = [cpu_snapshot()]
        stamps[tr.epoch] = [time.perf_counter()]
        counts[tr.epoch] = []

    def on_batch_end(tr):
        torch.cuda.synchronize()
        stamps[tr.epoch].append(time.perf_counter())
        counts[tr.epoch].append(trainer_state["pending"])

    def on_epoch_end(tr):  # runs before validation
        torch.cuda.synchronize()
        epoch_wall[tr.epoch].append(time.perf_counter())
        epoch_cpu[tr.epoch].append(cpu_snapshot())
        trainer_state["batch_size"] = tr.batch_size
        mem["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        mem["max_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
        mem["nvidia_smi_memory_used_mib"] = nvidia_smi().get("memory.used")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    batch = 16
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=epochs,
        batch=batch,
        workers=8,
        amp=amp,
        device=0,
        optimizer="MuSGD",  # what 'auto' picks for real (>10k iteration) runs; short runs would get AdamW
        close_mosaic=0,  # keep mosaic on in both epochs (with close_mosaic=10 and epochs=2 it would close at once)
        val=False,
        plots=False,
        save=False,
        project=str(work_dir / "runs"),
        name=f"{weights.stem}-amp{int(amp)}",
        exist_ok=True,
        verbose=False,
        seed=0,
    )
    total = time.perf_counter() - t0
    out_json.write_text(
        json.dumps(
            {
                "model": weights.stem,
                "amp": amp,
                "batch": batch,
                "effective_batch": trainer_state.get("batch_size", batch),  # Ultralytics halves it on OOM
                "imgsz": IMGSZ,
                "workers": 8,
                **summarize_training(stamps, counts, epoch_wall, epoch_cpu),
                "total_train_call_s": round(total, 2),
                **mem,
                **WATCH.tag({}),  # other_gpu_load_pct + contention_samples over the whole run
            },
            indent=2,
        )
    )


def section_train(
    weights_dir: Path, work_dir: Path, data_yaml: Path, epochs: int, max_wait_s: float, repeats: int
) -> list[dict]:
    """Each configuration runs `repeats` times (runs are short, so one noisy neighbour can skew a single run)."""
    runs = [(m, True) for m in TRAIN_MODELS] + [("yolo26n", False)]
    rows = []
    for rep_i, (name, amp) in [(i, r) for i in range(repeats) for r in runs]:
        out = work_dir / f"train-{name}-amp{int(amp)}-r{rep_i}.json"
        out.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_train-one",
            str(weights_dir / f"{name}.pt"),
            "--amp",
            str(int(amp)),
            "--epochs",
            str(epochs),
            "--work-dir",
            str(work_dir),
            "--_data-yaml",
            str(data_yaml),
            "--_out-json",
            str(out),
        ]
        cpu_load = wait_for_quiet_cpu(min(max_wait_s, 600), below_pct=20.0)  # the data loader needs the CPU
        wait_for_quiet_gpu(max_wait_s)
        print(f"[train] {name} amp={int(amp)} (cpu load before {cpu_load}%) ...", flush=True)
        subprocess.run(cmd, check=True, cwd=work_dir)
        WATCH.reset()  # our own child was on the GPU; it reports other processes itself
        r = json.loads(out.read_text())
        r["repeat"] = rep_i
        r["system_cpu_pct_before"] = cpu_load
        rows.append(r)
        last = r["epochs"][-1]
        print(
            f"[train] {name} amp={int(amp)} r{rep_i}: median-batch equivalent {r['steady_images_per_s']:.0f} img/s, "
            f"last epoch {last['images_per_s']:.0f} img/s "
            f"(median batch {last['median_batch_ms']:.1f} ms), peak reserved {r.get('max_memory_reserved_mib')} MiB",
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", type=Path, default=Path("."))
    ap.add_argument("--work-dir", type=Path, default=Path("D:/folo-bench"))
    ap.add_argument("--out", type=Path, default=Path("bench/baselines/results.json"))
    ap.add_argument("--sections", default="params,gpu_fwd,gpu_e2e,cpu,train_step,train")
    ap.add_argument("--cpu-threads", default="", help="comma list, default: logical cores, physical cores, 8")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-repeats", type=int, default=2)
    ap.add_argument("--train-append", action="store_true", help="append train runs to the existing results")
    ap.add_argument("--quick", action="store_true", help="fewer iterations (smoke test)")
    ap.add_argument("--max-wait", type=float, default=3600, help="seconds to wait for other GPU jobs per section")
    ap.add_argument("--amp", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--_train-one", dest="train_one", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--_out-json", dest="out_json", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--_data-yaml", dest="data_yaml", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    isolate_ultralytics(work_dir)
    seed = 0
    data_dir = work_dir / f"data-{args.n_train}-{args.n_val}-seed{seed}"  # one directory per dataset, never resized

    os.environ["FOLO_BENCH_PIDS"] = ",".join(p for p in [os.environ.get("FOLO_BENCH_PIDS"), str(os.getpid())] if p)
    if args.train_one:
        train_one(args.train_one, args.data_yaml, work_dir, bool(args.amp), args.epochs, args.out_json)
        return

    import psutil
    import torch

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    weights_dir = args.weights_dir.resolve()
    # A missing checkpoint would make Ultralytics download it next to the given path, i.e. into --weights-dir.
    missing = [f"{m}.pt" for m in dict.fromkeys(MODELS + TRAIN_MODELS) if not (weights_dir / f"{m}.pt").is_file()]
    if missing:
        raise SystemExit(f"missing weights in {weights_dir}: {', '.join(missing)}")
    if "train" in sections and args.n_train < 16 * 8:
        raise SystemExit("--n-train must give at least 8 batches of 16 per epoch for the steady-state statistics")
    data_yaml = make_dataset(data_dir, args.n_train, args.n_val, seed)
    threads = (
        [int(t) for t in args.cpu_threads.split(",")]
        if args.cpu_threads
        else list(dict.fromkeys([os.cpu_count(), psutil.cpu_count(logical=False), 8]))
    )

    env = environment()
    config = {
        "imgsz": IMGSZ,
        "models": MODELS,
        "train_models": TRAIN_MODELS,
        "synthetic_dataset": {"train": args.n_train, "val": args.n_val, "seed": seed, "classes": 3, "path": str(data_dir)},
        "epochs": args.epochs,
        "quick": args.quick,
        "cpu_threads": threads,
    }
    # Top-level environment/config describe the latest invocation; sections_meta records, per section, the
    # environment and config it was actually measured with (sections can come from different invocations).
    results: dict = {
        "schema": 2,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": env,
        "config": config,
        "sections_meta": {},
        "gpu_state": {},
    }
    prev: dict = {}
    if args.out.exists():  # merge, so sections can be re-run one at a time
        prev = json.loads(args.out.read_text())
        prev_meta = prev.get("sections_meta", {})
        for k in ("params", "gpu_fwd", "gpu_e2e", "cpu", "train_step", "train"):
            if k in prev and (k not in sections or (k == "train" and args.train_append)):
                results["sections_meta"][k] = prev_meta.get(
                    k, {"environment": prev["environment"], "config": prev["config"], "backfilled": True}
                )
            if k in prev and k not in sections:
                results[k] = prev[k]
        results["gpu_state"] = {k: v for k, v in prev.get("gpu_state", {}).items() if k not in sections}
        if args.train_append and "train" in sections and "train" in results["sections_meta"]:
            def train_key(m: dict) -> tuple:
                e, c = m["environment"], m["config"]
                ds = c["synthetic_dataset"]
                return (e["torch"], e["ultralytics"], e["gpu"], e["cpu"], ds["train"], ds["val"], c.get("epochs", 2), c["quick"])

            if train_key(results["sections_meta"]["train"]) != train_key({"environment": env, "config": config}):
                raise SystemExit("--train-append: the existing train runs used a different environment or config")

    WATCH.start()
    for s in sections:
        still_busy = wait_for_quiet_gpu(args.max_wait) if s not in ("params", "cpu") else {}
        t0 = time.perf_counter()
        before = nvidia_smi()
        if s == "params":
            results[s] = section_params(weights_dir)
        elif s == "gpu_fwd":
            results[s] = section_gpu_fwd(weights_dir, args.quick)
        elif s == "gpu_e2e":
            results[s] = section_gpu_e2e(weights_dir, data_dir, args.quick)
        elif s == "cpu":
            results[s] = section_cpu(weights_dir, data_dir, threads, args.quick, min(args.max_wait, 120))
        elif s == "train_step":
            results[s] = section_train_step(weights_dir, args.quick)
        elif s == "train":
            prev_runs = prev.get("train", []) if args.train_append else []
            new = section_train(weights_dir, work_dir, data_yaml, args.epochs, args.max_wait, args.train_repeats)
            first = max((r["repeat"] for r in prev_runs), default=-1) + 1
            for r in new:
                r["repeat"] += first
            results[s] = prev_runs + new
        else:
            raise SystemExit(f"unknown section {s!r}")
        rows = results[s] if isinstance(results[s], list) else []
        results["gpu_state"][s] = {
            "before": before,
            "after": nvidia_smi(),
            "other_gpu_load_at_start": still_busy,
            "rows_with_other_gpu_load": sum(bool(r.get("other_gpu_load_pct")) for r in rows),
            "rows_unobserved": sum(r.get("contention_samples") == 0 for r in rows),
            "section_wall_s": round(time.perf_counter() - t0, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),  # sections may come from different runs
        }
        if not (s == "train" and args.train_append and "train" in results["sections_meta"]):
            results["sections_meta"][s] = {"environment": env, "config": config}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"== {s} done in {time.perf_counter() - t0:.1f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
