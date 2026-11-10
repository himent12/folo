"""Timing helpers of ultralytics_bench.py and the statistics computed from them: per-call latency and
throughput (CUDA or host), per-batch training throughput, and the CPU use around a measurement.
"""

from __future__ import annotations

import statistics
import time


def percentile(values: list[float], q: float) -> float:
    """The sorted value at index round(q * (n - 1)), q in [0, 1]."""
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summarize(ms: list[float], batch: int) -> dict:
    med = statistics.median(ms)
    return {
        "iters": len(ms),
        "median_ms": round(med, 4),
        "p90_ms": round(percentile(ms, 0.9), 4),
        "min_ms": round(min(ms), 4),
        "mean_ms": round(statistics.fmean(ms), 4),
        "images_per_s_at_median": round(batch * 1000.0 / med, 2),
    }


def time_cuda(fn, batch: int, warmup: int, iters: int, tp_iters: int) -> dict:
    """Latency: each call synchronised on both sides. Throughput: tp_iters calls back to back, one final sync."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
    res = summarize(ms, batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(tp_iters):
        fn()
    torch.cuda.synchronize()
    res["throughput_images_per_s"] = round(batch * tp_iters / (time.perf_counter() - t0), 2)
    res["throughput_iters"] = tp_iters
    return res


def time_host(fn, batch: int, warmup: int, iters: int) -> dict:
    """For calls that end on the host (fn must block until its result is on the CPU)."""
    for _ in range(warmup):
        fn()
    ms = []
    t_all = time.perf_counter()
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1e3)
    total = time.perf_counter() - t_all
    res = summarize(ms, batch)
    res["throughput_images_per_s"] = round(batch * iters / total, 2)
    return res


def wait_for_quiet_cpu(max_wait_s: float, below_pct: float = 15.0) -> float:
    """Block until system CPU load (1 s sample) is below below_pct, or max_wait_s passes. Returns the last sample."""
    import psutil

    t0 = time.perf_counter()
    while (load := psutil.cpu_percent(interval=1.0)) >= below_pct and time.perf_counter() - t0 < max_wait_s:
        if 10 <= int(time.perf_counter() - t0) % 60 < 13:  # about once a minute, and only for real waits
            print(f"[wait] CPU busy: {load}%; waited {time.perf_counter() - t0:.0f} s", flush=True)
        time.sleep(2)
    return load


def cpu_snapshot() -> tuple[float, float, float]:
    """(wall s, CPU-seconds used by the whole system, CPU-seconds used by this process and its data loader workers)."""
    import psutil

    t = psutil.cpu_times()
    ours = 0.0
    me = psutil.Process()
    for p in [me, *me.children(recursive=True)]:
        try:
            c = p.cpu_times()
            ours += c.user + c.system
        except psutil.Error:
            pass
    return time.perf_counter(), t.user + t.system, ours


def cpu_cores_busy(a: tuple[float, float, float], b: tuple[float, float, float]) -> dict:
    wall = b[0] - a[0]
    return {
        "ours": round((b[2] - a[2]) / wall, 2),
        "others": round(max(0.0, (b[1] - a[1]) - (b[2] - a[2])) / wall, 2),
    }


def summarize_training(
    stamps: dict[int, list[float]],
    counts: dict[int, list[int]],
    epoch_wall: dict[int, list[float]],
    epoch_cpu: dict[int, list[tuple[float, float, float]]],
) -> dict:
    """Per-epoch and steady-state statistics from batch-end timestamps (s) and per-batch image counts.

    stamps[e] is [epoch start, end of batch 1, ..., end of batch n]; counts[e] has the n batch sizes.
    images_per_s (per epoch) is images actually processed / epoch wall time: sustained throughput, stalls included.
    The *_at_median_batch / steady_* figures are the median of per-batch images/s, a diagnostic that ignores stalls.
    """
    epochs_out = []
    all_rates: list[float] = []
    all_dts: list[float] = []
    for e in sorted(stamps):
        s, n = stamps[e], counts[e]
        if len(n) != len(s) - 1:
            raise ValueError(f"epoch {e}: {len(s) - 1} batch timestamps but {len(n)} batch sizes")
        dts = [(b - a) * 1e3 for a, b in zip(s, s[1:])]
        rates = [k * 1000 / dt for k, dt in zip(n, dts)]
        wall = epoch_wall[e][1] - epoch_wall[e][0]
        steady = slice(2, None)  # the first batches of an epoch wait on the data loader's first prefetch
        first = slice(5, None) if e == min(stamps) else slice(None)  # run start: worker spawn, cudnn/allocator warm-up
        all_rates += rates[first]
        all_dts += dts[first]
        epochs_out.append(
            {
                "epoch": e,
                "batches": len(dts),
                "images": sum(n),
                "wall_s": round(wall, 3),
                "images_per_s": round(sum(n) / wall, 2),
                "median_batch_ms": round(statistics.median(dts[steady]), 3),
                "p90_batch_ms": round(percentile(dts[steady], 0.9), 3),
                "images_per_s_at_median_batch": round(statistics.median(rates[steady]), 2),
                "cpu_cores_busy": cpu_cores_busy(*epoch_cpu[e]),
            }
        )
    return {
        "epochs": epochs_out,
        "steady_batches": len(all_dts),
        "steady_median_batch_ms": round(statistics.median(all_dts), 3),
        "steady_images_per_s": round(statistics.median(all_rates), 2),  # median-batch equivalent
    }
