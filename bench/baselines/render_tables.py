"""Print the Markdown tables of docs/BENCHMARKS.md from a results JSON written by ultralytics_bench.py.

    python bench/baselines/render_tables.py bench/baselines/results-rtx4080s.json            # print
    python bench/baselines/render_tables.py bench/baselines/results-rtx4080s.json --update docs/BENCHMARKS.md

--update replaces the text between the `<!-- tables:begin -->` and `<!-- tables:end -->` markers.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, nd=2) -> str:
    return f"{x:.{nd}f}"


def headline(d: dict) -> None:
    """One row per model: the numbers FOLO is measured against (see docs/BENCHMARKS.md for which is which)."""

    def pick(rows, **kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    print("\n### Headline (medians; img/s = throughput)\n")
    rows = []
    for m in d["params"]:
        fwd = d.get("gpu_fwd", [])
        b1 = pick(fwd, model=m, batch=1)
        e2e = d.get("gpu_e2e", [])
        cpu = pick(d.get("cpu", []), model=m)
        cpu_fwd = min(pick(cpu, kind="forward"), key=lambda r: r["median_ms"])
        cpu_e2e = min([r for r in cpu if r["kind"].startswith("e2e")], key=lambda r: r["median_ms"])
        rows.append(
            [
                m,
                f(pick(fwd, model=m, batch=1, precision="fp32", cudnn_benchmark=False)[0]["median_ms"]),
                f(pick(fwd, model=m, batch=1, precision="fp16", cudnn_benchmark=False)[0]["median_ms"]),
                f(min(r["median_ms"] for r in b1)),
                f(pick(fwd, model=m, batch=16, precision="fp32", cudnn_benchmark=False)[0]["throughput_images_per_s"], 0),
                f(pick(fwd, model=m, batch=16, precision="fp16", cudnn_benchmark=False)[0]["throughput_images_per_s"], 0),
                f(pick(e2e, model=m, batch=1, precision="fp32")[0]["median_ms"]),
                f(pick(e2e, model=m, batch=16, precision="fp16")[0]["throughput_images_per_s"], 0),
                f"{f(cpu_fwd['median_ms'], 1)} ({cpu_fwd['threads']} thr)",
                f"{f(cpu_e2e['median_ms'], 1)} ({cpu_e2e['threads']} thr)",
            ]
        )
    print(
        table(
            [
                "model",
                "GPU fwd b1 fp32 ms",
                "GPU fwd b1 fp16 ms",
                "GPU fwd b1 best ms",
                "GPU fwd b16 fp32 img/s",
                "GPU fwd b16 fp16 img/s",
                "GPU e2e b1 fp32 ms",
                "GPU e2e b16 fp16 img/s",
                "CPU fwd b1 best ms",
                "CPU e2e b1 best ms",
            ],
            rows,
        )
    )
    if "train" in d or "train_step" in d:
        print()
        rows = []
        for r in d.get("train_step", []):
            runs = pick(d.get("train", []), model=r["model"], amp=r["amp"])
            # sustained throughput: images / wall time of the last epoch (includes data loader stalls)
            best = max((x["epochs"][-1]["images_per_s"] for x in runs), default=None)
            lo = min((x["epochs"][-1]["images_per_s"] for x in runs), default=None)
            mem = max((x["max_memory_allocated_mib"] for x in runs), default="")
            rows.append(
                [
                    r["model"],
                    "AMP" if r["amp"] else "fp32 (TF32 convs)",
                    f(r["throughput_images_per_s"], 0),
                    f(best, 0) if best else "not run",
                    f"{f(lo, 0)}-{f(best, 0)} ({len(runs)} runs)" if runs else "",
                    mem,
                ]
            )
        print(
            table(
                [
                    "model",
                    "precision",
                    "train step img/s",
                    "trainer last-epoch img/s, best",
                    "trainer range",
                    "trainer peak alloc MiB",
                ],
                rows,
            )
        )


SECTIONS = ("params", "gpu_fwd", "gpu_e2e", "cpu", "train_step", "train")


def section_meta(d: dict, section: str) -> tuple[dict, dict]:
    """(environment, config) a section was measured with; schema 1 files only have the top-level ones."""
    m = d.get("sections_meta", {}).get(section)
    return (m["environment"], m["config"]) if m else (d["environment"], d["config"])


def env_rows(env: dict) -> list[list[str]]:
    smi = env.get("nvidia_smi", {})
    return [
        ["GPU", f"{env['gpu']} ({smi.get('memory.total', '?')} MiB), driver {smi.get('driver_version', '?')}"],
        ["CPU", f"{env['cpu']} ({env['cpu_physical']} cores / {env['cpu_logical']} threads)"],
        ["OS", env["os"]],
        ["Python", env["python"]],
        ["torch", f"{env['torch']} (CUDA {env['torch_cuda']}, cuDNN {env['cudnn']})"],
        ["ultralytics", env["ultralytics"]],
    ]


def main(path: str) -> None:
    d = json.load(open(path, encoding="utf-8"))
    present = [s for s in SECTIONS if s in d]
    envs = {s: env_rows(section_meta(d, s)[0]) for s in present}
    print("### Machine\n")
    if len({json.dumps(v) for v in envs.values()}) <= 1:
        rows = next(iter(envs.values()), env_rows(d["environment"]))
        print(table(["item", "value"], [*rows, ["sections", ", ".join(present)], ["date", d["date"]]]))
    else:  # sections measured on different machines or software versions: never present them as one
        for s in present:
            print(f"\n`{s}`:\n")
            print(table(["item", "value"], envs[s]))

    if "params" in d:
        headline(d)
        print("\n### Model size (Ultralytics `model.info()`, 640x640)\n")
        print(
            table(
                ["model", "params", "GFLOPs", "fused params", "fused GFLOPs", "FOLO params", "FOLO GFLOPs"],
                [
                    [m, f"{p['params']:,}", f(p["gflops_640"], 1), f"{p['fused_params']:,}", f(p["fused_gflops_640"], 1), "", ""]
                    for m, p in d["params"].items()
                ],
            )
        )

    if "gpu_fwd" in d:
        print("\n### GPU forward pass (input already on the GPU, ms; img/s = pipelined throughput)\n")
        rows = []
        for r in d["gpu_fwd"]:
            rows.append(
                [
                    r["model"],
                    r["precision"],
                    r["batch"],
                    "on" if r["cudnn_benchmark"] else "off",
                    f(r["median_ms"]),
                    f(r["p90_ms"]),
                    f(r["throughput_images_per_s"], 0),
                    "",
                ]
            )
        print(table(["model", "precision", "batch", "cudnn.benchmark", "median", "p90", "img/s", "FOLO median"], rows))

    if "gpu_e2e" in d:
        print("\n### GPU end to end (Ultralytics predictor, decoded uint8 BGR 640x640 in, boxes on the host, ms)\n")
        rows = []
        for r in d["gpu_e2e"]:
            s = r["ultralytics_speed_ms_per_image"]
            rows.append(
                [
                    r["model"],
                    r["precision"],
                    r["batch"],
                    f(r["median_ms"]),
                    f(r["p90_ms"]),
                    f(r["throughput_images_per_s"], 0),
                    f"{f(s['preprocess'])} / {f(s['inference'])} / {f(s['postprocess'])}",
                    "",
                ]
            )
        print(
            table(
                ["model", "precision", "batch", "median", "p90", "img/s", "UL pre / inf / post per img", "FOLO median"],
                rows,
            )
        )

    if "cpu" in d:
        print("\n### CPU, fp32, batch 1 (ms)\n")
        rows = []
        for r in d["cpu"]:
            rows.append(
                [
                    r["model"],
                    r["kind"],
                    r["threads"],
                    f(r["median_ms"]),
                    f(r["p90_ms"]),
                    f(r["cpu_cores_busy"]["others"], 1) if "cpu_cores_busy" in r else "n/a",
                    "",
                ]
            )
        print(table(["model", "kind", "torch threads", "median", "p90", "other procs' CPU cores", "FOLO median"], rows))

    if "train_step" in d:
        print("\n### Training step only (fixed batch 16 on the GPU, forward + loss + backward + SGD)\n")
        rows = []
        for r in d["train_step"]:
            rows.append(
                [
                    r["model"],
                    "AMP fp16" if r["amp"] else "fp32 (TF32 convs)",
                    f(r["median_ms"], 1),
                    f(r["p90_ms"], 1),
                    f(r["throughput_images_per_s"], 0),
                    r["max_memory_allocated_mib"],
                    "",
                ]
            )
        print(table(["model", "precision", "median ms/step", "p90", "img/s", "peak alloc MiB", "FOLO img/s"], rows))

    if "train" in d:
        cfg = section_meta(d, "train")[1]
        n_epochs = cfg.get("epochs", 2)
        n_img = cfg["synthetic_dataset"]["train"]
        print(
            f"\n### Ultralytics trainer (synthetic {n_img}-image set, batch 16, workers 8, mosaic on, {n_epochs} epochs)\n"
        )
        rows = []
        for r in d["train"]:
            e = r["epochs"][-1]
            cpu = e.get("cpu_cores_busy", {})
            others = r.get("other_gpu_load_pct") or {}
            rows.append(
                [
                    r["model"],
                    "AMP" if r["amp"] else "fp32 (TF32 convs)",
                    r.get("repeat", 0),
                    f(r["steady_images_per_s"], 0) if "steady_images_per_s" in r else "",
                    f(e["images_per_s"], 0),
                    f(e["median_batch_ms"], 1),
                    f(cpu["others"], 1) if cpu else "n/a",
                    ", ".join(f"{k} {v}%" for k, v in others.items()) or "none",
                    r.get("max_memory_allocated_mib", ""),
                    r.get("max_memory_reserved_mib", ""),
                ]
            )
        print(
            table(
                [
                    "model",
                    "precision",
                    "run",
                    "median-batch equivalent img/s",
                    f"epoch-{n_epochs} img/s",
                    f"epoch-{n_epochs} median batch ms",
                    f"other procs' CPU cores (epoch {n_epochs})",
                    "other GPU load",
                    "peak alloc MiB",
                    "peak reserved MiB",
                ],
                rows,
            )
        )

if __name__ == "__main__":
    args = sys.argv[1:]
    src = args[0] if args and not args[0].startswith("--") else "bench/baselines/results-rtx4080s.json"
    if "--update" in args:
        doc_path = args[args.index("--update") + 1]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(src)
        doc = open(doc_path, encoding="utf-8").read()
        begin, end = "<!-- tables:begin -->", "<!-- tables:end -->"
        head, rest = doc.split(begin, 1)
        _, tail = rest.split(end, 1)
        with open(doc_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(head + begin + "\n" + buf.getvalue().strip() + "\n" + end + tail)
    else:
        main(src)
