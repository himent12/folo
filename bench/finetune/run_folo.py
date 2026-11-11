"""Fine-tune a FOLO detector (folo-n by default) through a running `folo serve` (local or cloud) and score every run
with the common scorer.

    # cloud: the server is a control plane, one rented pod per run, three side by side
    python bench/finetune/run_folo.py --server http://127.0.0.1:7979 --data-root D:/folo-data/finetune \
        --work-dir D:/folo-data/runs-finetune/folo --results bench/finetune/results --seeds 0,1,2 --epochs 100 \
        --compute cc_1f2e:"NVIDIA GeForce RTX 4090@community" --max-hours 0.5 --parallel 3 --budget-usd 1.0

    # local: the server trains the runs one after another on --device (a smoke run on the CPU here)
    python bench/finetune/run_folo.py --server http://127.0.0.1:7979 ... --device cpu --epochs 2 --imgsz 320

    # another size: --arch names the model, --weights must be one of that architecture
    python bench/finetune/run_folo.py ... --arch folo-s --weights <published name or model id of folo-s weights>

For each (dataset, variant, seed) of --arch it registers the dataset (`POST /api/datasets`, YOLO), starts a run (`POST
/api/runs` with FOLO's defaults: epochs, batch 16, 640 px, `optimizer: auto`, `weights` = --weights for the
`pretrained` variant, none for `scratch`, and `compute` for a cloud run), follows it to the end, then evaluates the
run's best model (`Run.model_id`, its `weights/best.folo`) on the evaluation split with this machine's `folo val
--save-json` (conf 0.001, iou 0.7, at most 300 detections, the model's 640 px) against the dataset's
annotations/instances_<split>.json, and scores the detections with pycocotools (scorer.py). FOLO's patience
(100 epochs, not settable through the API) cannot stop a run of at most 100 epochs early.

Restartable: a run whose result file exists is skipped; a run started before (work-dir/state.json, or a run of the
same name on the server) is followed again instead of started twice. On Ctrl+C, an error or a spent budget every
run it started that is still active is stopped through the API, and the server's cloud instances of those runs are
checked (and terminated) before it exits.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import datasets  # noqa: E402

TERMINAL = {"completed", "failed", "stopped"}
USER_AGENT = "folo-bench-finetune/1"


class Api:
    """The few calls of docs/API.md this needs, with `requests` and a custom User-Agent (RunPod's proxy blocks
    python-urllib's)."""

    def __init__(self, base: str, token: str | None = None):
        import requests

        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"

    def call(self, method: str, path: str, body=None, ok=(200, 201, 204), retries: int = 5):
        last = None
        for attempt in range(retries):
            try:
                r = self.s.request(method, self.base + path, json=body, timeout=60)
            except Exception as e:  # connection reset, timeout: retried
                last = e
                time.sleep(min(30, 2**attempt))
                continue
            if r.status_code in ok:
                return r.json() if r.content else None
            if r.status_code >= 500 and attempt + 1 < retries:
                last = RuntimeError(f"{method} {path}: {r.status_code} {r.text[:300]}")
                time.sleep(min(30, 2**attempt))
                continue
            raise ApiError(r.status_code, r.text)
        raise RuntimeError(f"{method} {path}: {last}")

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body)


class ApiError(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:500]}")
        self.status = status
        try:
            self.code = json.loads(text)["error"]["code"]
        except Exception:
            self.code = None


def parse_compute(text: str | None, max_hours: float, disk_gb: int | None) -> dict | None:
    """`<connection id>:<offer id>` (the offer id may itself contain ':') as TrainConfig.compute."""
    if not text:
        return None
    conn, sep, offer = text.partition(":")
    if not sep or not conn or not offer:
        raise SystemExit(f"--compute {text!r}: want <connection id>:<offer id>")
    c = {"connection_id": conn, "offer_id": offer, "max_hours": max_hours}
    if disk_gb is not None:
        c["disk_gb"] = disk_gb
    return c


def train_config(key: common.RunKey, dataset_id: str, args, compute: dict | None) -> dict:
    """The POST /api/runs body of one run: FOLO's defaults but for the protocol's fixed settings."""
    cfg = {
        "name": run_name(key),
        "dataset_id": dataset_id,
        "arch": key.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": key.seed,
        "device": "auto" if compute else args.device,
    }
    if key.variant == "pretrained":
        cfg["weights"] = args.weights
    if compute:
        cfg["compute"] = compute
    return cfg


def run_name(key: common.RunKey) -> str:
    return f"ft-{key.model}-{key.dataset}-{key.variant}-s{key.seed}"


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def wall_seconds(run: dict) -> float | None:
    a, b = parse_time(run.get("started")), parse_time(run.get("finished"))
    return (b - a).total_seconds() if a and b else None


class State:
    """work-dir/state.json: run stem -> server run id, so a restart follows a run instead of starting it again."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = json.loads(path.read_text()) if path.is_file() else {}

    def get(self, stem):
        with self.lock:
            return self.data.get(stem)

    def set(self, stem, run_id):
        with self.lock:
            self.data[stem] = run_id
            common.write_json_atomic(self.path, self.data)


class Matrix:
    def __init__(self, args):
        self.args = args
        self.api = Api(args.server, args.token)
        self.results = Path(args.results)
        self.work = Path(args.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.state = State(self.work / "state.json")
        self.compute = parse_compute(args.compute, args.max_hours, args.disk_gb)
        self.stop = threading.Event()
        self.active: dict[str, str] = {}  # run id -> stem, the runs this process started or follows
        self.lock = threading.Lock()
        self.dataset_ids: dict[str, str] = {}
        self.spent: dict[str, float] = {}  # run id -> cloud cost so far
        self.base: dict | None = None  # the --weights model as the server lists it (pretrained variant)

    def base_model(self) -> dict:
        """The --weights model on the server (a published name or a model id), checked to be an --arch model, so a
        run cannot train another architecture than its result file names."""
        if self.base is None:
            want = self.args.weights
            models = self.api.get("/api/models")["models"]
            hits = [m for m in models if want in (m["id"], m["name"])]
            if not hits:
                raise SystemExit(f"--weights {want!r}: no such model on the server (folo weights pull {want}?)")
            m = hits[0]
            if m["arch"] != self.args.arch:
                raise SystemExit(f"--weights {want!r} is a {m['arch']} model, not --arch {self.args.arch}")
            m = dict(m)
            p = Path(m.get("path") or "")
            m["sha256"] = datasets.sha256_file(p) if p.is_file() else None
            self.base = m
        return self.base

    # ---------------------------------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def spent_usd(self) -> float:
        with self.lock:
            return sum(self.spent.values())

    def dataset_id(self, name: str) -> str:
        with self.lock:
            if name in self.dataset_ids:
                return self.dataset_ids[name]
        path = (Path(self.args.data_root) / name).resolve()
        ds = self.api.post("/api/datasets", {"name": f"ft-{name}", "path": str(path), "format": "yolo"})
        want = list(datasets.SPECS[name].classes)
        if ds["classes"] != want:
            raise RuntimeError(f"{name}: the server reads classes {ds['classes']}, expected {want}")
        with self.lock:
            self.dataset_ids[name] = ds["id"]
        return ds["id"]

    def find_run(self, key: common.RunKey) -> dict | None:
        """The run this key started before: by the id in state.json, else the newest run of its name that is not
        failed or stopped (the server lists newest first)."""
        rid = self.state.get(key.stem)
        if rid:
            try:
                run = self.api.get(f"/api/runs/{rid}")
                if run["status"] not in ("failed", "stopped"):
                    return run
            except ApiError as e:
                if e.status != 404:
                    raise
        for run in self.api.get("/api/runs")["runs"]:
            if run["name"] == run_name(key) and run["status"] not in ("failed", "stopped"):
                return run
        return None

    def follow(self, run: dict, key: common.RunKey) -> dict:
        rid = run["id"]
        with self.lock:
            self.active[rid] = key.stem
        last_line = None
        try:
            while True:
                run = self.api.get(f"/api/runs/{rid}")
                cloud = run.get("cloud") or {}
                if cloud.get("cost_usd") is not None:
                    with self.lock:
                        self.spent[rid] = float(cloud["cost_usd"])
                line = f"{key.stem}: {run['status']}" + (f" ({cloud.get('phase')})" if cloud else "")
                line += f" epoch {run.get('epoch')}/{run.get('epochs')}"
                if run.get("best"):
                    line += f" best mAP50-95 {run['best']['map50_95']:.3f}"
                if line != last_line:
                    self.log(line)
                    last_line = line
                done = run["status"] in TERMINAL and (not cloud or cloud.get("phase") == "released")
                if done:
                    return run
                if self.stop.is_set():
                    raise KeyboardInterrupt
                if self.args.budget_usd is not None and self.spent_usd() > self.args.budget_usd:
                    self.log(f"budget of ${self.args.budget_usd:.2f} spent (${self.spent_usd():.2f}): stopping")
                    self.stop.set()
                    raise KeyboardInterrupt
                time.sleep(self.args.poll_s)
        finally:
            with self.lock:
                if run.get("status") in TERMINAL and (not run.get("cloud") or run["cloud"].get("phase") == "released"):
                    self.active.pop(rid, None)

    def evaluate(self, key: common.RunKey, run: dict) -> dict:
        """`folo val --save-json` of the run's best model on the evaluation split, scored by scorer.py."""
        import scorer

        spec = datasets.SPECS[key.dataset]
        data_dir = (Path(self.args.data_root) / key.dataset).resolve()
        gt_path = data_dir / "annotations" / f"instances_{spec.eval_split}.json"
        model = self.api.get(f"/api/models/{run['model_id']}")
        if model.get("arch") != key.model:
            raise RuntimeError(f"{key.stem}: the run trained a {model.get('arch')}, not {key.model}")
        weights = Path(model.get("path") or "")
        if not weights.is_file():
            weights = self.work / key.stem / "best.folo"
            weights.parent.mkdir(parents=True, exist_ok=True)
            r = self.api.s.get(f"{self.api.base}/api/models/{run['model_id']}/weights", timeout=600)
            r.raise_for_status()
            weights.write_bytes(r.content)
        out_dir = self.work / key.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        det_path = out_dir / f"detections-{spec.eval_split}.json"
        cmd = [
            self.args.folo, "val", "--model", str(weights), "--data", str(data_dir), "--split", spec.eval_split,
            "--conf", "0.001", "--iou", "0.7", "--device", self.args.val_device, "--save-json", str(det_path), "--json",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"folo val failed ({p.returncode}): {p.stderr[-2000:]}")
        val = json.loads(p.stdout.strip().splitlines()[-1])
        gt = json.loads(gt_path.read_text())
        metrics = scorer.score(gt, json.loads(det_path.read_text()))
        system = self.api.get("/api/system")
        rec = common.result_record(
            key,
            epochs_planned=run["config"]["epochs"],
            epochs_run=run.get("epoch"),
            imgsz=run["config"]["imgsz"],
            batch=run["config"]["batch"],
            weights=(
                {"name": self.args.weights, "sha256": self.base_model()["sha256"]}
                if key.variant == "pretrained"
                else None
            ),
            eval_split=spec.eval_split,
            metrics=metrics,
            tool_reported={
                "map50_95": val["map50_95"],
                "map50": val["map50"],
                "source": "folo val of best.folo (FOLO's own metric)",
                "best_epoch": (run.get("best") or {}).get("epoch"),
                "train_best_map50_95": (run.get("best") or {}).get("map50_95"),
            },
            train_wall_s=wall_seconds(run),
            gpu=run.get("device"),
            versions={
                "folo_server": self.api.get("/api/health").get("version"),
                "folo_cli": subprocess.run([self.args.folo, "--version"], capture_output=True, text=True).stdout.strip(),
                "dataset_layout_sha256": json.loads((data_dir / "meta.json").read_text())["layout_sha256"],
                "resolved": run.get("resolved"),
            },
            compute=(
                {
                    "where": "cloud",
                    "provider": run["cloud"].get("provider"),
                    "offer": run["cloud"].get("offer"),
                    "cost_usd": run["cloud"].get("cost_usd"),
                    "max_hours": run["cloud"].get("max_hours"),
                    "val_device": val.get("device"),
                }
                if run.get("cloud")
                else {"where": "local", "server_data_dir": system.get("data_dir"), "val_device": val.get("device")}
            ),
        )
        rec["run_id"] = run["id"]
        rec["detections_file"] = str(det_path)
        return rec

    def one(self, key: common.RunKey) -> str:
        if self.stop.is_set():
            return "skipped"
        run = self.find_run(key)
        if run is None:
            if self.args.budget_usd is not None and self.spent_usd() >= self.args.budget_usd:
                self.log(f"{key.stem}: not started, budget spent")
                return "skipped"
            ds_id = self.dataset_id(key.dataset)
            run = self.api.post("/api/runs", train_config(key, ds_id, self.args, self.compute))
            with self.lock:
                self.active[run["id"]] = key.stem
            self.log(f"{key.stem}: started {run['id']}")
            if self.stop.is_set():  # a stop that came while the run was being created: stop_active may have missed it
                self.api.post(f"/api/runs/{run['id']}/stop")
        else:
            self.log(f"{key.stem}: following {run['id']} ({run['status']})")
        self.state.set(key.stem, run["id"])
        run = self.follow(run, key)
        if run["status"] != "completed" or not run.get("model_id"):
            self.log(f"{key.stem}: {run['status']} ({run.get('error_code')}: {run.get('error')})")
            return run["status"]
        rec = self.evaluate(key, run)
        common.write_json_atomic(key.result_path(self.results), rec)
        m = rec["metrics"]
        self.log(f"{key.stem}: AP {m['ap']:.4f} AP50 {m['ap50']:.4f} (folo val {rec['tool_reported']['map50_95']:.4f})")
        return "completed"

    # ---------------------------------------------------------------------------------------------------------
    def stop_active(self, wait_s: float = 600) -> None:
        """Stop every run still active, wait for each to end (a cloud run: to release its instance), then terminate
        any live instance of those runs that is left."""
        time.sleep(2)  # let a worker that is creating a run register it
        with self.lock:
            active = dict(self.active)
        for rid, stem in active.items():
            try:
                self.api.post(f"/api/runs/{rid}/stop")
                self.log(f"{stem}: stop requested ({rid})")
            except ApiError as e:
                if e.status != 409:  # run_not_active: it ended meanwhile
                    self.log(f"{stem}: stop failed: {e}")
        deadline = time.time() + wait_s
        pending = set(active)
        while pending and time.time() < deadline:
            for rid in list(pending):
                try:
                    run = self.api.get(f"/api/runs/{rid}")
                except Exception:
                    continue
                cloud = run.get("cloud")
                if run["status"] in TERMINAL and (not cloud or cloud.get("phase") == "released"):
                    pending.discard(rid)
            time.sleep(3)
        if pending:
            self.log(f"runs still not released after {wait_s:.0f} s: {sorted(pending)}")
        self.reap(set(active))

    def reap(self, run_ids: set[str]) -> None:
        try:
            instances = self.api.get("/api/cloud/instances")["instances"]
        except ApiError:
            return  # a server without the cloud feature
        for inst in instances:
            if inst["status"] in ("terminated", "lost"):
                continue
            if inst.get("run_id") in run_ids or not run_ids:
                self.log(f"instance {inst['id']} ({inst['name']}) is {inst['status']}: terminating")
                try:
                    self.api.call("DELETE", f"/api/cloud/instances/{inst['id']}")
                except Exception as e:
                    self.log(f"instance {inst['id']}: terminate failed: {e}")

    def live_instances(self) -> list[dict]:
        try:
            return [i for i in self.api.get("/api/cloud/instances")["instances"] if i["status"] not in ("terminated", "lost")]
        except ApiError:
            return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a FOLO detector through folo serve")
    ap.add_argument("--server", required=True, help="the folo serve URL (the control plane for cloud runs)")
    ap.add_argument("--token", default=None, help="its --token, if it has one (or FOLO_TOKEN)")
    ap.add_argument("--data-root", required=True, help="folder of datasets.py fetch")
    ap.add_argument("--work-dir", required=True, help="state.json, detections and downloaded weights go here")
    ap.add_argument("--results", required=True)
    ap.add_argument("--arch", default="folo-n", help="the architecture trained (and in the result names): folo-n, folo-s, ...")
    ap.add_argument(
        "--weights",
        default="folo-n-coco-v2",
        help="the pretrained variant's weights: a published name or a model id on the server (a .folo file in its "
        "<data dir>/models/ lists as one); must be an --arch model",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="TrainConfig.device of local runs (cpu, gpu:<n>, auto), required without --compute so that no run lands "
        "on a shared GPU by default; cloud runs always use auto (the pod's GPU)",
    )
    ap.add_argument("--datasets", default=",".join(datasets.SPECS))
    ap.add_argument("--variants", default="pretrained,scratch")
    ap.add_argument("--seeds", default="0,1,2", help="seeds of the pretrained variant")
    ap.add_argument("--scratch-seeds", default="0", help="seeds of the scratch control")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--compute", default=None, help="<connection id>:<offer id> for cloud runs (omit: local)")
    ap.add_argument("--max-hours", type=float, default=0.5, help="time limit of each cloud run")
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=1, help="runs followed side by side (cloud: pods at once)")
    ap.add_argument("--budget-usd", type=float, default=None, help="stop when the cloud runs' cost passes this")
    ap.add_argument("--folo", default="folo", help="the folo binary for val")
    ap.add_argument("--val-device", default="auto")
    ap.add_argument("--poll-s", type=float, default=10)
    args = ap.parse_args(argv)
    import os

    args.token = args.token or os.environ.get("FOLO_TOKEN")
    names = [s.name for s in datasets.pick(args.datasets)]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    keys = []
    for name in names:
        for v in variants:
            for seed in common.parse_seeds(args.seeds if v == "pretrained" else args.scratch_seeds):
                keys.append(common.RunKey("folo", args.arch, name, v, seed))
    if args.compute and args.device not in (None, "auto"):
        raise SystemExit("--device is for local runs; a cloud run trains on its pod's GPU (device auto)")
    if not args.compute and args.device is None:
        raise SystemExit("local runs need --device (cpu, gpu:<n> or auto); cloud runs need --compute")
    m = Matrix(args)
    if "pretrained" in variants:
        m.base_model()  # fail before anything starts on a wrong --weights / --arch
    todo = [k for k in keys if not k.result_path(m.results).is_file()]
    for k in keys:
        if k not in todo:
            m.log(f"{k.stem}: done already, skipped")
    if not todo:
        return 0

    def on_signal(signum, frame):
        raise KeyboardInterrupt

    for sig in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), on_signal)

    outcome: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, args.parallel))
    try:
        futures = {pool.submit(m.one, k): k for k in todo}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for f in done:
                k = futures[f]
                try:
                    outcome[k.stem] = f.result()
                except KeyboardInterrupt:
                    outcome[k.stem] = "interrupted"
                except Exception as e:
                    outcome[k.stem] = f"error: {e}"
                    m.log(f"{k.stem}: error: {e}")
            if m.stop.is_set():
                break
    except KeyboardInterrupt:
        m.log("interrupted: stopping the active runs")
    finally:
        m.stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        m.stop_active()
        live = m.live_instances()
        if live:
            m.log(f"live cloud instances on the server: {[(i['id'], i['name'], i['status']) for i in live]}")
        if m.spent:
            m.log(f"cloud cost of the runs followed (estimate): ${m.spent_usd():.3f}")
    for stem, o in sorted(outcome.items()):
        m.log(f"{stem}: {o}")
    return 0 if all(o == "completed" for o in outcome.values()) and len(outcome) == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
