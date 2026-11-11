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

Restartable: a run whose result file exists is skipped; a run started before is followed again instead of started
twice. Each work dir has an owner id and each run a name that carries it and the fingerprint of its configuration
(run_state.py), so a run is only ever adopted by the matrix and configuration that started it; the request that
creates a run is never repeated blindly (after a lost answer the run is looked up by its name first). On Ctrl+C,
an error or a spent budget every run of this matrix that is still active is stopped through the API, the server's
cloud instances of those runs are checked (and terminated), and whatever could not be confirmed stopped is listed
with the commands that finish the job (the exit code is then 1). Work dirs of the run_folo.py before owner ids
(state.json without `"version": 2`) are not resumed: their runs are left to the version that started them.
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
import folo_api  # noqa: E402
import run_state  # noqa: E402
import scorer  # noqa: E402
from folo_api import Api, ApiError  # noqa: E402

TERMINAL = {"completed", "failed", "stopped"}


def ended(run: dict) -> bool:
    """The run is over and holds nothing: terminal, and a cloud run's instance released."""
    cloud = run.get("cloud")
    return run["status"] in TERMINAL and (not cloud or cloud.get("phase") == "released")


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


def train_config(key: common.RunKey, dataset_id: str, args, compute: dict | None, name: str) -> dict:
    """The POST /api/runs body of one run (`name`: run_state.run_name): FOLO's defaults but for the protocol's fixed
    settings."""
    cfg = {
        "name": name,
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


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def wall_seconds(run: dict) -> float | None:
    a, b = parse_time(run.get("started")), parse_time(run.get("finished"))
    return (b - a).total_seconds() if a and b else None


class Matrix:
    def __init__(self, args, api=None):
        self.args = args
        self.api = api or Api(args.server, args.token)
        self.results = Path(args.results)
        self.work = Path(args.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.state = run_state.State(self.work / "state.json")
        self.compute = parse_compute(args.compute, args.max_hours, args.disk_gb)
        self.stop = threading.Event()
        self.sleep = time.sleep
        self.active: dict[str, str] = {}  # run id -> stem, the runs this process started or follows
        self.reserved: dict[str, float] = {}  # stem -> projected cost of a launch being submitted (the budget's)
        self.discarded: dict[str, str] = {}  # run id -> stem: repeats of a creation, stopped (and reaped at the end)
        self.lock = threading.Lock()
        self.dataset_ids: dict[str, str] = {}
        self.layouts: dict[str, str] = {}  # dataset -> layout digest, hashed from its files once per invocation
        self.spent: dict[str, float] = {}  # run id -> cloud cost so far
        self.finished: set[str] = set()  # run ids that completed (their cost is a full run's)
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

    def data_dir(self, name: str) -> Path:
        return (Path(self.args.data_root) / name).resolve()

    def layout(self, name: str) -> str:
        """The layout digest of dataset `name`, hashed from its files (datasets.verify_layout) the first time."""
        with self.lock:
            if name in self.layouts:
                return self.layouts[name]
        digest = datasets.verify_layout(datasets.SPECS[name], self.data_dir(name))
        with self.lock:
            self.layouts[name] = digest
        return digest

    def run_spec(self, key: common.RunKey) -> dict:
        """Everything that defines `key`'s run: the input of its fingerprint, kept in state.json."""
        spec = {
            "dataset": key.dataset,
            "dataset_layout_sha256": self.layout(key.dataset),
            "arch": key.model,
            "variant": key.variant,
            "seed": key.seed,
            "epochs": self.args.epochs,
            "imgsz": self.args.imgsz,
            "batch": self.args.batch,
            "device": "auto" if self.compute else self.args.device,
            "compute": self.compute,
            "weights": None,
        }
        if key.variant == "pretrained":
            b = self.base_model()
            # The file's sha256 when the server's files are on this machine, else its model id on the server.
            spec["weights"] = {"name": self.args.weights, "sha256": b["sha256"]} if b["sha256"] else {
                "name": self.args.weights, "model_id": b["id"]
            }
        return spec

    # ---------------------------------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def spent_usd(self) -> float:
        with self.lock:
            return sum(self.spent.values())

    def reserve(self, stem: str) -> bool:
        """--budget-usd is checked before each launch against what is spent plus what the active runs, the launches
        being submitted and this one will still cost: each projected at the mean cost of the finished cloud runs
        (--run-cost-usd until one has finished). The check and the reservation of this launch's share are one step
        under the lock, so launches side by side cannot both pass a budget that covers one; `register` turns the
        reservation into an active run, `release` drops it. Active runs are never stopped for the budget (a stopped
        run's money is lost); --budget-hard-usd does that."""
        if self.args.budget_usd is None:
            return True
        with self.lock:
            done = [c for rid, c in self.spent.items() if rid in self.finished]
            per_run = sum(done) / len(done) if done else self.args.run_cost_usd
            active = [rid for rid in self.active if rid not in self.finished]
            spent_active = sum(self.spent.get(rid, 0.0) for rid in active)
            projected = (
                sum(self.spent.values())
                + sum(max(0.0, per_run - self.spent.get(rid, 0.0)) for rid in active)
                + sum(self.reserved.values())
                + per_run
            )
            ok = projected <= self.args.budget_usd
            if ok:
                self.reserved[stem] = per_run
        if not ok:
            self.log(
                f"budget: ${sum(self.spent.values()):.3f} spent (${spent_active:.3f} on {len(active)} active runs, "
                f"{len(self.reserved)} launching), ${projected:.3f} projected with one more run at ${per_run:.3f} "
                f"> ${self.args.budget_usd:.2f}"
            )
        return ok

    def register(self, run_id: str, stem: str) -> None:
        """`run_id` is a run of this process from now on (its reservation, if any, becomes the run)."""
        with self.lock:
            self.active[run_id] = stem
            self.reserved.pop(stem, None)

    def release(self, stem: str) -> None:
        with self.lock:
            self.reserved.pop(stem, None)

    def dataset_id(self, name: str) -> str:
        with self.lock:
            if name in self.dataset_ids:
                return self.dataset_ids[name]
        ds = self.api.post("/api/datasets", {"name": f"ft-{name}", "path": str(self.data_dir(name)), "format": "yolo"})
        want = list(datasets.SPECS[name].classes)
        if ds["classes"] != want:
            raise RuntimeError(f"{name}: the server reads classes {ds['classes']}, expected {want}")
        with self.lock:
            self.dataset_ids[name] = ds["id"]
        return ds["id"]

    # ---------------------------------------------------------------------------------------------------------
    def adoptable(self, key: common.RunKey, name: str, preferred: str | None = None) -> dict | None:
        """The run named `name` that is not failed or stopped (live, or completed and waiting to be scored), or
        None. Names carry the owner id and the fingerprint, so such a run is this matrix's run of this
        configuration. More than one (a creation sent again after its answer was lost) keeps one (`preferred`, else
        a completed one, else the oldest) and stops the others."""
        runs = [
            r
            for r in self.api.get("/api/runs")["runs"]
            if r.get("name") == name and r["status"] not in ("failed", "stopped")
        ]
        if not runs:
            return None
        keep = (
            next((r for r in runs if r["id"] == preferred), None)
            or next((r for r in runs if r["status"] == "completed"), None)
            or runs[-1]  # newest first: the oldest
        )
        for r in runs:
            if r["id"] != keep["id"]:
                self.discard(r, key)
        return keep

    def discard(self, run: dict, key: common.RunKey) -> None:
        """Stop a repeat of one of this matrix's runs; clean-up checks it ended and reaps its instance."""
        self.log(f"{key.stem}: {run['id']} repeats {run['name']} ({run['status']}): stopping it")
        with self.lock:
            self.discarded[run["id"]] = key.stem
        if run["status"] not in TERMINAL:
            try:
                self.api.post(f"/api/runs/{run['id']}/stop")
            except Exception as e:
                self.log(f"{key.stem}: stopping {run['id']} failed (clean-up tries again): {e}")

    def resume(self, key: common.RunKey, name: str, fp: str, spec: dict) -> dict | None:
        """The run this work dir started before for `key` with this configuration, or None to start one. A run of
        another configuration that is still live or completed is an error: it is never adopted."""
        entry = self.state.entry(key.stem)
        if entry is None:
            legacy = self.state.legacy_run(key.stem)
            if legacy:
                raise RuntimeError(
                    f"{key.stem}: state.json lists run {legacy} of a run_folo.py before owner ids, which this version "
                    "does not adopt: let it finish with the version that started it, or use another --work-dir"
                )
            return None
        if entry["fingerprint"] != fp:
            old = self.adoptable(key, entry["name"], entry.get("run_id"))
            diff = run_state.differences(entry.get("config") or {}, spec)
            if old is not None:
                raise RuntimeError(
                    f"{key.stem}: this work dir has run {old['id']} ({entry['name']}, {old['status']}) of another "
                    f"configuration than the one requested ({name}); they differ in {diff} (stored, requested). Use "
                    "another --work-dir, or stop that run and remove its entry from state.json"
                )
            self.log(f"{key.stem}: {entry['name']} (another configuration: {diff}) ended without a result")
            return None
        return self.adoptable(key, name, entry.get("run_id"))

    def submit(self, key: common.RunKey, name: str, cfg: dict) -> dict:
        """`POST /api/runs`, never repeated blindly: after a failure that may have created the run (no answer, a
        5xx) it is looked up by its name, adopted when it is there, and only then sent again."""
        attempts = max(1, self.args.create_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return self.api.call("POST", "/api/runs", cfg, retries=1)
            except Exception as e:
                if not folo_api.ambiguous(e):
                    raise
                self.log(f"{key.stem}: creating {name} got no clear answer ({e}); looking it up by name")
            self.sleep(min(30, 2**attempt))
            run = self.adoptable(key, name)
            if run is not None:
                self.log(f"{key.stem}: the server had created it: {run['id']}")
                return run
            if self.stop.is_set():
                raise KeyboardInterrupt
        raise RuntimeError(f"{key.stem}: {name} was not created in {attempts} attempts (each looked up by name)")

    def launch(self, key: common.RunKey, name: str, fp: str, spec: dict) -> dict | None:
        """Start `key`'s run (None: the budget would not cover it). The intent is in state.json before the POST."""
        if not self.reserve(key.stem):
            self.log(f"{key.stem}: not started: the budget would not cover it")
            return None
        try:
            cfg = train_config(key, self.dataset_id(key.dataset), self.args, self.compute, name)
            self.state.set_intent(key.stem, name, fp, spec)
            run = self.submit(key, name, cfg)
            self.register(run["id"], key.stem)
        finally:
            self.release(key.stem)
        self.state.set_run(key.stem, run["id"])
        self.log(f"{key.stem}: started {run['id']} ({name})")
        if self.stop.is_set():  # a stop that came while the run was being created: stop_active may have missed it
            try:
                self.api.post(f"/api/runs/{run['id']}/stop")
            except Exception as e:
                self.log(f"{key.stem}: stop failed (clean-up tries again): {e}")
        return run

    def follow(self, run: dict, key: common.RunKey) -> dict:
        rid = run["id"]
        self.register(rid, key.stem)
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
                if ended(run):
                    if run["status"] == "completed":
                        with self.lock:
                            self.finished.add(rid)
                    return run
                if self.stop.is_set():
                    raise KeyboardInterrupt
                if self.args.budget_hard_usd is not None and self.spent_usd() > self.args.budget_hard_usd:
                    self.log(f"hard budget of ${self.args.budget_hard_usd:.2f} spent (${self.spent_usd():.2f}): stopping")
                    self.stop.set()
                    raise KeyboardInterrupt
                self.sleep(self.args.poll_s)
        finally:
            with self.lock:
                if "status" in run and ended(run):
                    self.active.pop(rid, None)

    def evaluate(self, key: common.RunKey, run: dict, trained_on: str) -> dict:
        """`folo val --save-json` of the run's best model on the evaluation split, scored by scorer.py. The dataset's
        files are hashed again first: they must still be the layout the run trained on (`trained_on`)."""
        spec = datasets.SPECS[key.dataset]
        data_dir = self.data_dir(key.dataset)
        layout = datasets.verify_layout(spec, data_dir)
        if layout != trained_on:
            raise RuntimeError(f"{key.stem}: {data_dir} has layout {layout} now, the run trained on {trained_on}")
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
                "dataset_layout_sha256": layout,
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
        spec = self.run_spec(key)
        fp = run_state.fingerprint(spec)
        name = run_state.run_name(key.stem, self.state.owner, fp)
        run = self.resume(key, name, fp, spec)
        for attempt in range(self.args.refused_retries + 1):
            if run is None:
                run = self.launch(key, name, fp, spec)
                if run is None:
                    return "skipped: the budget would not cover it"
            else:
                self.register(run["id"], key.stem)
                self.state.set_run(key.stem, run["id"])
                self.log(f"{key.stem}: following {run['id']} ({run['status']})")
            run = self.follow(run, key)
            # No capacity for the offer right now: nothing was rented, so wait and ask again.
            if run["status"] == "failed" and run.get("error_code") == "cloud_refused" and attempt < self.args.refused_retries:
                self.log(f"{key.stem}: refused ({run.get('error')}); trying again in {self.args.refused_wait_s:.0f} s")
                if self.stop.wait(self.args.refused_wait_s):
                    return "interrupted"
                run = None
                continue
            break
        if run["status"] != "completed" or not run.get("model_id"):
            self.log(f"{key.stem}: {run['status']} ({run.get('error_code')}: {run.get('error')})")
            return f"{run['status']} ({run.get('error_code')}: {run.get('error')})"
        rec = self.evaluate(key, run, spec["dataset_layout_sha256"])
        common.write_json_atomic(key.result_path(self.results), rec)
        m = rec["metrics"]
        self.log(f"{key.stem}: AP {m['ap']:.4f} AP50 {m['ap50']:.4f} (folo val {rec['tool_reported']['map50_95']:.4f})")
        return "completed"

    # ---------------------------------------------------------------------------------------------------------
    def owned_live_runs(self) -> dict[str, str]:
        """Runs on the server whose name carries this matrix's owner id and that have not ended: run id -> name."""
        owner = self.state.owner
        return {
            r["id"]: r["name"]
            for r in self.api.get("/api/runs")["runs"]
            if run_state.owned(r.get("name"), owner) and not ended(r)
        }

    def stop_active(self, wait_s: float = 600, settle_s: float = 2, poll_s: float = 3) -> list[str]:
        """Stop every run of this matrix still active (those this process follows, repeats it discarded, and any
        live run on the server whose name carries its owner id), wait for each to end (a cloud run: to release its
        instance), then terminate any live instance of those runs that is left. A failure on one run (no answer
        too) does not keep the others from being stopped, and the instances are reaped whatever happened before.
        Returns what could not be confirmed stopped ([] when everything was)."""
        time.sleep(settle_s)  # let a worker that is creating a run register it
        with self.lock:
            runs = dict(self.active)
            runs.update(self.discarded)
        problems: list[str] = []
        try:
            try:
                for rid, name in self.owned_live_runs().items():
                    runs.setdefault(rid, name)
            except Exception as e:
                self.log(f"could not list the server's runs to find this matrix's: {e}")
            unsent = set()
            for rid, stem in runs.items():
                if self.request_stop(rid, stem) is False:
                    unsent.add(rid)
            deadline = time.time() + wait_s
            pending = set(runs)
            while pending and time.time() < deadline:
                for rid in sorted(pending):
                    try:
                        run = self.api.get(f"/api/runs/{rid}")
                    except Exception:
                        continue
                    if ended(run):
                        pending.discard(rid)
                    elif rid in unsent and self.request_stop(rid, runs[rid]) is not False:
                        unsent.discard(rid)
                if pending:
                    time.sleep(poll_s)
            for rid in sorted(pending):
                why = "its stop request got no answer" if rid in unsent else f"not seen ended within {wait_s:.0f} s"
                problems.append(f"run {rid} ({runs[rid]}): {why}")
        finally:
            problems += self.reap(set(runs))
        return problems

    def request_stop(self, rid: str, stem: str) -> bool | None:
        """`POST /api/runs/{rid}/stop`: True when asked, None when the run had ended or is gone, False on failure."""
        try:
            self.api.post(f"/api/runs/{rid}/stop")
            self.log(f"{stem}: stop requested ({rid})")
            return True
        except ApiError as e:
            if e.status == 409:  # run_not_active: it ended meanwhile
                return None
            self.log(f"{stem}: stop failed ({rid}): {e}")
            return None if e.status == 404 else False
        except Exception as e:  # no answer: carry on with the other runs, and ask again while waiting
            self.log(f"{stem}: stop failed ({rid}): {e}")
            return False

    def reap(self, run_ids: set[str]) -> list[str]:
        """Terminate the live instances of `run_ids` (runs of this matrix), never anyone else's: another runner,
        Studio or a person may have cloud runs on the same server. Returns what could not be confirmed terminated."""
        if not run_ids:
            return []
        try:
            instances = self.api.get("/api/cloud/instances")["instances"]
        except ApiError as e:
            if e.status < 500:
                return []  # a server without the cloud feature
            return [f"could not list the cloud instances: {e}"]
        except Exception as e:
            return [f"could not list the cloud instances: {e}"]
        problems = []
        for inst in instances:
            if inst["status"] in ("terminated", "lost") or inst.get("run_id") not in run_ids:
                continue
            self.log(f"instance {inst['id']} ({inst['name']}) is {inst['status']}: terminating")
            what = f"instance {inst['id']} ({inst['name']}) of run {inst['run_id']}"
            try:
                got = self.api.call("DELETE", f"/api/cloud/instances/{inst['id']}")
            except Exception as e:
                problems.append(f"{what}: terminate failed: {e}")
                continue
            if (got or {}).get("status") != "terminated":
                problems.append(f"{what}: {(got or {}).get('status')}, not confirmed terminated")
        return problems

    def live_instances(self) -> list[dict]:
        try:
            return [i for i in self.api.get("/api/cloud/instances")["instances"] if i["status"] not in ("terminated", "lost")]
        except Exception:
            return []

    def unconfirmed(self, problems: list[str]) -> None:
        """The loud end of a clean-up that could not confirm everything stopped."""
        try:
            data_dir = self.api.get("/api/system").get("data_dir") or "<its data dir>"
        except Exception:
            data_dir = "<the server's data dir>"
        bar = "!" * 100
        self.log(bar)
        self.log(f"NOT CONFIRMED STOPPED ({len(problems)}): these may still be running and costing money")
        for p in problems:
            self.log(f"  {p}")
        self.log(f"finish by hand: folo cloud instances --data-dir {data_dir}")
        self.log(f"  then for each live instance of those runs: folo cloud terminate <instance id> --data-dir {data_dir}")
        self.log(f"  and for each run still active: curl -X POST {self.args.server}/api/runs/<run id>/stop")
        self.log(bar)


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
    ap.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="launch no run that the cloud budget would not cover (spent + active runs + this one, projected)",
    )
    ap.add_argument(
        "--budget-hard-usd", type=float, default=None, help="stop the active runs when the cloud cost passes this"
    )
    ap.add_argument("--run-cost-usd", type=float, default=0.15, help="projected cost of a run until one has finished")
    ap.add_argument("--refused-retries", type=int, default=40, help="launches refused for capacity are retried")
    ap.add_argument("--refused-wait-s", type=float, default=60)
    ap.add_argument(
        "--create-attempts",
        type=int,
        default=5,
        help="POST /api/runs attempts for a run whose creation got no clear answer (each looked up by name first)",
    )
    ap.add_argument("--folo", default="folo", help="the folo binary for val")
    ap.add_argument("--val-device", default="auto")
    ap.add_argument("--poll-s", type=float, default=10)
    args = ap.parse_args(argv)
    import os

    args.token = args.token or os.environ.get("FOLO_TOKEN")
    names = [s.name for s in datasets.pick(args.datasets)]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    # Variant by variant in the order given, seed-major: a matrix cut short (a spent budget) still has every dataset.
    keys = []
    for v in variants:
        for seed in common.parse_seeds(args.seeds if v == "pretrained" else args.scratch_seeds):
            for name in names:
                keys.append(common.RunKey("folo", args.arch, name, v, seed))
    if args.compute and args.device not in (None, "auto"):
        raise SystemExit("--device is for local runs; a cloud run trains on its pod's GPU (device auto)")
    if not args.compute and args.device is None:
        raise SystemExit("local runs need --device (cpu, gpu:<n> or auto); cloud runs need --compute")
    scorer.require()  # before anything is launched or rented: a run that cannot be scored is money lost
    m = Matrix(args)
    if "pretrained" in variants:
        m.base_model()  # fail before anything starts on a wrong --weights / --arch
    todo = [k for k in keys if not k.result_path(m.results).is_file()]
    # Runs already active on the server (a restart) are followed first, before anything new is launched.
    live = {r["id"] for r in m.api.get("/api/runs")["runs"] if r["status"] not in TERMINAL}
    todo.sort(key=lambda k: m.state.run_id(k.stem) not in live)
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
    problems: list[str] = []
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
        problems = m.stop_active()
        live = m.live_instances()
        if live:
            m.log(f"live cloud instances on the server: {[(i['id'], i['name'], i['status']) for i in live]}")
        if m.spent:
            m.log(f"cloud cost of the runs followed (estimate): ${m.spent_usd():.3f}")
    return finish(m, todo, outcome, problems)


def finish(m: Matrix, todo: list[common.RunKey], outcome: dict[str, str], problems: list[str]) -> int:
    """Each run's outcome, then the verdict: 0 only when every run of `todo` has its result file and clean-up
    confirmed everything stopped; else 1, with one line naming the runs without a result and why."""
    for stem, o in sorted(outcome.items()):
        m.log(f"{stem}: {o}")
    missing = {k.stem: outcome.get(k.stem, "not run") for k in todo if not k.result_path(m.results).is_file()}
    if missing:
        m.log(f"FAILED: {len(missing)} of {len(todo)} runs have no result: " + "; ".join(f"{s}: {o}" for s, o in missing.items()))
    if problems:
        m.unconfirmed(problems)
    return 1 if missing or problems else 0


if __name__ == "__main__":
    sys.exit(main())
