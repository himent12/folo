"""A fake FOLO server for the run_folo.py tests (test_run_folo*.py): the run and cloud endpoints it calls, in memory,
with the failures the tests need. No test in here."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import run_folo  # noqa: E402
import run_state  # noqa: E402
from folo_api import ApiError, TransportError  # noqa: E402

KEY = common.RunKey("folo", "folo-n", "bccd", "scratch", 0)
OTHER = common.RunKey("folo", "folo-n", "signature", "scratch", 0)
LAYOUT = "ab" * 32


class FakeServer:
    """The run and cloud endpoints run_folo.py calls, in memory. Failures:
    - `lose`: the next creations take effect, then raise TransportError (the answer lost);
    - `late`: the next creations raise TransportError before taking effect and commit later: at the next POST
      (`late_commits_on_post`) or at `commit_late()` (a request the server was still processing);
    - `fail_create_5xx`: the next creations answer 502 and create nothing;
    - `create_gate`: creations wait for this event (a POST that hangs) before taking effect;
    - `create_delay_s`: creations take this long (launches side by side);
    - `fail_stop`: stops of a run id raise TransportError that many times (-1: always);
    - `fail_get`: GET /api/runs/<id> of these ids raises TransportError;
    - `list_fails`: GET /api/runs raises TransportError."""

    def __init__(self):
        self.lock = threading.Lock()
        self.runs: list[dict] = []  # oldest first; listed newest first
        self.instances: list[dict] = []
        self.posts = 0
        self.lose = 0
        self.late = 0
        self.late_commits_on_post = True
        self.pending: list[dict] = []
        self.fail_create_5xx = 0
        self.create_gate: threading.Event | None = None
        self.fail_stop: dict[str, int] = {}
        self.fail_get: set[str] = set()
        self.list_fails = False
        self.create_delay_s = 0.0
        self.stops: list[str] = []
        self.deleted: list[str] = []

    def add_run(self, name, status="running", cloud=True, cost=None):
        with self.lock:
            rid = f"run_{len(self.runs) + 1}"
            run = {"id": rid, "name": name, "status": status, "config": {}, "epoch": 0, "epochs": 1}
            if cloud:
                run["cloud"] = {"phase": "training" if status not in run_folo.TERMINAL else "released"}
                if cost is not None:
                    run["cloud"]["cost_usd"] = cost
                self.instances.append({"id": f"ci_{rid}", "name": f"pod-{rid}", "run_id": rid, "status": "running"})
            self.runs.append(run)
            return run

    def commit_late(self, cost=None) -> list[dict]:
        with self.lock:
            pending, self.pending = self.pending, []
        return [self.add_run(b["name"], cost=cost) for b in pending]

    def run(self, rid):
        return next(r for r in self.runs if r["id"] == rid)

    def live(self, name) -> list[dict]:
        return [r for r in self.runs if r["name"] == name and not run_folo.ended(r)]

    def call(self, method, path, body=None, ok=(200, 201, 204), retries=5):
        if method == "POST" and path == "/api/runs":
            if self.late_commits_on_post:
                self.commit_late()
            if self.create_gate is not None:
                self.create_gate.wait(10)
            time.sleep(self.create_delay_s)
            with self.lock:
                self.posts += 1
                if self.fail_create_5xx:
                    self.fail_create_5xx -= 1
                    raise ApiError(502, "bad gateway")
                if self.late:
                    self.late -= 1
                    self.pending.append(body)
                    raise TransportError("POST /api/runs: no answer: read timed out")
                lose = self.lose > 0
                self.lose -= lose
            run = self.add_run(body["name"])
            if lose:
                raise TransportError("POST /api/runs: no answer: read timed out")
            return dict(run)
        if method == "POST" and path == "/api/datasets":
            return {"id": f"ds_{body['name']}", "classes": ["RBC", "WBC", "Platelets"] if "bccd" in body["name"] else ["signature"]}
        if method == "POST" and path.endswith("/stop"):
            rid = path.split("/")[3]
            with self.lock:
                left = self.fail_stop.get(rid, 0)
                if left:
                    self.fail_stop[rid] = left - 1 if left > 0 else left
                    raise TransportError(f"POST {path}: no answer: connection refused")
                run = self.run(rid)
                if run["status"] in run_folo.TERMINAL:
                    raise ApiError(409, '{"error":{"code":"run_not_active"}}')
                self.stops.append(rid)
                run["status"] = "stopped"
                if "cloud" in run:
                    run["cloud"]["phase"] = "released"
                return dict(run)
        if method == "GET" and path == "/api/runs":
            if self.list_fails:
                raise TransportError("GET /api/runs: no answer")
            with self.lock:
                return {"runs": [dict(r) for r in reversed(self.runs)]}
        if method == "GET" and path.startswith("/api/runs/"):
            rid = path.split("/")[3]
            if rid in self.fail_get:
                raise TransportError(f"GET {path}: no answer")
            return dict(self.run(rid))
        if method == "GET" and path == "/api/cloud/instances":
            return {"instances": [dict(i) for i in self.instances]}
        if method == "DELETE" and path.startswith("/api/cloud/instances/"):
            iid = path.rsplit("/", 1)[1]
            inst = next(i for i in self.instances if i["id"] == iid)
            inst["status"] = "terminated"
            self.deleted.append(iid)
            return dict(inst)
        if method == "GET" and path == "/api/system":
            return {"data_dir": "D:/fake-data"}
        raise AssertionError(f"unexpected {method} {path}")

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body)


def args(work: Path, **over) -> SimpleNamespace:
    a = dict(
        server="http://fake", token=None, results=str(work / "results"), work_dir=str(work), data_root=str(work / "data"),
        compute="cc_1:NVIDIA A40@secure", max_hours=0.5, disk_gb=None, arch="folo-n", weights="folo-n-coco-v2",
        device=None, epochs=100, imgsz=640, batch=16, budget_usd=None, budget_hard_usd=None, run_cost_usd=0.15,
        refused_retries=0, refused_wait_s=0, poll_s=0, create_attempts=3, folo="folo", val_device="cpu", parallel=1,
        reconcile_s=30, settle_s=5, stop_wait_s=1,
    )
    a.update(over)
    return SimpleNamespace(**a)


def identity(m: run_folo.Matrix, key=KEY):
    spec = m.run_spec(key)
    fp = run_state.fingerprint(spec)
    return run_state.run_name(key.stem, m.state.owner, fp), fp, spec


class Case(unittest.TestCase):
    """A temporary root and a fake server; `matrix()` makes a Matrix on a work dir under it that is released (its
    work-dir lock too) when the test ends."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)  # registered first: runs after every matrix is released
        self.root = Path(tmp.name)
        self.server = FakeServer()

    def matrix(self, work: Path | str = "a", server: FakeServer | None = None, **over) -> run_folo.Matrix:
        work = self.root / work
        m = run_folo.Matrix(args(work, **over), api=server or self.server)
        self.addCleanup(m.release)
        m.layouts = {"bccd": LAYOUT, "signature": LAYOUT}  # instead of hashing a real dataset
        m.pause = lambda s: None
        m.log = lambda msg: m.logged.append(msg)
        m.logged = []
        return m
