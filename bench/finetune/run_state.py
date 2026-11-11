"""Which run on the server is which run of a matrix: owner ids, fingerprints, unique run names and state.json.

Run names on a FOLO server are not unique, so a name made of the run key alone (`ft-folo-n-bccd-pretrained-s0`) is
shared by every matrix with the same datasets and seeds. A matrix therefore has an **owner id**, made once for its
work dir and kept in `state.json` before anything is submitted, and each run a **fingerprint**: the sha256 of
everything that defines the run (the dataset and its layout digest, the architecture, the weights' name and
sha256, epochs, image size, batch, seed, variant, device and compute). A run's name carries both,
`ft-<stem>-<owner>-<fingerprint[:12]>`, so a run of that exact name is provably this matrix's run of this
configuration, and any live run whose name carries the owner id is this matrix's to stop.

`state.json` (version 2): `{"version": 2, "owner": "<10 hex>", "runs": {<stem>: {"name", "fingerprint", "config",
"run_id"}}}`. An entry is written before its run is submitted (`run_id` null until the server answered), so a
creation whose answer was lost can be looked up by name afterwards. A state.json of an older run_folo.py (stem ->
run id, no owner) is kept under `legacy`; its runs are not adopted.

One runner per work dir: `lock_work_dir` takes an OS lock on `<work dir>/run_folo.lock` before the state is read and
holds it until the runner has stopped its runs (the OS drops it when the process dies), so two runners can neither
overwrite each other's entries nor stop each other's runs as their own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from pathlib import Path

import common

VERSION = 2
LOCK_NAME = "run_folo.lock"


def lock_work_dir(work: Path):
    """Take the work dir's lock (an open file the caller keeps, and closes to let go), or exit naming the lock file
    when another runner holds it. msvcrt on Windows, flock elsewhere: held by the OS, released when the process
    ends however it ends."""
    path = Path(work) / LOCK_NAME
    f = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(
            f"{work} is in use by another run_folo.py (it holds {path}): one runner per work dir; wait for that one "
            "to end, or use another --work-dir"
        ) from None
    return f


def unlock_work_dir(f) -> None:
    if f.closed:
        return
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    f.close()


def fingerprint(spec: dict) -> str:
    """sha256 of the canonical JSON of a run's defining settings."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_name(stem: str, owner: str, fp: str) -> str:
    return f"ft-{stem}-{owner}-{fp[:12]}"


def owned(name: str, owner: str) -> bool:
    """`name` is a run name of the matrix with this owner id."""
    return re.fullmatch(rf"ft-.+-{re.escape(owner)}-[0-9a-f]{{12}}", name or "") is not None


def differences(old: dict, new: dict) -> dict:
    """The settings that differ between two run specs, as {key: [old, new]}."""
    return {k: [old.get(k), new.get(k)] for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}


class State:
    """work-dir/state.json (see the module doc). Thread-safe; every change is written at once (atomically)."""

    def __init__(self, path: Path, new_owner=lambda: secrets.token_hex(5)):
        self.path = path
        self.lock = threading.Lock()
        data = json.loads(path.read_text()) if path.is_file() else {}
        if data.get("version") != VERSION:
            legacy = {k: v for k, v in data.items() if isinstance(v, str)}
            data = {"version": VERSION, "owner": new_owner(), "runs": {}}
            if legacy:
                data["legacy"] = legacy
            common.write_json_atomic(path, data)  # the owner id is on disk before anything is submitted
        self.data = data

    @property
    def owner(self) -> str:
        return self.data["owner"]

    def entry(self, stem: str) -> dict | None:
        with self.lock:
            e = self.data["runs"].get(stem)
            return dict(e) if e else None

    def run_id(self, stem: str) -> str | None:
        return (self.entry(stem) or {}).get("run_id")

    def entries(self) -> dict[str, dict]:
        """stem -> a copy of its entry."""
        with self.lock:
            return {k: dict(v) for k, v in self.data["runs"].items()}

    def stem_of(self, name: str) -> str:
        """The stem whose entry has run name `name` (else the name itself)."""
        return next((k for k, v in self.entries().items() if v["name"] == name), name)

    def legacy_run(self, stem: str) -> str | None:
        """The run id an older run_folo.py recorded for `stem`."""
        return self.data.get("legacy", {}).get(stem)

    def set_intent(self, stem: str, name: str, fp: str, spec: dict) -> None:
        """Record the run about to be submitted (before the POST)."""
        with self.lock:
            self.data["runs"][stem] = {"name": name, "fingerprint": fp, "config": spec, "run_id": None}
            common.write_json_atomic(self.path, self.data)

    def set_run(self, stem: str, run_id: str) -> None:
        with self.lock:
            self.data["runs"][stem]["run_id"] = run_id
            common.write_json_atomic(self.path, self.data)
