"""The reconciliation and clean-up half of run_folo.py's Matrix (a mixin; run_folo.py holds the other half): one run
per name while the matrix runs, and every run of the matrix stopped at its end, whatever fails on the way.

The server has no idempotency key for `POST /api/runs`, so a creation that timed out can still commit after the
runner looked it up and sent it again: two runs of one name, two pods. The exposure is bounded on the client:
- after every creation, also a successful one, the name is looked up again (more than one run: one is kept);
- every `--reconcile-s` while the matrix runs, the server's runs of the owner id are listed: a live run of a
  tracked name that is not the tracked run is a repeat; it is stopped, its instance reaped at the end, and its cost
  charged to the budget (`spent`);
- clean-up, after an error, Ctrl+C or a normal end, first lets no new creation start and lets the workers settle
  (`close`, `settle`), then lists the owner's runs (retried), stops every live one, and reaps their instances.
A repeat is looked for once its name's POST has been answered: a POST that commits later is found at the next
reconciliation after that (an interval, plus however long the listing and its retries take), and one that commits
after clean-up's last listing, also while clean-up still waits for other runs, is bounded only by the cloud run's
own `max_hours`. A repeat that had already ended when it was found is not stopped but still charged.

What clean-up cannot confirm (a creation whose worker never returned, a listing that failed, a run not seen ended,
an instance not seen terminated) is returned as a problem: run_folo.py prints them with the commands that finish
the job, and exits 1.
"""

from __future__ import annotations

import time
from concurrent.futures import wait
from pathlib import Path

import run_state
from folo_api import ApiError

TERMINAL = {"completed", "failed", "stopped"}


def ended(run: dict) -> bool:
    """The run is over and holds nothing: terminal, and a cloud run's instance released."""
    cloud = run.get("cloud")
    return run["status"] in TERMINAL and (not cloud or cloud.get("phase") == "released")


def cost_of(run: dict) -> float | None:
    c = (run.get("cloud") or {}).get("cost_usd")
    return None if c is None else float(c)


class Ownership:
    """Expects of the Matrix: api, args, state, lock, stop, log, pause, results, and the run books `active` (run id
    -> stem), `tracked` (name -> the run kept for it), `inflight` (name -> stem: a POST on its way), `discarded`
    (repeats: run id -> stem), `restop` (repeats whose stop failed), `spent`, `closing` and `unsettled`."""

    discovery_attempts = 6  # GET /api/runs in clean-up: about 30 s of backoff in all

    def close(self) -> None:
        """No creation starts from now on (checked under the lock, just before each POST); workers stop following."""
        with self.lock:
            self.closing = True
        self.stop.set()

    def settle(self, futures, timeout_s: float) -> bool:
        """Wait for the workers to return, at most `timeout_s` (a POST in flight ends within the API's request
        timeout); True when all did. Creations still in flight after that are reported by `clean_up` as unresolved,
        and the work dir stays locked until the process ends (`unsettled`): such a worker can still write the state.
        Futures the pool cancelled before they started never ran and are not waited for (`wait` would count them
        as pending for the whole timeout)."""
        done, running = wait([f for f in futures if not f.cancelled()], timeout=timeout_s)
        if running:
            self.log(f"{len(running)} worker(s) still busy after {timeout_s:.0f} s")
            self.unsettled = True
        return not running

    # ---------------------------------------------------------------------------------------------------------
    def discard(self, run: dict, stem: str) -> None:
        """Stop a repeat of one of this matrix's runs; its cost counts, clean-up checks it ended and reaps it."""
        self.log(f"{stem}: {run['id']} repeats {run['name']} ({run['status']}): stopping it")
        with self.lock:
            self.discarded[run["id"]] = stem
            if cost_of(run) is not None:
                self.spent[run["id"]] = cost_of(run)
        if run["status"] not in TERMINAL and self.request_stop(run["id"], stem) is False:
            with self.lock:
                self.restop.add(run["id"])

    def reconcile(self) -> list[str]:
        """List the server's runs of this owner id and stop the repeats: per name, every live run that is not the
        tracked one (a name with a POST in flight is left alone: its answer decides). A name this process does not
        track yet (a run of an earlier invocation, to be resumed) is only reconciled when it has more than one live
        run. Returns the repeats found this time."""
        try:
            runs = self.api.get("/api/runs")["runs"]
        except Exception as e:
            self.log(f"reconcile: could not list the server's runs: {e}")
            return []
        owner = self.state.owner
        owned = [r for r in runs if run_state.owned(r.get("name"), owner)]
        # Read the books after the listing: a run in it was created before, so its POST is either still in flight
        # (inflight) or answered and registered (tracked).
        with self.lock:
            tracked, inflight, known = dict(self.tracked), set(self.inflight), dict(self.discarded)
            for r in owned:
                if r["id"] in known and cost_of(r) is not None:
                    self.spent[r["id"]] = cost_of(r)
            restop = [r for r in owned if r["id"] in self.restop and not ended(r)]
        for r in restop:
            if self.request_stop(r["id"], known[r["id"]]) is not False:
                with self.lock:
                    self.restop.discard(r["id"])
        self.charge_ended_repeats(owned, tracked, inflight)
        live: dict[str, list[dict]] = {}
        for r in owned:
            if not ended(r) and r["id"] not in known:
                live.setdefault(r["name"], []).append(r)
        found = []
        for name, rs in live.items():
            if name in inflight:
                continue
            keep = tracked.get(name)
            if keep is None:
                if len(rs) < 2:
                    continue
                persisted = self.state.run_id(self.state.stem_of(name))
                keep = persisted if any(r["id"] == persisted for r in rs) else rs[-1]["id"]  # newest first
            for r in rs:
                if r["id"] != keep:
                    self.discard(r, self.state.stem_of(name))
                    found.append(r["id"])
        return found

    def charge_ended_repeats(self, owned: list[dict], tracked: dict, inflight: set) -> None:
        """A repeat that ended before it was found (a late commit that failed or finished on its own) needs no stop,
        but its pod was rented: every ended run of a tracked name that is not the tracked run counts in `spent`."""
        with self.lock:
            for r in owned:
                keep = tracked.get(r["name"])
                if keep is None or r["id"] == keep or r["name"] in inflight or not ended(r):
                    continue
                if cost_of(r) is not None:
                    self.spent[r["id"]] = cost_of(r)

    def maybe_reconcile(self) -> None:
        now = time.monotonic()
        if now - self.last_reconcile >= self.args.reconcile_s:
            self.last_reconcile = now
            self.reconcile()

    # ---------------------------------------------------------------------------------------------------------
    def discover(self) -> tuple[list[dict] | None, str | None]:
        """The server's runs (retried with backoff), or (None, why it failed)."""
        try:
            return self.api.call("GET", "/api/runs", retries=self.discovery_attempts)["runs"], None
        except Exception as e:
            return None, str(e)

    def clean_up(self, wait_s: float | None = None, poll_s: float = 3) -> list[str]:
        """After `close` and `settle`: stop every run of this matrix still active (those this process follows, the
        repeats it found, every live run on the server whose name carries its owner id), wait for each to end (a
        cloud run: to release its instance), then terminate any live instance of those runs that is left. A failure
        on one run does not keep the others from being stopped, and the instances are reaped whatever happened
        before. Returns what could not be confirmed ([] when everything was)."""
        wait_s = self.args.stop_wait_s if wait_s is None else wait_s
        with self.lock:
            runs = dict(self.active)
            runs.update(self.discarded)
            unresolved = dict(self.inflight)
        problems = [
            f"creation of {name} ({stem}) unresolved: its POST /api/runs had not returned when clean-up ran, so the "
            "run may exist; look for that name in the server's runs"
            for name, stem in unresolved.items()
        ]
        try:
            listing, why = self.discover()
            entries = {
                stem: e
                for stem, e in self.state.entries().items()
                if not (Path(self.results) / f"{stem}.json").is_file()  # scored: its run is over
            }
            if listing is None:
                problems.append(
                    f"could not list the server's runs ({why}): a run this matrix created without learning its id "
                    "cannot be found or stopped from here"
                )
                for stem, e in entries.items():
                    if e.get("run_id"):
                        runs.setdefault(e["run_id"], stem)  # tried one by one below
                    elif e["name"] not in unresolved:
                        problems.append(
                            f"creation intent {e['name']} ({stem}) unconfirmed: whether the server created it is not "
                            "known; look for that name in the server's runs"
                        )
            else:
                owner = self.state.owner
                owned = [r for r in listing if run_state.owned(r.get("name"), owner)]
                with self.lock:
                    tracked = dict(self.tracked)
                self.charge_ended_repeats(owned, tracked, set(unresolved))
                for r in owned:
                    if not ended(r):
                        runs.setdefault(r["id"], self.state.stem_of(r["name"]))
                # Every intent without a run id is now found (live: stopped below) or confirmed absent.
            unsent = {rid for rid, stem in runs.items() if self.request_stop(rid, stem) is False}
            deadline = time.time() + wait_s
            pending = set(runs)
            while pending:
                for rid in sorted(pending):
                    try:
                        run = self.api.get(f"/api/runs/{rid}")
                    except Exception:
                        continue
                    if cost_of(run) is not None:
                        with self.lock:
                            self.spent[rid] = cost_of(run)
                    if ended(run):
                        pending.discard(rid)
                    elif rid in unsent and self.request_stop(rid, runs[rid]) is not False:
                        unsent.discard(rid)
                if not pending or time.time() >= deadline:
                    break
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
        server = self.args.server
        bar = "!" * 100
        self.log(bar)
        self.log(f"NOT CONFIRMED STOPPED ({len(problems)}): these may still be running and costing money")
        for p in problems:
            self.log(f"  {p}")
        self.log(f"finish by hand: folo cloud instances --data-dir {data_dir}")
        self.log(f"  then for each live instance of those runs: folo cloud terminate <instance id> --data-dir {data_dir}")
        self.log(f"  a run by its name: curl -s {server}/api/runs, find the name (ft-...-{self.state.owner}-...), its id")
        self.log(f"  and for each run still active: curl -X POST {server}/api/runs/<run id>/stop")
        self.log(bar)
