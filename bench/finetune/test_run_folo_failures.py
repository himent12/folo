"""run_folo.py under partial failure, against the fake server of fake_folo.py: a creation that commits late, launches
in flight when the matrix shuts down, a server that cannot list its runs during clean-up, and two runners on one
work dir.

    python -m unittest discover -s bench/finetune -p "test_*.py"
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import run_folo  # noqa: E402
import run_state  # noqa: E402
from fake_folo import KEY, OTHER, Case, identity  # noqa: E402


def until(cond, timeout_s=5.0):
    """Wait (in the test, not the code under test) until `cond()` holds."""
    deadline = time.monotonic() + timeout_s
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.01)


class LateCommits(Case):
    """A: a POST that timed out but commits after the lookup made for it."""

    def test_a_retry_that_succeeds_after_a_late_commit_is_reconciled_at_once(self):
        m = self.matrix()
        self.server.late = 1  # the first POST times out; the server commits it just before the retry is handled
        name, fp, spec = identity(m)
        run = m.launch(KEY, name, fp, spec)
        late = next(r for r in self.server.runs if r["id"] != run["id"])
        self.assertEqual(self.server.posts, 2)
        self.assertEqual([r["id"] for r in self.server.live(name)], [run["id"]])
        self.assertEqual(self.server.stops, [late["id"]])
        self.assertEqual(m.discarded, {late["id"]: KEY.stem})
        self.assertEqual(m.active, {run["id"]: KEY.stem})

    def test_a_post_that_commits_after_every_lookup_is_stopped_by_the_next_reconciliation_and_charged(self):
        m = self.matrix(budget_usd=1.0)
        self.server.late = 1
        self.server.late_commits_on_post = False  # it commits later still: after the retry and its own lookup
        name, fp, spec = identity(m)
        run = m.launch(KEY, name, fp, spec)
        self.assertEqual(self.server.stops, [])
        (late,) = self.server.commit_late(cost=0.04)
        self.assertEqual(m.reconcile(), [late["id"]])
        self.assertEqual([r["id"] for r in self.server.live(name)], [run["id"]])
        self.assertEqual(self.server.stops, [late["id"]])
        self.assertAlmostEqual(m.spent_usd(), 0.04)  # charged to the budget, not ignored
        self.assertEqual(m.reconcile(), [])  # once
        m.close()
        self.assertEqual(m.clean_up(), [])
        self.assertIn(f"ci_{late['id']}", self.server.deleted)

    def test_reconciliation_leaves_creations_in_flight_and_runs_it_does_not_track_yet_alone(self):
        m = self.matrix()
        name, fp, spec = identity(m)
        other, *_ = identity(m, OTHER)
        resumable = self.server.add_run(other)  # a run of an earlier invocation, not followed yet: not a repeat
        self.server.add_run(name)
        self.server.add_run(name)
        m.inflight[name] = KEY.stem  # its POST is still in flight: what the worker gets back is not known yet
        self.assertEqual(m.reconcile(), [])
        self.assertEqual(self.server.stops, [])
        del m.inflight[name]
        self.assertEqual(len(m.reconcile()), 1)  # now two of one name, none tracked: the oldest is kept
        self.assertEqual(self.server.run(resumable["id"])["status"], "running")

    def test_a_late_repeat_of_a_run_that_already_ended_is_stopped(self):
        m = self.matrix()
        name, fp, spec = identity(m)
        run = m.launch(KEY, name, fp, spec)
        self.server.run(run["id"])["status"] = "completed"
        self.server.run(run["id"])["cloud"]["phase"] = "released"
        late = self.server.add_run(name)
        m.close()
        self.assertEqual(m.clean_up(), [])  # a normal exit reconciles too
        self.assertEqual(self.server.stops, [late["id"]])


class Shutdown(Case):
    """B: clean-up must not finish before a launch in flight has registered."""

    def setUp(self):
        super().setUp()
        self.gate = threading.Event()
        self.server.create_gate = self.gate
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.addCleanup(self.gate.set)

    def test_clean_up_waits_for_a_launch_in_flight_and_then_stops_it(self):
        m = self.matrix()
        fut = self.pool.submit(m.launch, KEY, *identity(m))
        until(lambda: m.inflight)  # the POST is on its way
        m.close()
        threading.Timer(0.3, self.gate.set).start()  # its answer comes after clean-up began
        m.settle([fut], timeout_s=5)
        self.assertEqual(m.clean_up(), [])
        (run,) = self.server.runs
        self.assertEqual(self.server.stops, [run["id"]])
        self.assertEqual(self.server.deleted, [f"ci_{run['id']}"])

    def test_a_launch_that_never_returns_is_reported_as_an_unresolved_creation(self):
        m = self.matrix()
        name, fp, spec = identity(m)
        fut = self.pool.submit(m.launch, KEY, name, fp, spec)
        until(lambda: m.inflight)
        m.close()
        m.settle([fut], timeout_s=0.2)  # the POST hangs past the bound
        problems = m.clean_up()
        self.assertTrue(any(name in p and "unresolved" in p for p in problems), problems)
        self.assertEqual(run_folo.finish(m, [], {}, problems), 1)
        self.assertIn(name, "\n".join(m.logged))

    def test_nothing_is_submitted_after_close(self):
        m = self.matrix()
        m.close()
        with self.assertRaises(KeyboardInterrupt):
            m.launch(KEY, *identity(m))
        self.assertEqual((self.server.posts, m.reserved, m.inflight), (0, {}, {}))


class Discovery(Case):
    """C: a clean-up that cannot list the server's runs has not confirmed anything."""

    def test_a_failed_listing_leaves_creation_intents_unconfirmed(self):
        m = self.matrix()
        name, fp, spec = identity(m)
        m.state.set_intent(KEY.stem, name, fp, spec)
        self.server.add_run(name)  # created, its answer and the lookup lost
        self.server.list_fails = True
        m.close()
        problems = m.clean_up()
        self.assertTrue(any("could not list the server's runs" in p for p in problems), problems)
        self.assertTrue(any(name in p and "unconfirmed" in p for p in problems), problems)
        self.assertEqual(run_folo.finish(m, [], {}, problems), 1)
        text = "\n".join(m.logged)
        self.assertIn("folo cloud instances --data-dir", text)
        self.assertIn(name, text)

    def test_persisted_run_ids_are_stopped_one_by_one_when_the_listing_fails(self):
        m = self.matrix()
        name, fp, spec = identity(m)
        m.state.set_intent(KEY.stem, name, fp, spec)
        run = self.server.add_run(name)
        m.state.set_run(KEY.stem, run["id"])
        m.release()
        m = self.matrix()  # a restart that never followed it
        self.server.list_fails = True
        m.close()
        problems = m.clean_up()
        self.assertEqual(self.server.stops, [run["id"]])
        self.assertIn(f"ci_{run['id']}", self.server.deleted)
        self.assertTrue(any("could not list the server's runs" in p for p in problems), problems)

    def test_an_intent_the_listing_does_not_show_is_confirmed_absent(self):
        m = self.matrix()
        name, fp, spec = identity(m)
        m.state.set_intent(KEY.stem, name, fp, spec)  # the POST never reached the server
        m.close()
        self.assertEqual(m.clean_up(), [])


LOCKER = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_state
handle = run_state.lock_work_dir(Path(sys.argv[2]))
print("locked", flush=True)
sys.stdin.read()
"""


class WorkDirLock(Case):
    """D: one runner per work dir, held by the OS."""

    def test_a_second_runner_on_the_same_work_dir_is_refused(self):
        a = self.matrix("a")
        with self.assertRaises(SystemExit) as e:
            self.matrix("a")
        self.assertIn(run_state.LOCK_NAME, str(e.exception))
        a.release()
        self.matrix("a")  # free again

    def test_a_runner_in_another_process_holds_the_work_dir_until_it_ends(self):
        work = self.root / "a"
        work.mkdir()
        for end in ("exit", "kill"):
            child = subprocess.Popen(
                [sys.executable, "-c", LOCKER, str(HERE), str(work)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(child.stdout.readline().strip(), "locked")
            with self.assertRaises(SystemExit):
                run_state.lock_work_dir(work)
            if end == "exit":
                child.stdin.close()
            else:
                child.kill()  # the OS releases a dead process's lock: no stale-PID logic
            child.wait(10)
            child.stdout.close()
            if child.stdin and not child.stdin.closed:
                child.stdin.close()
            deadline = time.monotonic() + 5
            while True:
                try:
                    run_state.lock_work_dir(work).close()
                    break
                except SystemExit:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)

    def test_json_files_are_written_through_unique_temporary_files(self):
        path = self.root / "state.json"
        seen = []
        real = common.os.replace

        def spy(src, dst):
            seen.append(Path(src).name)
            real(src, dst)

        common.os.replace = spy
        try:
            common.write_json_atomic(path, {"a": 1})
            common.write_json_atomic(path, {"a": 2})
        finally:
            common.os.replace = real
        self.assertEqual(len(set(seen)), 2)
        self.assertNotIn("state.json.tmp", seen)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["state.json"])
        self.assertEqual(path.read_text(), '{\n  "a": 2\n}\n')


if __name__ == "__main__":
    unittest.main()
