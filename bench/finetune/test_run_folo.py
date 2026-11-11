"""run_folo.py against a fake FOLO server (fake_folo.py): who owns which run, creation without blind repeats, the
budget of launches side by side, and a clean-up that survives a dead connection. test_run_folo_failures.py has the
partial failures: late commits, launches in flight at shutdown, a server that cannot list, a shared work dir.

    python -m unittest discover -s bench/finetune -p "test_*.py"

No server, network, numpy or pycocotools needed.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import folo_api  # noqa: E402
import run_folo  # noqa: E402
import run_state  # noqa: E402
from fake_folo import KEY, OTHER, Case, identity  # noqa: E402
from folo_api import ApiError, TransportError  # noqa: E402


class Owners(Case):
    def test_the_owner_id_is_on_disk_before_anything_is_submitted(self):
        m = self.matrix("a")
        state = json.loads((self.root / "a" / "state.json").read_text())
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["owner"], m.state.owner)
        self.assertEqual(self.server.posts, 0)
        name, fp, _ = identity(m)
        m.release()
        self.assertEqual(self.matrix("a").state.owner, m.state.owner)  # kept across restarts
        self.assertTrue(name.startswith(f"ft-{KEY.stem}-{m.state.owner}-{fp[:12]}"))
        self.assertTrue(run_state.owned(name, m.state.owner))
        self.assertFalse(run_state.owned(name, "0123456789"))

    def test_two_matrices_with_the_same_datasets_and_seeds_do_not_adopt_each_others_runs(self):
        a, b = self.matrix("a"), self.matrix("b")
        name_a, fp_a, spec_a = identity(a)
        run_a = a.launch(KEY, name_a, fp_a, spec_a)
        name_b, fp_b, spec_b = identity(b)
        self.assertEqual(fp_a, fp_b)  # the same configuration...
        self.assertNotEqual(name_a, name_b)  # ...but another matrix
        self.assertIsNone(b.resume(KEY, name_b, fp_b, spec_b))  # b starts its own
        run_b = b.launch(KEY, name_b, fp_b, spec_b)
        self.assertNotEqual(run_a["id"], run_b["id"])
        # A restart of a follows its own run, not b's (the newer one of the same key).
        a.release()
        a2 = self.matrix("a")
        self.assertEqual(a2.resume(KEY, name_a, fp_a, spec_a)["id"], run_a["id"])
        # b's clean-up stops its own run only.
        b.close()
        self.assertEqual(b.clean_up(), [])
        self.assertEqual(self.server.stops, [run_b["id"]])
        self.assertEqual(self.server.deleted, [f"ci_{run_b['id']}"])
        self.assertEqual(self.server.run(run_a["id"])["status"], "running")

    def test_a_run_of_another_configuration_is_refused_not_adopted(self):
        m = self.matrix("a")
        name, fp, spec = identity(m)
        run = m.launch(KEY, name, fp, spec)
        m.release()
        m2 = self.matrix("a", epochs=50)  # the same work dir, other settings
        name2, fp2, spec2 = identity(m2)
        self.assertNotEqual(fp2, fp)
        with self.assertRaises(RuntimeError) as e:
            m2.resume(KEY, name2, fp2, spec2)
        msg = str(e.exception)
        self.assertIn(run["id"], msg)
        self.assertIn(name, msg)
        self.assertIn(name2, msg)
        self.assertIn("'epochs': [100, 50]", msg)
        m2.release()
        # Other weights are another configuration too (the scored run must be the requested weights' run).
        m3 = self.matrix("a")
        m3.base = {"id": "m_9", "sha256": "cd" * 32}
        pre = common.RunKey("folo", "folo-n", "bccd", "pretrained", 0)
        n3, f3, s3 = identity(m3, pre)
        m3.launch(pre, n3, f3, s3)
        m3.release()
        m4 = self.matrix("a")
        m4.base = {"id": "m_10", "sha256": "ef" * 32}
        with self.assertRaises(RuntimeError):
            m4.resume(pre, *identity(m4, pre))
        # Once the other configuration's run has ended without a result, the new one may start.
        m4.api.post(f"/api/runs/{run['id']}/stop")
        self.assertIsNone(m4.resume(KEY, name2, fp2, spec2))

    def test_a_state_file_of_the_older_runner_is_not_adopted(self):
        (self.root / "a").mkdir()
        (self.root / "a" / "state.json").write_text(json.dumps({KEY.stem: "run_old"}))
        m = self.matrix("a")
        self.assertEqual(json.loads((self.root / "a" / "state.json").read_text())["legacy"], {KEY.stem: "run_old"})
        with self.assertRaises(RuntimeError) as e:
            m.resume(KEY, *identity(m))
        self.assertIn("run_old", str(e.exception))
        self.assertIsNone(m.resume(OTHER, *identity(m, OTHER)))


class Creation(Case):
    def setUp(self):
        super().setUp()
        self.m = self.matrix("a")

    def test_a_lost_answer_is_looked_up_by_name_not_posted_again(self):
        self.server.lose = 1
        name, fp, spec = identity(self.m)
        run = self.m.launch(KEY, name, fp, spec)
        self.assertEqual(self.server.posts, 1)
        self.assertEqual([r["id"] for r in self.server.runs], [run["id"]])
        self.assertEqual(self.m.state.run_id(KEY.stem), run["id"])
        self.assertEqual(self.m.active, {run["id"]: KEY.stem})

    def test_the_intent_is_in_state_before_the_post(self):
        name, fp, spec = identity(self.m)
        seen = []
        real = self.server.call

        def spy(method, path, body=None, **kw):
            if path == "/api/runs" and method == "POST":
                seen.append(json.loads((self.root / "a" / "state.json").read_text())["runs"][KEY.stem])
            return real(method, path, body, **kw)

        self.server.call = spy
        self.m.launch(KEY, name, fp, spec)
        self.assertEqual(seen, [{"name": name, "fingerprint": fp, "config": spec, "run_id": None}])

    def test_a_5xx_that_created_nothing_is_sent_again_once(self):
        self.server.fail_create_5xx = 1
        name, fp, spec = identity(self.m)
        run = self.m.launch(KEY, name, fp, spec)
        self.assertEqual(self.server.posts, 2)
        self.assertEqual([r["id"] for r in self.server.runs], [run["id"]])

    def test_a_refusal_is_not_retried(self):
        real = self.server.call

        def refuse(method, path, body=None, **kw):
            if path == "/api/runs" and method == "POST":
                raise ApiError(400, '{"error":{"code":"invalid_config"}}')
            return real(method, path, body, **kw)

        self.server.call = refuse
        with self.assertRaises(ApiError):
            self.m.launch(KEY, *identity(self.m))
        self.assertEqual((self.m.reserved, self.m.inflight), ({}, {}))

    def test_repeats_of_one_name_keep_one_and_stop_the_others(self):
        name, fp, spec = identity(self.m)
        first = self.server.add_run(name)
        second = self.server.add_run(name)
        self.server.add_run(name, status="failed")  # a refused launch: ignored
        self.m.state.set_intent(KEY.stem, name, fp, spec)
        kept = self.m.resume(KEY, name, fp, spec)
        self.assertEqual(kept["id"], first["id"])  # the oldest
        self.assertEqual(self.server.stops, [second["id"]])
        self.assertEqual(self.m.discarded, {second["id"]: KEY.stem})
        # Clean-up confirms the repeat ended and reaps its instance.
        self.m.close()
        self.assertEqual(self.m.clean_up(), [])
        self.assertIn(f"ci_{second['id']}", self.server.deleted)

    def test_the_api_sends_a_creation_once(self):
        class Session:
            def __init__(self, answers):
                self.answers, self.n = list(answers), 0

            def request(self, *a, **kw):
                self.n += 1
                x = self.answers.pop(0)
                if isinstance(x, Exception):
                    raise x
                return x

        def answer(status):
            return SimpleNamespace(status_code=status, content=b"{}", text="{}", json=lambda: {})

        api = folo_api.Api.__new__(folo_api.Api)
        api.base, api.timeout_s, api.sleep = "http://x", 1, lambda s: None
        api.s = Session([OSError("reset")] * 5)
        with self.assertRaises(TransportError):
            api.call("POST", "/api/runs", {}, retries=1)
        self.assertEqual(api.s.n, 1)
        api.s = Session([answer(503)] * 3)
        with self.assertRaises(ApiError) as e:
            api.call("GET", "/api/runs", retries=3)
        self.assertEqual((api.s.n, e.exception.status), (3, 503))
        self.assertTrue(folo_api.ambiguous(e.exception))
        api.s = Session([OSError("reset"), answer(200)])
        self.assertEqual(api.call("GET", "/api/runs"), {})
        self.assertFalse(folo_api.ambiguous(ApiError(409, "")))


class Budget(Case):
    def test_a_reservation_holds_its_share_until_it_is_a_run_or_released(self):
        m = self.matrix("a", budget_usd=0.15, run_cost_usd=0.15)
        self.assertTrue(m.reserve(KEY.stem))
        self.assertFalse(m.reserve(OTHER.stem))  # the first launch is not registered yet: its share is reserved
        m.register("run_x", KEY.stem, "ft-x")
        self.assertEqual(m.reserved, {})
        self.assertFalse(m.reserve(OTHER.stem))  # now it is an active run
        m.active.clear()
        self.assertTrue(m.reserve(OTHER.stem))
        m.release_reservation(OTHER.stem)
        self.assertEqual(m.reserved, {})

    def test_two_launches_side_by_side_cannot_both_pass_a_budget_for_one(self):
        m = self.matrix("a", budget_usd=0.15, run_cost_usd=0.15)
        self.server.create_delay_s = 0.2  # both workers check the budget while the first creation is in flight
        out = {}
        go = threading.Barrier(2)

        def worker(key):
            go.wait()
            out[key.stem] = m.launch(key, *identity(m, key))

        threads = [threading.Thread(target=worker, args=(k,)) for k in (KEY, OTHER)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.server.posts, 1)
        self.assertEqual(sorted(v is None for v in out.values()), [False, True])


class Cleanup(Case):
    def setUp(self):
        super().setUp()
        self.m = self.matrix("a")

    def two_runs(self):
        r1 = self.m.launch(KEY, *identity(self.m))
        r2 = self.m.launch(OTHER, *identity(self.m, OTHER))
        return r1["id"], r2["id"]

    def test_no_answer_on_the_first_stop_still_stops_the_others_and_reaps(self):
        r1, r2 = self.two_runs()
        self.server.fail_stop[r1] = 1  # the first stop request gets no answer; the second one does
        self.m.close()
        self.assertEqual(self.m.clean_up(), [])
        self.assertEqual(sorted(self.server.stops), sorted([r1, r2]))
        self.assertEqual(sorted(self.server.deleted), sorted([f"ci_{r1}", f"ci_{r2}"]))

    def test_what_cannot_be_confirmed_is_listed_and_exits_non_zero(self):
        r1, r2 = self.two_runs()
        self.server.fail_stop[r1] = -1  # the server never answers for r1
        self.server.fail_get.add(r1)
        self.m.close()
        problems = self.m.clean_up(wait_s=0.2, poll_s=0.05)
        self.assertEqual(self.server.stops, [r2])
        self.assertIn(f"ci_{r2}", self.server.deleted)  # reaped all the same
        self.assertIn(f"ci_{r1}", self.server.deleted)  # its instance too: the instance list still answered
        self.assertEqual(len(problems), 1)
        self.assertIn(r1, problems[0])
        self.assertIn("no answer", problems[0])
        code = run_folo.finish(self.m, [], {}, problems)
        self.assertEqual(code, 1)
        text = "\n".join(self.m.logged)
        self.assertIn("NOT CONFIRMED STOPPED", text)
        self.assertIn("folo cloud instances --data-dir D:/fake-data", text)
        self.assertIn("folo cloud terminate <instance id>", text)

    def test_runs_of_this_owner_on_the_server_are_stopped_even_if_this_process_lost_them(self):
        name, fp, spec = identity(self.m)
        self.m.state.set_intent(KEY.stem, name, fp, spec)
        lost = self.server.add_run(name)  # created, but its answer never came and the process died
        self.m.release()
        mine = self.matrix("a")  # a restart
        mine.close()
        self.assertEqual(mine.clean_up(), [])
        self.assertEqual(self.server.stops, [lost["id"]])

    def test_an_unreachable_instance_list_is_a_problem(self):
        self.two_runs()
        real = self.server.call

        def no_instances(method, path, body=None, **kw):
            if path == "/api/cloud/instances":
                raise TransportError("no answer")
            return real(method, path, body, **kw)

        self.server.call = no_instances
        self.m.close()
        problems = self.m.clean_up()
        self.assertEqual(len(problems), 1)
        self.assertIn("could not list the cloud instances", problems[0])


class Finish(Case):
    def test_a_run_without_a_result_file_is_a_failure(self):
        m = self.matrix("a")
        outcome = {KEY.stem: "error: No module named 'numpy'", OTHER.stem: "completed"}
        common.write_json_atomic(OTHER.result_path(m.results), {"version": 1})
        self.assertEqual(run_folo.finish(m, [KEY, OTHER], outcome, []), 1)
        line = next(x for x in m.logged if x.startswith("FAILED"))
        self.assertIn("1 of 2 runs have no result", line)
        self.assertIn(f"{KEY.stem}: error: No module named 'numpy'", line)
        common.write_json_atomic(KEY.result_path(m.results), {"version": 1})
        self.assertEqual(run_folo.finish(m, [KEY, OTHER], outcome, []), 0)


if __name__ == "__main__":
    unittest.main()
