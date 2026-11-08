"""Thin client for boat.dev CPU sandboxes (no GPUs there: use them for downloads, ffmpeg, dedupe, CPU sweeps).

    from boat import Boat; b = Boat()
    sid = b.create("xlarge", name="fetch1", ttl=6 * 3600)      # waits until ready
    b.run(sid, "nproc")                                         # sync, <= 600 s -> dict(exitCode, stdout, stderr)
    pid = b.start(sid, "long job > out.log 2>&1")               # detached; b.proc(sid, pid) polls
    b.put(sid, "local.py", "work/local.py"); b.get(sid, "work/out.tar", "local.tar")
    b.stop(sid); b.delete(sid)

CLI:  python tools/boat.py ls | run <sid> <cmd> | stop <sid> | rm <sid>
Key: ~/.folo/boat.key (outside the repo; never print it). Paths in the sandbox are relative to /home/user.
"""
import json
import sys
import time
from pathlib import Path

import requests

API = "https://boat.dev/api/v1"
KEY = (Path.home() / ".folo" / "boat.key").read_text().strip()


class Boat:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {KEY}"

    def _r(self, method, path, **kw):
        for i in range(6):
            r = self.s.request(method, API + path, timeout=kw.pop("timeout", 60) if i == 0 else 60, **kw)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 10)))
                continue
            break
        try:
            js = r.json()
        except ValueError:
            js = {"ok": False, "status": r.status_code, "message": r.text[:500]}
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {js.get('code')} {js.get('message')}")
        return js

    def ls(self):
        return self._r("GET", "/sandboxes?limit=200")["sandboxes"]

    def info(self, sid):
        js = self._r("GET", f"/sandboxes/{sid}")
        return js.get("sandbox", js)

    def wait(self, sid, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.info(sid).get("state")
            if st in ("ready", "idle", "running"):
                return st
            if st in ("error", "cancelled", "archived"):
                raise RuntimeError(f"sandbox {sid} state {st}")
            time.sleep(5)
        raise TimeoutError(sid)

    def create(self, type="default", name=None, ttl=3600, wait=True, **extra):
        body = {"type": type, "ttlSeconds": ttl, **extra}
        if name:
            body["name"] = name
        sid = self._r("POST", "/sandboxes", json=body)["sandbox"]["id"]
        if wait:
            self.wait(sid)
        return sid

    def resume(self, sid, **body):
        self._r("POST", f"/sandboxes/{sid}/resume", json=body)
        self.wait(sid)

    def run(self, sid, cmd, timeout=600, cwd=None):
        body = {"command": cmd, "timeoutSeconds": timeout}
        if cwd:
            body["cwd"] = cwd
        return self._r("POST", f"/sandboxes/{sid}/commands", json=body, timeout=timeout + 60)

    def start(self, sid, cmd, cwd=None):
        body = {"command": cmd, "detached": True}
        if cwd:
            body["cwd"] = cwd
        return self._r("POST", f"/sandboxes/{sid}/commands", json=body)["processId"]

    def proc(self, sid, pid):
        return self._r("GET", f"/sandboxes/{sid}/commands/{pid}")

    def put(self, sid, local, remote):
        u = self._r("POST", f"/sandboxes/{sid}/files/upload-url", json={"path": remote, "ttlSeconds": 3600})
        with open(local, "rb") as f:
            r = requests.request(u.get("method", "PUT"), u["url"], data=f, headers=u.get("headers") or {},
                                 timeout=3600)
        r.raise_for_status()

    def get(self, sid, remote, local):
        u = self._r("POST", f"/sandboxes/{sid}/files/download-url", json={"path": remote, "ttlSeconds": 3600})
        with requests.request(u.get("method", "GET"), u["url"], stream=True, timeout=3600) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    def stop(self, sid):
        return self._r("POST", f"/sandboxes/{sid}/stop", json={})

    def delete(self, sid):
        return self._r("DELETE", f"/sandboxes/{sid}", headers={"X-Ascii-Confirm-Delete": sid})


if __name__ == "__main__":
    b, a = Boat(), sys.argv[1:]
    if not a or a[0] == "ls":
        for s in b.ls():
            print(s.get("id"), s.get("name"), s.get("state"), s.get("type", ""), s.get("ip", ""))
    elif a[0] == "run":
        r = b.run(a[1], " ".join(a[2:]))
        print(r.get("stdout", ""), r.get("stderr", ""), "exit", r.get("exitCode"))
    elif a[0] == "stop":
        print(json.dumps(b.stop(a[1]))[:300])
    elif a[0] == "rm":
        print(json.dumps(b.delete(a[1]))[:300])
