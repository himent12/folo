"""Provision GitHub Actions self-hosted runners for FOLO on boat.dev sandboxes (see docs/CI.md).

    python tools/gha_boat_runner.py up <n>        # create sandbox folo-gha-linux-<n>, register and start its runner
    python tools/gha_boat_runner.py renew <n>     # extend the sandbox TTL and make sure the runner is running
    python tools/gha_boat_runner.py down <n>      # remove the runner registration and delete the sandbox

Runner names: n = 1 is the original `folo-linux-boat`, n >= 2 is `folo-linux-boat-<n>`. Sandbox ids are kept in
~/.folo/gha_linux<n>_sid. Needs `gh` (logged in, repo admin) and the boat key in ~/.folo/boat.key.
Registration tokens are fetched at runtime and never printed, including in error messages.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boat import Boat  # noqa: E402

REPO = "himent12/folo"
RUNNER_VERSION = "2.338.2"
TTL = 24 * 3600
SID_DIR = Path.home() / ".folo"
# The brackets stop pgrep from matching the shell command line that contains this very pattern.
LISTENER = "pgrep -f '[R]unner[.]Listener' >/dev/null"


def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout.strip()


def runner_name(n):
    return "folo-linux-boat" if n == 1 else f"folo-linux-boat-{n}"


def sid_file(n):
    return SID_DIR / f"gha_linux{n}_sid"


def redact(text, secret):
    return str(text).replace(secret, "***") if secret else str(text)


def wait_online(name, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        runners = json.loads(gh("api", f"repos/{REPO}/actions/runners"))["runners"]
        if any(r["name"] == name and r["status"] == "online" for r in runners):
            return True
        time.sleep(5)
    return False


def start_listener(b, sid):
    # Detached, so the listener outlives the command; then confirm the process exists.
    b.start(sid, "./run.sh > runner.log 2>&1", cwd="gha")
    for _ in range(12):
        if b.run(sid, LISTENER, timeout=30).get("exitCode") == 0:
            return True
        time.sleep(5)
    return False


def up(n):
    f, name = sid_file(n), runner_name(n)
    if f.exists():
        raise SystemExit(f"{f} already exists (sandbox {f.read_text().strip()}); use renew, or down first")
    b = Boat()
    tok = gh("api", "-X", "POST", f"repos/{REPO}/actions/runners/registration-token", "-q", ".token")
    sid = b.create("large", name=f"folo-gha-linux-{n}", ttl=TTL)
    f.write_text(sid)  # saved before setup, so a failed setup can still be cleaned up with `down`
    cmd = f"""set -e
mkdir -p ~/gha && cd ~/gha
curl -sSL -o r.tgz https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz
tar xzf r.tgz && rm r.tgz
./config.sh --url https://github.com/{REPO} --token {tok} --name {name} --labels folo-linux --work _work --unattended --replace >/dev/null
"""
    try:
        r = b.run(sid, cmd, timeout=600)
    except Exception as e:  # noqa: BLE001 - re-raised without the token
        raise SystemExit("runner setup failed: " + redact(e, tok)) from None
    if r.get("exitCode") != 0:
        raise SystemExit("runner setup failed: " + redact(r.get("stderr") or "", tok)[-500:])
    if not start_listener(b, sid) or not wait_online(name):
        raise SystemExit(f"runner {name} did not come online (sandbox {sid}); see ~/gha/runner.log there")
    print(f"runner {name} online on sandbox {sid}")


def renew(n):
    b, name = Boat(), runner_name(n)
    sid = sid_file(n).read_text().strip()
    state = b.info(sid).get("state")
    if state in ("ready", "idle", "running"):
        b._r("PATCH", f"/sandboxes/{sid}", json={"ttlSeconds": TTL})
    else:
        b.resume(sid, ttlSeconds=TTL)
    if b.run(sid, LISTENER, timeout=30).get("exitCode") != 0 and not start_listener(b, sid):
        raise SystemExit(f"runner listener on {sid} did not start")
    if not wait_online(name):
        raise SystemExit(f"runner {name} is not online")
    print(f"renewed {sid}; runner {name} online")


def down(n):
    name, f = runner_name(n), sid_file(n)
    runners = json.loads(gh("api", f"repos/{REPO}/actions/runners"))["runners"]
    for r in runners:
        if r["name"] == name:
            gh("api", "-X", "DELETE", f"repos/{REPO}/actions/runners/{r['id']}")
    if f.exists():
        sid = f.read_text().strip()
        try:
            Boat().delete(sid)
        except RuntimeError as e:
            if "404" not in str(e):  # already gone counts as deleted; anything else keeps the state file
                raise
        f.unlink()
    print(f"removed {name}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("up", "renew", "down"):
        raise SystemExit(__doc__)
    {"up": up, "renew": renew, "down": down}[sys.argv[1]](int(sys.argv[2]))
