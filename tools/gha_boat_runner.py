"""Provision a GitHub Actions self-hosted runner for FOLO on a boat.dev sandbox (see docs/CI.md).

    python tools/gha_boat_runner.py up <n>        # create sandbox folo-gha-linux-<n> and start runner folo-linux-boat-<n>
    python tools/gha_boat_runner.py renew <n>     # extend the sandbox TTL (resume with a new ttl)
    python tools/gha_boat_runner.py down <n>      # remove the runner registration and delete the sandbox

Needs `gh` (logged in, repo admin) and the boat key in ~/.folo/boat.key. Tokens are never printed.
Sandbox ids are stored in ~/.folo/gha_linux<n>_sid.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boat import Boat  # noqa: E402

REPO = "himent12/folo"
RUNNER_VERSION = "2.338.2"
TTL = 24 * 3600
SID_DIR = Path.home() / ".folo"


def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout.strip()


def sid_file(n):
    return SID_DIR / f"gha_linux{n}_sid"


def up(n):
    b = Boat()
    tok = gh("api", "-X", "POST", f"repos/{REPO}/actions/runners/registration-token", "-q", ".token")
    sid = b.create("large", name=f"folo-gha-linux-{n}", ttl=TTL)
    sid_file(n).write_text(sid)
    cmd = f"""set -e
mkdir -p ~/gha && cd ~/gha
curl -sSL -o r.tgz https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz
tar xzf r.tgz && rm r.tgz
./config.sh --url https://github.com/{REPO} --token {tok} --name folo-linux-boat-{n} --labels folo-linux --work _work --unattended --replace >/dev/null
"""
    r = b.run(sid, cmd, timeout=600)
    if r.get("exitCode") != 0:
        raise SystemExit("runner setup failed: " + (r.get("stderr") or "").replace(tok, "***")[-500:])
    b.start(sid, "cd ~/gha && nohup ./run.sh > runner.log 2>&1 &")
    print(f"runner folo-linux-boat-{n} started on sandbox {sid}")


def renew(n):
    b = Boat()
    sid = sid_file(n).read_text().strip()
    b.resume(sid, ttlSeconds=TTL)
    b.start(sid, "cd ~/gha && (pgrep -f Runner.Listener >/dev/null || nohup ./run.sh > runner.log 2>&1 &)")
    print(f"renewed {sid}")


def down(n):
    name = f"folo-linux-boat-{n}"
    runners = json.loads(gh("api", f"repos/{REPO}/actions/runners"))["runners"]
    for r in runners:
        if r["name"] == name:
            gh("api", "-X", "DELETE", f"repos/{REPO}/actions/runners/{r['id']}")
    f = sid_file(n)
    if f.exists():
        Boat().delete(f.read_text().strip())
        f.unlink()
    print(f"removed {name}")


if __name__ == "__main__":
    action, n = sys.argv[1], int(sys.argv[2])
    {"up": up, "renew": renew, "down": down}[action](n)
