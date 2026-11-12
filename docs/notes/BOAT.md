# boat.dev sandboxes

For the maintainers' agents ([PROCESS](PROCESS.md)); it needs the maintainers' boat.dev account, and nothing in FOLO
depends on it.

boat.dev gives us on-demand **CPU-only** Ubuntu sandboxes (no GPU): `small` (2 vCPU / 4 GB), `default` (4 / 8),
`large` (8 vCPU / 16 GB). Preinstalled: python3.12, pip3, ffmpeg, docker, curl. EU datacentre IPs.

Good uses: Linux builds and test runs, long CPU benchmarks, dataset download / conversion / preprocessing.
Bad uses: training or GPU benchmarks (use the local RTX 4080).

- API: `https://boat.dev/api/v1`, `Authorization: Bearer <key>`. The key is in `~/.folo/boat.key` on the maintainers' machine.
  **Never print it, never commit it, never put it in a PR, log or task description.**
- Client: `tools/boat.py` (`from boat import Boat`): `create(type, name, ttl)`, `run(sid, cmd)` (sync, <= 600 s),
  `start(sid, cmd)` (detached) + `proc(sid, pid)`, `put` / `get` files, `stop`, `delete`.
- Rules: always pass `ttl` (auto-stop) and a `name` starting with `folo-`; at most 4 sandboxes at once; delete them
  when done.
