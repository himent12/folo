# Cloud compute

FOLO can train a run on a GPU that is not in this machine: one rented from a cloud provider for the length of the
run, or another machine of yours that runs `folo serve`. The run stays a normal run of this server: the same
`Run`, events, metrics, logs, samples and models, in Studio, Desktop, the API and MCP. Only where the trainer
executes changes.

```
   this machine (control plane)                                 the instance (worker)
 ┌──────────────────────────────┐   provider API (HTTPS)   ┌───────────────────────────────┐
 │ folo serve                   │ ───── launch / poll ───▶ │ provider: RunPod, ...         │
 │  cloud run executor          │ ◀──── terminate ──────── │                               │
 │   ├ dataset → tar → uploads  │                          │  folo serve --token …         │
 │   ├ POST /api/runs           │ ═══ docs/API.md over ══▶ │   uploads, runs, metrics,     │
 │   ├ poll metrics / logs      │     HTTPS + bearer token │   logs, samples, weights      │
 │   └ weights → runs/<id>/     │ ◀══════════════════════  │                               │
 └──────────────────────────────┘                          └───────────────────────────────┘
```

The worker is an ordinary `folo serve`: the control plane talks to it with the public API of docs/API.md and nothing
else. Everything a worker needs beyond a local server is three small API features, useful on their own: a token
(`folo serve --token`), uploads (`/api/uploads`) and cursors on a run's history. So any reachable FOLO server can
be a worker, and a provider only has to start one.

## Concepts
- **Provider**: a kind of compute FOLO knows how to use (`GET /api/cloud/providers`). `runpod` rents GPU pods from
  [RunPod](https://www.runpod.io); `remote` uses a `folo serve` that already runs somewhere (a workstation, a VM you
  started yourself at any cloud). A provider that *provisions* launches an instance per run and terminates it
  afterwards; `remote` does not.
- **Connection**: a provider plus your credentials for it (`/api/cloud/connections`): an API key for RunPod, a URL
  and token for a remote server. Credentials are checked when the connection is made, kept in the data dir
  (`cloud/connections.json`, readable by the owner only) and never returned by the API, logged or sent anywhere but
  the provider.
- **Offer**: something a connection can run on right now (`…/connections/{id}/offers`): a GPU type with its memory,
  price per hour and availability. A remote server has one offer, its own GPU, at no price.
- **Instance**: a machine FOLO launched (`/api/cloud/instances`), with what it costs so far. FOLO terminates every
  instance it launches; the list is where you check, and where you terminate one by hand.
- **Cloud run**: a run whose `TrainConfig.compute` names a connection and an offer. `Run.cloud` says where it is in
  its lifecycle.

## A cloud run, step by step
`POST /api/runs` with `compute` answers at once with the run (`queued`); a thread of its own drives it from there
(cloud runs do not wait in the local training queue and do not use the local GPU, so several can run side by side
and next to a local run). `TrainConfig.device` must be omitted or `"auto"` (a cloud run always trains on the device
its worker selects; any other value is `400 invalid_config`). `compute.max_hours` defaults to `12` (valid range
`0.25..=720`, i.e. 15 minutes to 30 days) and `compute.disk_gb` is optional (`10..=4096` GB; when omitted, sized to
twice the packed dataset size plus 20 GB, capped at 4096 GB). `Run.cloud.phase` and the SSE `cloud` event follow the
lifecycle, and `Run.cloud.ended_in` records the phase the run was in when it entered `releasing` (so a failed or
stopped run shows exactly where it stopped):

| phase | what happens | `Run.status` |
|---|---|---|
| `provisioning` | the provider is asked for an instance of the offer (a `remote` connection skips this) | `queued` |
| `booting` | wait for the instance, then for its worker to answer; check it (below) | `queued` |
| `syncing` | upload the dataset (and the weights or checkpoint to start from) | `queued` |
| `training` | the worker trains; metrics, logs and samples are mirrored; checkpoints are copied back | the worker's: `queued` while it waits there, then `running` |
| `collecting` | the run ended on the worker: copy `best.folo` and `last.folo` | unchanged (`running` / `stopping`) |
| `releasing` | terminate the instance | unchanged |
| `released` | done; the final status is published | `completed` / `stopped` / `failed` |

- **Worker check (booting).** The worker must answer `/api/system` with the token and list the features
  `uploads`, `run_cursors` and `run_checkpoint` there; a provisioned instance must also resolve `device: auto` to
  a GPU (a rented GPU that FOLO cannot use would train on the CPU at GPU prices; a `remote` server is used as it
  is), and a run with `amp` needs `capabilities.amp.available`. Otherwise the run fails with `cloud_boot_failed`
  before anything is uploaded, and the message says which check failed (a worker that is too old, an instance
  without a usable GPU adapter with the adapters it reported, mixed precision without tensor cores). A
  provisioned instance gets 15 minutes to pass the check; a `remote` server one minute to answer. While the worker
  does not answer yet, the provider is asked about the instance every few seconds: one that is gone (its setup
  failed and it terminated itself) fails the run at once, with where it installs FOLO from, instead of at the end
  of the 15 minutes.
- **Dataset and starting weights.** The dataset is sent as FOLO reads it, not as it lies on disk: a tar stream in the
  canonical YOLO layout (`data.yaml`, `images/<split>/<n>.<ext>`, `labels/<split>/<n>.txt`, files numbered in dataset
  order; images byte for byte, labels written from the parsed boxes), produced on the fly by `folo_data::pack` and
  cut into upload parts. So datasets of any format and layout travel the same way, images the loader skips stay home,
  and image ids are the same on both sides (validation samples line up). Both the dataset upload and any starting
  weights or resume checkpoint upload (`weights/last.folo`) carry an upload fingerprint: a worker that already has
  that dataset or weights file (such as a `remote` server on later runs) answers with the existing resource and
  nothing is sent.
- **Mirroring.** While the worker trains, the control plane polls it (about once a second; backing off exponentially
  up to 30 s while it does not answer, which the run's log says once, and again when it answers): the run, new steps
  and epochs (`/metrics?after_step=&after_epoch=`), new log lines (`/logs?after=`, up to 5,000 lines per page) and
  each new epoch's samples. While the worker does not answer, `Run.cloud.unanswered_since` says since when (Studio:
  "Worker not answering"), so a page whose numbers stand still says why. They go through the same sink as a local
  trainer's events, so `metrics.jsonl`, `log.jsonl`, `samples/`, SSE and `Run` behave as for a local run (subscribers
  get every epoch and at most the newest step of each poll). Polling, not SSE, because provider proxies cut long
  connections; every request is short (10 s connect, 90 s request timeout) and retried.
- **Checkpoints.** `weights/last.folo` is copied to the run folder after an epoch when the last copy is older than
  10 minutes (`Run.cloud.synced` is when), and both `last.folo` and `best.folo` at the end. While the run trains, the
  copy runs on its own thread (`folo-cloud-copy-<id>`), so the live view, a stop and the collection at the end never
  wait for it (a large checkpoint takes minutes on a slow link). It fetches into a staging file that only the run's
  own thread moves into place, so a copy that outlives its run cannot put an old checkpoint over a newer one. One
  that fails is tried again at least a minute later, not at every epoch.
- **Stopping.** `POST /api/runs/{id}/stop` stops the worker's run, collects its checkpoint and releases the
  instance; before `training` it releases at once. `Run.model_id` is set as for a local run, and a stop after the
  worker's last batch changes nothing, as for a local run: the run ends `completed` with all its results.
- **Deadline.** `compute.max_hours` (default 12) bounds what a run can cost: the instance is asked to stop the run
  when the time is up (it ends `stopped`, with its checkpoint, and can be resumed), and is terminated whatever
  happens. Providers that can (`self_terminates`) also arm the instance to terminate itself 10 minutes after the
  deadline, so the bound holds when this machine is off.
- **Failure.** Anything that fails after an instance exists still releases it (also a bug: the release runs
  while the panic unwinds). A worker that stops answering for 10 minutes while the provider says the instance is
  gone (for `remote`: while it does not answer at all) fails the run with `cloud_instance_lost`, and so does an
  instance terminated by hand; the checkpoints copied so far stay, so it can be resumed. So does a copy of the final
  weights that fails while the provider says the instance is gone (or stopped): whichever request noticed it, a
  gone instance is a lost one; `cloud_sync_failed` is a copy that failed from an instance that is still there. A
  launch request that fails as `cloud_unavailable` may have gone through: the instance is looked up by its name, and one that exists is
  used. One that is not listed may only be listed later: the run fails, and the reaper keeps looking for its
  instance by its name (*Instances are never left behind*). A failure code is in `Run.error_code`:
  `cloud_auth_failed`, `cloud_refused` (no capacity, no balance),
  `cloud_unavailable`, `cloud_boot_failed` (the worker check, an instance whose setup failed, or a worker that
  cannot be downloaded: nothing is rented then), `cloud_sync_failed`, `cloud_instance_lost`.
- **Resuming.** `POST /api/runs` with `resume_run_id` resumes a stopped or failed cloud run on a new instance of
  its `compute` (which may change, like `device`), with a new `Run.cloud`: the copied `weights/last.folo` is
  uploaded and the worker continues it (`checkpoint`, docs/API.md), so the step numbers go on. With `"compute":
  null` the run resumes on this machine instead and is a local run from then on (no `Run.cloud`; what its
  instances cost stays in its log).
- **Restarts.** Once it trains, a cloud run does not need this server: close the laptop and it keeps training. A
  server that shuts down lets go of the runs that train, with a log line saying so (`folo mcp` does the same at the
  end of its session), and the next start picks each one up again and fetches the history it missed from the
  worker's cursors. A run that does not train yet has nothing to lose: a shutdown stops it and releases its
  instance, and a start does the same for one a crash left in that state. One process follows a run at a time: the
  one that holds `runs/<id>/cloud.lock`; another shows the run as it was last persisted and refuses to stop it
  (`409 run_active`).

### Chunked transfer reliability
All large transfers between the control plane and a worker go through `folo_client::transfer` in short, retried
requests so provider proxies (such as RunPod's 100 s timeout) and transient drops cost a retry rather than the whole
transfer, while keeping client memory around one part:

- **Chunked uploads (`POST /api/uploads`, `PUT /api/uploads/{id}/parts/{n}`, `POST /api/uploads/{id}/complete`).**
  The stream is cut into parts of at most `min(Upload.max_part_bytes, 8 MiB)` sent in order. Transient errors retry
  up to 6 attempts with exponential backoff (`500 ms` doubling up to `8 s`, polling cancellation every `50 ms` so stops
  take effect promptly). After a failed part `PUT`, the client
  queries `GET /api/uploads/{id}` to see the server's next expected part index: if the part actually landed and only
  its response was lost (`part == n + 1`), the client advances without resending; if still at `n`, it resends; any
  other mismatch fails with `upload_conflict`. On `complete`, if a transient error lost the response and the retry
  sees `404 upload_not_found`, a fingerprinted upload re-posts `POST /api/uploads` to recover the `existing` dataset
  or model instead of failing. On cancellation or final failure, the session is deleted best-effort
  (`DELETE /api/uploads/{id}`).
- **ETag-validated ranged downloads (`GET /api/models/{id}/weights`).** Downloads use `Range` requests with
  `If-Match` pinned to the first part's `ETag`, so a checkpoint overwritten mid-download (`412 Precondition Failed`)
  restarts cleanly from byte 0 rather than mixing two epochs (up to 3 restarts — 4 attempts in all — before failing).
  Each range response streams directly into the hidden temporary file (`*.download`) next to the destination rather
  than buffering the part in memory, checking cancellation while streaming; on a transient error while fetching an
  `8 MiB` part, the temporary file is truncated and rewound back to the last completed byte offset before retrying with
  the same stop-checked backoff so partial bytes from a dropped connection never corrupt the file. The first request
  fetches `8 MiB` and measures link speed: if the remaining bytes are at most `192 MiB` and fit within `30 s` at that
  rate, the rest is fetched in one streamed request (on a transient error, it rewinds and immediately falls back to
  `8 MiB` parts without sleeping).
  Because a single request streams one open file version, a file rewritten between the first and second request (a
  small model finishing epochs every second) is retried whole in one request when the full file fits those same
  bounds. Once complete, the temporary file is flushed with `sync_all` and atomically renamed into place (retrying
  briefly if another process holds the local file open, e.g. on Windows); leftover `.download` files from a crash are
  swept up when the run is next followed.

## Instances are never left behind
Rented GPUs cost money by the hour, so the ledger is written before the provider is called and survives crashes:
- `cloud/instances/<id>.json` is created (`launching`, with the unique instance name `folo-<run id>-<nonce>`)
  *before* the launch request. A server that dies in between finds the record on start, asks the provider for
  instances of that name and terminates them. The reaper leaves alone the instances of runs a process follows (it
  holds their `cloud.lock`), so a launch in flight is never touched.
- An instance whose launch was never confirmed (the answer was lost, or a crash came first) has no provider id, and
  a provider may list a new instance late, so not finding it by its name proves little. For 5 minutes after its
  record was written it stays `terminating` and is looked for at every pass of the reaper. After that it ends:
  `terminated`, with `ended` at its `created` and so no cost (nobody saw it), and its connection can be removed.
  The reaper still looks for it: every 10 minutes until its watchdog is due (the later of the run's `deadline`
  and its time limit from the launch, plus 15 minutes for its setup to start, the 10 minutes and the few it keeps
  trying), then every hour, for 7 days from the launch (or until the watchdog is due, if that is later), and once
  more at the end. The watchdog normally stops the instance itself, but it can fail: it gives up after five failed
  calls, and a container that restarts counts again; that is why FOLO keeps looking. One that shows up is
  terminated like any other (`terminating`, then `terminated` with the time it ran from its launch, a warning in
  the server's log naming it). A look that cannot be made (the connection was removed, the provider does not
  answer) is the record's `error` and does not count: the next pass of the reaper looks again, also for the last
  look, so the watch only ends with a look that got its answer. After that, an instance the provider never listed
  in all that time is not looked for again (the record goes as ended records do), and the provider's console is the
  place to check. A record whose `created` cannot be read, or lies in the future (a clock that went back), skips
  the 5 minutes.
- Terminating is idempotent and retried: a request that fails leaves the instance `terminating`, and a reaper
  thread retries every 30 s (and on every start) until the provider confirms. Studio and Desktop show such an
  instance with its error, and a run does not wait for more than a minute of that: it finishes with a warning in
  its log.
- An instance whose run is no longer active is terminated on start. `DELETE /api/cloud/instances/{id}` (and
  `folo cloud terminate`, which works without a server) terminates one by hand; its run then fails with
  `cloud_instance_lost`.
- A connection cannot be removed while it has live instances (`409 connection_in_use`).

Cost is an estimate: `price_per_hour` of the offer times the time from the launch to the confirmed termination.
An instance that was gone when FOLO looked again (this machine was off, the server crashed) ended at a time nobody
saw. When its provider arms it to terminate itself (`self_terminates`) and its worker had answered, its cost is
counted to when it should have done so: its time limit, the 10 minutes and the few it keeps trying, after that
answer (the ledger keeps when). An instance that was still there, or whose worker never answered, is counted to
the moment its end is confirmed; one that was never found costs nothing until it is. `Run.cloud.cost_usd` follows the estimate at every phase and once a minute while
the run trains. The provider's bill is the truth.

## Providers

### `runpod`
Rents a GPU pod per run through RunPod's REST API v2 (`https://api.runpod.io/v2`) and reaches the worker through
RunPod's HTTPS proxy (`https://<pod id>-7878.proxy.runpod.net`), so the token and the data are encrypted in
transit. The proxy cuts connections after 100 seconds, which is why uploads, downloads and mirroring are all
short requests. It also answers `404` itself while it cannot reach the pod's port, sometimes for a moment after
the worker has already answered: a `404` that is not FOLO's own counts as a failure of the moment and is tried
again (as Cloudflare's `52x` are).
- **Credentials**: an API key (`api_key`; RunPod console → Settings → API Keys, with write access to pods).
  Checking it reads the account's email and balance (GraphQL `myself`).
- **Offers**: every NVIDIA GPU type of the catalog once per cloud it is available in (`<gpu type id>@secure`,
  `<gpu type id>@community`), with the on-demand price and RunPod's stock level as availability. The Secure Cloud and
  Community Cloud catalogs are fetched concurrently (falling back inline if a helper thread cannot be spawned,
  preserving Secure-first error precedence and deterministic sorting). Spot
  (interruptible) pods are not offered, and neither are other vendors' GPUs (the worker finds its GPU through the
  NVIDIA container runtime). Provider API calls retry `429 Too Many Requests` for all methods and transient
  `502`/`503`/`504` responses for idempotent methods (`GET`, `DELETE`, `HEAD`, `PUT` — never `POST`, so a launch is
  never duplicated) up to 3 attempts, honoring `Retry-After` up to `10 s` per wait (`20 s` cumulative; default `2 s`).
  Pre-launch worker URL probes share a pooled HTTP client draining up to 16 KiB to reuse keep-alive connections.
- **Instance**: an on-demand pod of the offer's GPU type with one GPU, a container disk sized for the dataset
  (`compute.disk_gb`, `10..=4096`, default: twice the packed dataset plus 20 GB), port `7878/http`, and the
  bootstrap script as its command. The script arms a watchdog before anything else: 10 minutes after the run's time
  limit, counted from boot, the pod terminates itself with its pod-scoped key (`self_terminates`, up to 5 attempts
  30 s apart, falling back to `runpodctl`), and it does the same 60 s after the script ends on any error or exit
  (leaving time to read the tail of its log in RunPod's console), so a broken pod does not idle at GPU prices. One
  gap remains: a pod whose package install fails has no `curl` to call RunPod with; it exits, and this server
  terminates it when the boot times out.
- **Worker**: the pod starts from `ubuntu:22.04` (overridable via `FOLO_CLOUD_RUNPOD_IMAGE` for experiments — any
  image with `bash` and `apt-get` whose entrypoint can be replaced), installs `folo` for Linux from the release of
  this server's version with `install.sh` (checksum verified) and runs `folo serve --host 0.0.0.0 --port 7878
  --tensor-cores` with its token in `FOLO_TOKEN` (set on the pod, never written into the script). `worker_base_url`
  (a connection field, taking precedence over `FOLO_CLOUD_WORKER_BASE_URL`; must be `http://` or `https://` with a
  host and no spaces or control characters) names another place to download `install.sh` and the release archive
  from: a mirror, or your own build (`<url>/install.sh`, `<url>/folo-x86_64-unknown-linux-gnu.tar.gz` and its
  `.sha256`).
- **The download is checked before a pod is rented.** FOLO asks that place for `install.sh` and the archive's
  `.sha256` as a pod would, without credentials. When it answers that they are not there (`404` for a release
  that is not published or a private repository), the run fails with `cloud_boot_failed` before anything is
  rented, and the connection's check says so in its `note` (Settings → Cloud, `folo cloud connections`). No
  answer, a server error and a refusal of the moment (`408`, `425`, `429`) are not that: the pod may reach what
  this machine cannot.
- **The GPU.** FOLO's GPU backend is wgpu on Vulkan. The script installs the Vulkan loader and asks the NVIDIA
  runtime for its graphics libraries (`NVIDIA_DRIVER_CAPABILITIES=all`). The manifest that runtime mounts names
  the driver's GLX library, which gives Vulkan no device in a container without an X server, so the script asks
  `folo doctor` for the GPU and, when there is none, points the loader at the driver's EGL library instead (the
  same driver's headless entry; on the pods checked, this is what finds the GPU). A pod where neither does fails
  the worker check within minutes.

### `remote`
A FOLO server you run yourself: `folo serve --host 0.0.0.0 --token <secret> --tensor-cores` on the machine with the
GPU (add `--tensor-cores` if you want to train with `amp: true`, and `--tls-cert <cert.pem> --tls-key <key.pem>` to
serve HTTPS natively).
- **Credentials**: its `url` (required) and `token` (optional; required when the remote server runs with `--token` /
  `FOLO_TOKEN`). `url` must start with `https://` or `http://`, contain no embedded `user:password@`, no query string
  or `#fragment`, and no spaces or control characters. Use `https://` across networks you do not control (serve TLS
  natively with `--tls-cert` / `--tls-key`, or put the server behind a TLS proxy or a tunnel); for self-signed or
  private-CA `https://` workers, set `FOLO_TLS_CA=<ca.pem>` in the control plane's environment (honored by `folo cloud`
  and by `folo serve` when running jobs on `remote` workers). When `FOLO_TLS_CA` is set, it replaces the bundled web
  PKI roots for every `https://` `remote` connection in that process, so if you connect to both private-CA and publicly
  signed `remote` workers from the same control plane, include all required CA roots in that PEM bundle (provisioned
  RunPod workers continue to use the bundled public roots).
  `http://` is accepted, with a warning in the connection's `note` unless the host is `localhost`, loopback
  (`127.0.0.0/8`, `::1`), RFC 1918 private IPv4, IPv4 link-local, or IPv6 unique-local (`fc00::/7`) / link-local
  (`fe80::/10`). Scheme and certificate mismatches fail with actionable hints: `https://` against a plain-HTTP
  `folo serve` recommends `http://` or enabling TLS, `http://` against a TLS `folo serve` recommends `https://`, and
  untrusted certificates point to `FOLO_TLS_CA`.
- **Offers**: one offer — id `"gpu"` for the GPU `device: auto` selects, or `"cpu"` when the remote server has no
  usable GPU — with no price.
- No instance is launched or terminated; runs wait in that server's own queue. What a run leaves there stays
  there: the uploaded dataset (reused by later runs of the same files) and the worker's own copy of the run, named
  `<name> (<run id>)`. A server that is too old to be a worker is refused when the connection is made
  (`cloud_refused`, naming its version).
- A start request whose answer was lost is found again as the new run of that name on the server, so two FOLO
  servers that share one `remote` worker and have a run of the same name and id (ids are short) can take each
  other's run for their own in that moment. Give each control plane its own worker, or distinct run names, until
  the worker keeps an identity of each attempt.

### Adding a provider
A provider is one type implementing `folo_cloud::Provider` (check credentials, list offers, launch, look up, find
by name, terminate) plus an entry in `folo_cloud::providers()`. The lifecycle above is provider-agnostic
(`folo_server::cloud`), and tests run it against an in-process worker. `folo serve` now supports native TLS
(`--tls-cert` / `--tls-key`); providers that expose a raw TCP port instead of an HTTPS endpoint (Vast.ai, Lambda) are
not included yet. A new id
also goes into the provider lists that tests assert (`crates/folo-cloud/src/lib.rs`, `crates/folo-cli/tests/cloud.rs`
and `crates/folo-server/tests/mock_cloud.rs`), and its mock offers into `MockStore::offers_of`
(`crates/folo-server/src/mock/mod.rs`).

## Securing a server (`--token` and native TLS)
`folo serve --token <secret>` (or `FOLO_TOKEN`) requires `Authorization: Bearer <secret>` on every `/api` request
except `GET /api/health`; anything else is `401 unauthorized`. Use it whenever the server listens beyond loopback.
To encrypt traffic natively without an external proxy, pass `--tls-cert <cert.pem>` (`FOLO_TLS_CERT`) and
`--tls-key <key.pem>` (`FOLO_TLS_KEY`): `folo serve` validates the PEM certificate chain and matching private key
upfront (never falling back to plaintext HTTP) and serves HTTPS + SSE directly via pure-Rust `rustls`.
A server on loopback answers only loopback `Host` names (docs/API.md, Conventions); one reached through a tunnel
that keeps its own `Host` needs `--allow-host <name>`, and `--token` too.
The comparison is constant-time and the token is never logged. `folo mcp --server <url> --token <secret>` (plus
`--tls-ca <ca.pem>` / `FOLO_TLS_CA` for self-signed or private-CA servers) and the
`remote` provider send it. Studio and Desktop do not sign in to a token-protected server yet: such a server is for
API clients (Studio served by one says "Not connected" and why, instead of failing on every screen).

### Security hygiene
- **Owner-only storage**: `cloud/connections.json` (provider credentials) and `runs/<id>/cloud.json` (each provisioned
  run's generated 256-bit random hex worker token) are written atomically with mode `0600` on Unix (and inherit the
  user-private profile ACL under `~/.folo` on Windows). Neither file is ever served by the HTTP API.
- **Redaction and log scrubbing**: API and CLI records return every secret field as `••••` plus its last 4 characters
  (or `••••` alone when shorter than 8 characters). `Debug` implementations omit secret values, and provider or worker
  responses are scrubbed (`[redacted]`) for any secret of at least 8 characters before reaching error messages, logs
  or offer names.
- **No secrets on the command line or in scripts**: `folo cloud connect` rejects literal secrets in `--secret` or
  `--set` (shell history and process listings are readable) and requires `--secret KEY=env:VAR` or `--secret KEY=-`
  (one line from stdin). Provisioned workers receive their token via the pod's `FOLO_TOKEN` environment variable,
  never in the bootstrap script or on `folo serve`'s argv.
- **No credentials through MCP**: `folo mcp` exposes read/inspect/terminate tools and `train` with `compute`, but
  intentionally has no tool to create connections so API keys and tokens never enter an agent's transcript.

## Using it

### CLI workflow
`folo cloud` operates directly on the data dir (`cloud/connections.json`, `cloud/instances/`) under `cloud/.lock`,
so it works with or without `folo serve` running and a running server sees its changes immediately (docs/CLI.md,
*Cloud*).

**Connect RunPod and inspect offers:**
```sh
folo cloud providers
RUNPOD_API_KEY=rpa_... folo cloud connect runpod --secret api_key=env:RUNPOD_API_KEY
folo cloud offers cc_4d2a              # pick an offer id, e.g. NVIDIA RTX A4000@secure
folo cloud instances                   # monitor live & recent instances and estimated cost
folo cloud terminate ci_8a2f           # manual kill switch if needed
```

**Connect a remote FOLO server:**
```sh
# On the remote GPU machine (native TLS with --tls-cert / --tls-key across untrusted networks):
FOLO_TOKEN="$(openssl rand -hex 32)" folo serve --host 0.0.0.0 --port 7878 --tensor-cores \
  --tls-cert /etc/folo/fullchain.pem --tls-key /etc/folo/privkey.pem

# On your local machine:
folo cloud connect remote --name lab-4090 \
  --set url=https://gpu-box.example.com:7878 \
  --secret token=- < token.txt
folo cloud check cc_9b1c               # re-verify reachability, token and worker features
```

### Studio, Desktop, API and MCP
- **Studio / Desktop**: **Settings → Cloud** adds connections (with masked secret inputs, live balance/status and any
  connection `note`) and lists live/recent instances with a manual Terminate button. **New run → Run on** lets you
  pick a connection and offer, shows the reference-based cost estimate (`est_usd` and `max_usd` from `max_hours`),
  and launches the cloud run. The run detail page tracks `Run.cloud.phase`, upload/download `progress`,
  `unanswered_since` ("Worker not answering"), `ended_in`, instance status and running cost.
- **API**: docs/API.md, *Cloud compute*, *Uploads* and `TrainConfig.compute`.
- **MCP**: `cloud_connections`, `cloud_offers`, `cloud_instances`, `cloud_terminate`, and `train` takes `compute`
  (docs/MCP.md). Credentials are never entered through MCP.

## Troubleshooting and error recovery

| error code / symptom | likely cause | what to do |
|---|---|---|
| `cloud_auth_failed` (`422`) | Invalid/revoked RunPod API key, key missing pod read+write permissions, or wrong/missing `--token` on a `remote` server. | For RunPod, verify key permissions in RunPod Console → Settings → API Keys; for `remote`, verify `--token` / `FOLO_TOKEN`. Re-test with `folo cloud check <id>`. |
| `cloud_refused` (`409`) | RunPod out of stock (`low` stock often means none), low account balance (`402`), account pool restriction (`403`), or a `remote` server too old (missing `uploads`, `run_cursors` or `run_checkpoint`). | Pick another offer (`folo cloud offers <id>`), top up balance, or upgrade the `remote` `folo` binary. |
| `cloud_unavailable` (`502`) | Provider API unreachable, or `remote` server down, scheme mismatch (`https://` against plain HTTP or `http://` against TLS), or untrusted self-signed / private-CA certificate. | Check network / server status. Follow the scheme hint (`http://` vs `https://`), or set `FOLO_TLS_CA=<ca.pem>` for a self-signed or private-CA `remote` server. |
| `cloud_boot_failed` (`502`) | Worker download URL answered `404` (unpublished dev version or private repo, checked before renting), pod failed setup and self-terminated, worker lacked a usable Vulkan GPU (`llvmpipe` / no adapter), or `amp: true` requested without tensor cores. | For dev builds, set `worker_base_url` or `FOLO_CLOUD_WORKER_BASE_URL` to a hosted `install.sh` + archive; check the pod log in RunPod's console; for `amp` on a `remote` server, start it with `--tensor-cores`. |
| `cloud_sync_failed` (`502`) | Dataset/weights upload or final weight download failed after retries while the instance was still alive (e.g. prolonged network outage or worker disk full). | Check `compute.disk_gb` and link stability, then resume (`resume_run_id`) or start again; `remote` servers reuse already-uploaded fingerprinted datasets. |
| `cloud_instance_lost` (`502`) | Instance was terminated by hand, stopped/disappeared for 10 min (`lost_grace`), `remote` worker was unreachable for 10 min, or worker restarted and lost the run (`run_not_found`). | Checkpoints already synced to `runs/<id>/weights/last.folo` stay on disk. Resume the run (`resume_run_id`) on a new instance or locally (`"compute": null`). |
| `offer_not_found` (`404`) | Offer ID is stale or no longer in the provider catalog. | Refresh offers (`GET /api/cloud/connections/{id}/offers?refresh=true` or `folo cloud offers <id>`). |
| `connection_in_use` (`409`) | Tried to disconnect a connection that still has live instances or active cloud runs. | Stop active runs and wait for `folo cloud instances` to show all instances `terminated` (or terminate them manually). |
| `invalid_config` (`400`) | `device` set to something other than `auto`, `max_hours` outside `0.25..=720`, or `disk_gb` outside `10..=4096`. | Omit `device` (or pass `"auto"`) and keep limits in range. |
| `Worker not answering` (`unanswered_since` set) | Temporary proxy blip or network drop while training continues on the worker. | Wait — the worker keeps training and polling backs off up to 30 s, then catches up gap-free via cursors when reachable again. |
| Instance stuck `terminating` | Provider API failed to confirm termination. | The server's reaper retries every 30 s (and on restart) until confirmed; run `folo cloud terminate <ci_id>` to retry immediately or verify in the provider console. |
| `cloud/connections.json` malformed (`internal`) | File was hand-edited into invalid JSON. FOLO never overwrites it. | Fix the JSON syntax or move the file aside so FOLO can create a fresh one. |

## Limits
- One GPU per run; on-demand instances only.
- Validation (`/api/val`, `/eval`), inference and export of a cloud run's models run on this machine, as for any
  model.
- A cloud run does not feed this server's throughput table, and its time estimate is the reference machine's
  (docs/BENCHMARKS.md), so the cost estimate is a rough one.
- The dataset always travels from this machine, at the speed of its uplink: a 174 MB dataset is two minutes from a
  home line, COCO's 20 GB would be an hour or more of a rented GPU waiting. A worker cannot fetch a public dataset
  itself yet. For the folo-s COCO weights (examples/coco, 2026-11-11) the pod was therefore set up by hand: COCO
  straight from cocodataset.org (three minutes in a datacentre) and `folo train` on the pod.
- A RunPod worker installs the release of this server's version. When that version has no published release (a
  development build) or comes from a private fork, give the connection a worker download URL; FOLO says so before
  it rents anything.

## Checking a provider against the real service
The provider's tests run against a fixture of its API. Two things check it against the service itself:
- `cargo test -p folo-cloud --test runpod_live -- --ignored` with `RUNPOD_API_KEY` set: the key, the account and
  the catalog, and with `FOLO_LIVE_LAUNCH=1` a pod that is launched, found by its name, seen running and
  terminated (a few cents).
- A real run. Build the worker's archive from the commit under test (the `release` workflow run by hand makes
  `folo-x86_64-unknown-linux-gnu.tar.gz` and its `.sha256` without releasing anything), serve it with
  `scripts/install.sh` at a URL the pod can reach, connect with that as the worker download URL, and start a run
  with `compute`.

RunPod was checked this way on 2026-11-10 (REST v2; A40 pods on Secure Cloud, NVIDIA driver 595.91; folo-n on
600 images at 320 px and on 3,000 at 640 px, in f32 and with `amp`):
- A pod trains 40 to 90 seconds after the launch request when its host has the image (the install, the upload),
  about four minutes when it has to pull it. `folo doctor` on it finds the GPU with tensor cores, mixed precision
  trains on them, and metrics, logs, samples and both weights files arrive in the local run folder.
- A 174 MB dataset goes up in 8 MiB parts in two minutes from a home line; `weights/last.folo` (42 MB) comes back
  in two to four seconds, also while the worker rewrites it every second.
- A run stopped by hand is collected and released within 20 seconds, and resumes on a new pod. An instance
  terminated by hand fails its run with `cloud_instance_lost` within seconds; resumed, the run finishes on a new
  pod from the checkpoint copied here.
- A server killed while a run trained follows it again at its next start and fetches what it missed. A pod whose
  server never came back terminated itself 10 minutes after its time limit.
- An offer without capacity is refused as `cloud_refused` at once (RunPod's `low` stock often means none), and a
  pod whose setup fails terminates itself within a minute.
