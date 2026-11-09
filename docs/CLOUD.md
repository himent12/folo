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
and next to a local run). `Run.cloud.phase` and the SSE `cloud` event follow it:

| phase | what happens | `Run.status` |
|---|---|---|
| `provisioning` | the provider is asked for an instance of the offer (a `remote` connection skips this) | `queued` |
| `booting` | wait for the instance, then for its worker to answer; check it (below) | `queued` |
| `syncing` | upload the dataset (and the weights or checkpoint to start from) | `queued` |
| `training` | the worker trains; metrics, logs and samples are mirrored; checkpoints are copied back | the worker's: `queued` while it waits there, then `running` |
| `collecting` | the run ended on the worker: copy `best.folo` and `last.folo` | unchanged (`running` / `stopping`) |
| `releasing` | terminate the instance | unchanged |
| `released` | done; the final status is published | `completed` / `stopped` / `failed` |

- **Worker check (booting).** The worker must answer `/api/health` with the token, list the features `uploads`,
  `run_cursors` and `run_checkpoint` in `/api/system`, and resolve `device: auto` to a GPU; a run with `amp` also
  needs `capabilities.amp.available`. Otherwise the run fails with `cloud_boot_failed` before anything is uploaded,
  and the message says which check failed (a worker that is too old, an instance without a usable GPU adapter
  with the adapters it reported, mixed precision without tensor cores). A provisioned instance gets 15 minutes to
  pass the check.
- **Dataset.** The dataset is sent as FOLO reads it, not as it lies on disk: a tar stream in the canonical YOLO
  layout (`data.yaml`, `images/<split>/<n>.<ext>`, `labels/<split>/<n>.txt`, files numbered in dataset order;
  images byte for byte, labels written from the parsed boxes), produced on the fly by `folo_data::pack` and cut
  into upload parts. So datasets of any format and layout travel the same way, images the loader skips stay home,
  and image ids are the same on both sides (validation samples line up). The upload carries the dataset's
  fingerprint: a worker that already has that dataset (a `remote` server after its first run) answers with it and
  nothing is sent.
- **Mirroring.** While the worker trains, the control plane polls it (about once a second; backing off to 30 s
  while it does not answer): the run, new steps and epochs (`/metrics?after_step=&after_epoch=`), new log lines
  (`/logs?after=`) and each new epoch's samples. They go through the same sink as a local trainer's events, so
  `metrics.jsonl`, `log.jsonl`, `samples/`, SSE and `Run` behave as for a local run (subscribers get every epoch
  and at most the newest step of each poll). Polling, not SSE, because provider proxies cut long connections;
  every request is short and retried.
- **Checkpoints.** `weights/last.folo` is copied to the run folder after an epoch when the last copy is older than
  10 minutes (`Run.cloud.synced` is when), and both files at the end. Downloads use `Range` requests in parts,
  validated by ETag, so a file replaced mid-download starts over instead of mixing two epochs.
- **Stopping.** `POST /api/runs/{id}/stop` stops the worker's run, collects its checkpoint and releases the
  instance; before `training` it releases at once. `Run.model_id` is set as for a local run.
- **Deadline.** `compute.max_hours` (default 12) bounds what a run can cost: the instance is asked to stop the run
  when the time is up (it ends `stopped`, with its checkpoint, and can be resumed), and is terminated whatever
  happens. Providers that can (`self_terminates`) also arm the instance to terminate itself 10 minutes after the
  deadline, so the bound holds when this machine is off.
- **Failure.** Anything that fails after an instance exists still releases it. A worker that stops answering
  for 10 minutes while the provider says the instance is gone fails the run with `cloud_instance_lost`; the
  checkpoints copied so far stay, so it can be resumed. A failure code is in `Run.error_code`:
  `cloud_auth_failed`, `cloud_refused` (no capacity, no balance), `cloud_unavailable`, `cloud_boot_failed`,
  `cloud_sync_failed`, `cloud_instance_lost`.
- **Resuming.** `POST /api/runs` with `resume_run_id` resumes a stopped or failed cloud run on a new instance of
  its `compute` (which may change, like `device`): the copied `weights/last.folo` is uploaded and the worker
  continues it (`checkpoint`, docs/API.md). With `"compute": null` the run resumes on this machine instead.
- **Restarts.** A cloud run does not need this server: close the laptop and it keeps training. On start the
  server picks up every cloud run that was `training` (its history is fetched from the worker's cursors) and
  releases the instances of runs that had not got that far (they become `stopped`). On shutdown cloud runs are
  left alone, with a log line saying so. One process follows a run at a time (`runs/<id>/cloud.lock`).

## Instances are never left behind
Rented GPUs cost money by the hour, so the ledger is written before the provider is called and survives crashes:
- `cloud/instances/<id>.json` is created (`launching`, with the unique instance name `folo-<run id>-<nonce>`)
  *before* the launch request. A server that dies in between finds the record on start, asks the provider for
  instances of that name and terminates them.
- Terminating is idempotent and retried: a request that fails leaves the instance `terminating`, and a reaper
  thread retries every 30 s (and on every start) until the provider confirms. Studio and Desktop show such an
  instance with its error, and a run does not wait for more than a minute of that: it finishes with a warning in
  its log.
- An instance whose run is no longer active is terminated on start. `DELETE /api/cloud/instances/{id}` (and
  `folo cloud terminate`, which works without a server) terminates one by hand; its run then fails with
  `cloud_instance_lost`.
- A connection cannot be removed while it has live instances (`409 connection_in_use`).

Cost is an estimate: `price_per_hour` of the offer times the time from the launch to the confirmed termination.
The provider's bill is the truth.

## Providers

### `runpod`
Rents a GPU pod per run through RunPod's REST API v2 (`https://api.runpod.io/v2`) and reaches the worker through
RunPod's HTTPS proxy (`https://<pod id>-7878.proxy.runpod.net`), so the token and the data are encrypted in
transit. The proxy cuts connections after 100 seconds, which is why uploads, downloads and mirroring are all
short requests.
- **Credentials**: an API key (`api_key`; RunPod console → Settings → API Keys, with write access to pods).
  Checking it reads the account's email and balance (GraphQL `myself`).
- **Offers**: every GPU type of the catalog once per cloud it is available in (`<gpu type id>@secure`,
  `<gpu type id>@community`), with the on-demand price and RunPod's stock level as availability. Spot
  (interruptible) pods are not offered.
- **Instance**: an on-demand pod of the offer's GPU type with one GPU, a container disk sized for the dataset
  (`compute.disk_gb`, default: twice the dataset plus 20 GB), port `7878/http`, and the bootstrap script as its
  command. It terminates itself after the deadline with its pod-scoped key (`self_terminates`).
- **Worker**: the pod starts from a stock Ubuntu image, installs `folo` for Linux from the release of this
  server's version with `install.sh` (checksum verified) and runs `folo serve --host 0.0.0.0 --token …
  --tensor-cores`. `worker_base_url` (a connection field, or `FOLO_CLOUD_WORKER_BASE_URL`) names another place to
  download `install.sh` and the release archive from: a mirror, or your own build
  (`<url>/install.sh`, `<url>/folo-x86_64-unknown-linux-gnu.tar.gz` and its `.sha256`). FOLO's GPU backend is
  wgpu on Vulkan: the script installs the Vulkan loader and asks the NVIDIA runtime for its graphics libraries
  (`NVIDIA_DRIVER_CAPABILITIES=all`); a pod where that yields no adapter fails the worker check within minutes.

### `remote`
A FOLO server you run yourself: `folo serve --host 0.0.0.0 --token <secret>` on the machine with the GPU.
- **Credentials**: its `url` and `token`. Use `https://` across networks you do not control (put the server
  behind a TLS proxy or a tunnel); `http://` is accepted, with a warning in the connection's `note` unless the
  host is loopback or a private address.
- **Offers**: one, the server's GPU (`/api/system`), with no price.
- No instance is launched or terminated; runs wait in that server's own queue; uploaded datasets stay there and
  are reused by later runs.

### Adding a provider
A provider is one type implementing `folo_cloud::Provider` (check credentials, list offers, launch, look up, find
by name, terminate) plus an entry in `folo_cloud::providers()`. The lifecycle above is provider-agnostic
(`folo_server::cloud`), and tests run it against an in-process worker. Providers that expose a raw TCP port
instead of an HTTPS endpoint (Vast.ai, Lambda) need TLS on the worker first and are not included yet.

## Securing a server (`--token`)
`folo serve --token <secret>` (or `FOLO_TOKEN`) requires `Authorization: Bearer <secret>` on every `/api` request
except `GET /api/health`; anything else is `401 unauthorized`. Use it whenever the server listens beyond loopback.
The comparison is constant-time and the token is never logged. `folo mcp --server <url> --token <secret>` and the
`remote` provider send it. Studio and Desktop do not sign in to a token-protected server yet: such a server is for
API clients.

## Using it
- **Studio / Desktop**: Settings → Cloud connects providers and lists instances; New run → Run on picks a
  connection and an offer and shows what the run would cost; a cloud run's page shows its phase, instance and
  cost.
- **CLI**: `folo cloud providers | connect | connections | check | disconnect | offers | instances | terminate`
  (docs/CLI.md); they work on the data dir, with or without a server running.
- **API**: docs/API.md, *Cloud compute*, *Uploads* and `TrainConfig.compute`.
- **MCP**: `cloud_connections`, `cloud_offers`, `cloud_instances`, `cloud_terminate`, and `train` takes `compute`
  (docs/MCP.md). Credentials are never entered through MCP.

## Limits
- One GPU per run; on-demand instances only.
- Validation (`/api/val`, `/eval`), inference and export of a cloud run's models run on this machine, as for any
  model.
- A cloud run does not feed this server's throughput table, and its time estimate is the reference machine's
  (docs/BENCHMARKS.md), so the cost estimate is a rough one.
- The RunPod provider is written against RunPod's documented API and tested against a fixture of it; run
  `cargo test -p folo-cloud --test runpod_live -- --ignored` with `RUNPOD_API_KEY` set to check it against the
  real service (it launches nothing unless `FOLO_LIVE_LAUNCH=1`).
