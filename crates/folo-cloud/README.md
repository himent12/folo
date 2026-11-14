# folo-cloud

Cloud compute providers behind one trait, `Provider` ([docs/CLOUD.md](../../docs/CLOUD.md)). A provider checks a
connection's credentials, lists what the connection can run on (offers) and, when it provisions, launches, looks up,
finds and terminates instances whose only job is to run `folo serve` (the worker). Everything after the launch
(uploading the dataset, starting and mirroring the run, collecting the weights, the instance ledger) is
provider-agnostic and lives in `folo_server::cloud`. Among the workspace crates it depends only on folo-api and
folo-client; it has no engine or server dependency. folo-server depends on it (`folo_cloud::providers()` in
`crates/folo-server/src/real/mod.rs` and `crates/folo-server/src/cloud/mod.rs`), and FOLO Desktop takes it as a
dev-dependency to check that its demo lists the real providers.

## Where things are

| path | what it holds |
|---|---|
| `src/lib.rs` | The `Provider` trait and its types (`Values`, `Account`, `Endpoint`, `LaunchSpec`, `InstanceState`, `Error`, `ErrorKind`), `providers()`, `WORKER_PORT`, where workers download FOLO from (`worker_base_url`, `worker_download`) |
| `src/runpod/` | RunPod: `mod.rs` (`RunPod`, REST v2 and GraphQL calls, answer types, `IMAGE`, `SELF_TERMINATE`), `provider.rs` (the `Provider` impl), `offers.rs` (secure and community clouds, GPU types as offers), `tests.rs` |
| `src/remote.rs` | `Remote`: a FOLO server that already runs (URL and token). It launches nothing; its one offer is the device the server trains on |
| `src/bootstrap.rs` | `script`: the bash script a provisioned instance runs to become a worker (watchdog, Vulkan setup, `install.sh`, `folo serve`) |
| `src/http.rs` | `Api` and `Reply`: JSON over HTTPS with a bearer key, timeouts, `429` waited out, secrets scrubbed from messages |
| `src/fixture.rs` | Test only (`#[cfg(test)]`): a scripted HTTP server on loopback and a bash syntax check |
| `tests/runpod_live.rs` | The RunPod provider against the real service, `#[ignore]`d |

## Start here

1. `Provider` in `src/lib.rs`: the whole contract; the provisioning methods have defaults that refuse.
2. `providers()` in `src/lib.rs`: the list the server and the CLI show.
3. `Remote` in `src/remote.rs`: the simplest provider (`check`, `offers`, `fixed_endpoint`).
4. `src/runpod/provider.rs`, then `src/runpod/mod.rs`: a provider that launches and terminates instances.
5. `script` in `src/bootstrap.rs`: what a launched instance does.

## Common changes

| to do this | touch these files |
|---|---|
| Add a provider | A type implementing `Provider` (a new module, like `src/remote.rs` or `src/runpod/`) and its entry in `providers()` (`src/lib.rs`); unit tests against `crate::fixture::Fixture`; the provider ids asserted in `providers_have_unique_ids_and_secret_fields_are_marked` (`src/lib.rs`), `crates/folo-cli/tests/cloud.rs` and `crates/folo-server/tests/mock_cloud.rs`; the mock store's offers in `MockStore::offers_of` (`crates/folo-server/src/mock/mod.rs`); the Providers section of docs/CLOUD.md. FOLO Desktop's `the_demo_lists_the_real_providers` (`apps/desktop/src/api/mock/mod.rs`) compares its demo list with `providers()` |
| Map a new provider failure | An `ErrorKind` variant and its code in `Error::code` (`src/lib.rs`), the code in `crates/folo-api/src/codes.rs`, and its hint in `provider_error_of` (`crates/folo-server/src/cloud/mod.rs`) |
| Change what a worker runs at boot | `src/bootstrap.rs` and its tests there (`bash_parses_the_script` needs bash); docs/CLOUD.md |

## Tests and benchmarks

```sh
cargo test -p folo-cloud                          # unit tests; no network, no account
RUNPOD_API_KEY=rpa_... cargo test -p folo-cloud --test runpod_live -- --ignored --nocapture
```

- The unit tests never reach a real provider: `RunPod::with_urls` points RunPod at a `Fixture`, a scripted HTTP
  server on a loopback port that answers in order, answers `599` to an unscripted request and records every
  request. `remote.rs` tests talk to a `Fixture` as the server.
- `fixture::bash_syntax` checks the bootstrap script with `bash -n` where bash is installed.
- `tests/runpod_live.rs` is `#[ignore]`d because it needs `RUNPOD_API_KEY` and the network. `check` and `offers`
  only read the account and catalog; `launch_wait_and_terminate` also needs `FOLO_LIVE_LAUNCH=1` and rents a GPU
  for a few minutes. Without the key each test says so and passes.
- The cloud lifecycle itself (launch, boot, train, collect, release) is tested in folo-server
  (`crates/folo-server/tests/cloud/`), against an in-process worker.
- No GPU needed. No benchmarks.

## Further reading

- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/CLOUD.md](../../docs/CLOUD.md): concepts, a cloud run step by step, the providers, limits.
- [docs/API.md](../../docs/API.md): the Cloud compute routes.
- [folo-server](../folo-server/README.md) (the control plane in `src/cloud/`), [folo-client](../folo-client/README.md),
  [folo-cli](../folo-cli/README.md) (`folo cloud`).
