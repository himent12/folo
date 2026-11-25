# folo-client

A blocking HTTP client for a FOLO server's API ([docs/API.md](../../docs/API.md)), and file transfers in parts on
top of it. It depends only on folo-api among the workspace crates. Four crates depend on it:

| crate | what it uses it for |
|---|---|
| folo-mcp | `folo mcp --server`: `HttpStore` (`crates/folo-mcp/src/proxy.rs`) sends every tool's store call through a `Client`, the labeling tools (`crates/folo-mcp/src/label/`) and benchmark tools (`crates/folo-mcp/src/bench/`) too |
| folo-server | The cloud control plane (`crates/folo-server/src/cloud/`) talks to a worker, uploads its dataset and checkpoint and downloads the weights (`transfer`) |
| folo-cloud | The `remote` provider asks the server for its health and system info (`crates/folo-cloud/src/remote.rs`); `Endpoint::client` builds a client of a worker |
| folo-cli | `serve_lock::serves` asks a server at a data dir's recorded address which data dir it serves; `folo label` and `folo bench` go through folo-mcp's `HttpStore` when a `folo serve` owns the data dir (or `--server` is set) |

`Client` knows the API's conventions and nothing about its routes (only `src/transfer.rs` names routes, the
upload protocol's `/uploads` family): paths are relative to `/api`, most request/response bodies are
JSON (with binary byte methods for uploads, downloads, and multipart payloads), non-2xx error answers decode the JSON
error envelope `{ "error": { code, message, hint } }` (while `get_range` also handles bodyless RFC 9110 statuses such
as `412` / `416` on `/weights`), and a token is sent as
`Authorization: Bearer`. `http://` and `https://` (rustls) both work.

## Where things are

| path | what it holds |
|---|---|
| `src/lib.rs` | `Client` (`get`, `post`, `put` (bytes), `put_json`, `delete`, raw bytes, `get_range`), `Error` (`is_unreachable`, `is_transient`, `code`), the codes `UNREACHABLE` and `NOT_FOLO`, `encode` for path segments |
| `src/transfer.rs` | `upload` (a stream a caller produces, sent in parts, resumed after a lost answer), `download` (a file fetched with `Range` requests into a temporary file, renamed into place), `remove_partial`, `Control` (progress, stop, backoff) |

## Start here

1. `Client::new` and `Client::token` in `src/lib.rs`: how a caller gets a client.
2. `Error` and `Error::is_transient` in `src/lib.rs`: which failures a caller may retry.
3. `upload` and `download` in `src/transfer.rs`: the two transfers the cloud control plane uses.

## Common changes

| to do this | touch these files |
|---|---|
| Call a new API route | Nothing here: callers pass the path (`client.get::<DatasetList>("/datasets")`) and a folo-api type. The route itself is folo-server's |
| Change the upload protocol | `upload`, `complete` and `Parts` in `src/transfer.rs` (the `/uploads` routes); the server side is folo-server's |
| Change which failures are retried | `Error::is_transient` in `src/lib.rs` and its test `only_some_statuses_are_transient`; `crates/folo-server/tests/transfer.rs` exercises the retries |

## Tests and benchmarks

```sh
cargo test -p folo-client                       # unit tests: segment encoding, transient statuses, token redaction
cargo test -p folo-server --test transfer       # upload and download against a real server and a lossy proxy
```

The transfer tests live in folo-server because they need a running server. Nothing needs a GPU or the network;
nothing is `#[ignore]`d. No benchmarks.

## Further reading

- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/API.md](../../docs/API.md): the routes and the error envelope (Uploads, Models for downloads).
- [docs/CLOUD.md](../../docs/CLOUD.md): how the control plane uses the client.
- [folo-mcp](../folo-mcp/README.md), [folo-cloud](../folo-cloud/README.md), [folo-server](../folo-server/README.md).
