# folo-test-support

folo-test-support holds the test helpers FOLO's crates share: scratch folders that remove themselves and a guard for
spawned processes. It has no dependencies and is never published (`publish = false`). It is a dev-dependency of
folo-nn, folo-gpu, folo-data, folo-vision, folo-wasm, folo-cloud, folo-server, folo-mcp and folo-cli, and of FOLO
Desktop and folo-ui in `apps/desktop`.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | `root`, `TestDir`, `TestFile`, `ChildGuard`, `ROOT_ENV`, and their own unit tests |
| `tests/hygiene.rs` | The workspace check that no test or test helper uses the system temp dir directly |

## What it provides

- `root()`: the folder test directories go in: `FOLO_TEST_TMP` when it is set and not empty (created if missing;
  it must be an absolute path), else the system temp dir.
- `TestDir::new(tag)`: a fresh `folo-<tag>-<pid>-<n>` directory under `root()`, removed with everything in it when
  the guard drops, also when the test panics.
- `TestDir::at(path)`: a guard for a directory at `path`, outside `root()`. It first deletes whatever already exists
  at `path`, then creates the directory fresh, so never point it at a directory whose contents you need.
- `TestFile::new(tag, name)`: the path of a file `name` in its own `TestDir` (the file is not created).
- `ChildGuard`: a spawned process that is killed and reaped when the guard drops. Declare it after the `TestDir`
  the process uses, so the process ends before the directory is removed (on Windows an open file cannot be deleted).

## What `tests/hygiene.rs` enforces

It scans the `src/` and `tests/` folders of every other crate under `crates/`, `apps/desktop/src`, and folo-ui's
`src`, `tests` and `examples`, and fails on any line outside a comment that contains `env::temp_dir` or
`temp_dir()`. A test that needs a folder takes a `TestDir` or a `TestFile` instead. Benchmarks are not checked:
`cargo test` does not run them.

## Tests

```sh
cargo test -p folo-test-support                                  # the helpers and the hygiene check
bash tools/test-tmp.sh cargo test --workspace                    # every test, as CI runs them
```

`tools/test-tmp.sh` runs the tests the way CI does and fails if they leave `folo-*` entries behind (AGENTS.md, Tests
and temp dirs).

## Further reading

- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [CI.md](../../docs/CI.md), *Test temp dirs*.
