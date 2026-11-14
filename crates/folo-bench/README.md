# folo-bench

folo-bench is the engine of FOLO's vision benchmarks ([BENCH.md](../../docs/BENCH.md)): the stress conditions that
make variants of an image, the catalog and plans, and every number of an entry's score. It is a pure library: no
files, no threads of its own (rayon data parallelism only), no server types, and every result is a function of its
inputs alone. It depends on folo-api (the wire types the scores are returned in), folo-core (the seeded RNG),
folo-data (the image decoder datasets use) and folo-vision (AP comes from its `Evaluator` and `CocoEval`, never a
second implementation). folo-server depends on it: `src/bench/` stores and schedules benchmarks and
`src/routes/bench.rs` serves `/api/bench`. FOLO Desktop uses it in `apps/desktop/src/api/mock_bench/`.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | `ENGINE_VERSION`, the `Error` type, the re-exports |
| `src/conditions/` | `mod.rs` (`Condition` and the parameter tables, per severity), `catalog.rs` (id, name, group, description, parameters, and the docs table), `transform.rs` (`apply`: a condition on pixels and boxes), `encoding.rs` (variant seeds, decoding, the canonical JPEG), `render.rs` (image file to canonical JPEG), `tests.rs` |
| `src/pixels.rs` | Pixel kernels: sRGB tables, seeded noise, separable and sparse convolution, resampling |
| `src/plan.rs` | `catalog` (`GET /api/bench/catalog`), `default_plan`, `check_plan`, `variants`, and their unit tests |
| `src/score.rs` | `evaluate`, `map`, `robustness`, `score`, `outcome`, `labels`: docs/BENCH.md's *Definitions* |
| `src/score/tests.rs` | Hand-worked cases for every number of the definitions |
| `tests/golden.rs` | Golden digests of every condition's pixels, the canonical JPEG, and one score |
| `tests/determinism.rs` | Variants and scores are identical for 1, 2, 3 and 8 rayon threads |
| `benches/conditions.rs` | Every condition at severity 3 on a 1280 x 720 image (`FOLO_BENCH_IMAGE=<file>` for a real one) |
| `benches/score.rs` | Scoring 500 images, at about 30 and at 300 detections per image |
| `examples/contact_sheet.rs` | Contact sheets of every condition at every severity, for looking at a real photo |

## Start here

1. `Condition` in `src/conditions/mod.rs`: the twelve conditions, and the rules on resolution and determinism.
2. `apply` in `src/conditions/transform.rs` and `render` in `src/conditions/render.rs`: how a variant is made.
3. `check_plan` and `default_plan` in `src/plan.rs`: what a plan may contain.
4. The module docs of `src/score.rs`, then `evaluate` and `score`: how an entry is graded.
5. `ENGINE_VERSION` in `src/lib.rs` and `tests/golden.rs`: what makes two benchmark results comparable.

## The golden test and `ENGINE_VERSION`

`ENGINE_VERSION` goes into every benchmark's fingerprint, so entries graded by different engines never compare as
equal. `tests/golden.rs` pins the pixels of every condition, the canonical JPEG of two variants and the total of a
hand-built entry, on every machine and thread count. A failure there means the engine computes something else than it
did. If that is intended (a condition or a definition of the score changed on purpose), bump `ENGINE_VERSION`, update
the `assert_eq!(ENGINE_VERSION, ..)` in the test, and re-bless the constants with the values the failure prints. If it
is not intended, the change has a bug.

## Common changes

| To do this | Touch these files |
|---|---|
| Change what a condition does | Its parameter table in `src/conditions/mod.rs` or its code in `transform.rs` / `src/pixels.rs`; `src/conditions/tests.rs`; bump `ENGINE_VERSION` and re-bless `tests/golden.rs`; BENCH.md (Conditions): regenerate its table with `FOLO_BLESS=1 cargo test -p folo-bench docs_table_is_in_step` |
| Add a condition | A variant of `Condition` and its parameter table (`src/conditions/mod.rs`); `ALL`, `id`, `name`, `group`, `description` and the parameters in `catalog.rs`; its arm in `apply` (`transform.rs`); `src/conditions/tests.rs`; a `PIXELS` entry in `tests/golden.rs` (in `Condition::ALL` order) with an `ENGINE_VERSION` bump; the count in the `src/lib.rs` docs; BENCH.md (the table, as above) |
| Change a definition of the score | `src/score.rs`; `src/score/tests.rs`; bump `ENGINE_VERSION` and re-bless `tests/golden.rs`; BENCH.md (*Definitions*) |
| Change what a plan may contain | `check_plan` and its unit tests in `src/plan.rs`; API.md (*Vision benchmarks*) |

## Tests and benchmarks

```sh
cargo test -p folo-bench
cargo bench -p folo-bench --bench conditions
cargo bench -p folo-bench --bench score
cargo run -p folo-bench --release --example contact_sheet -- <image> <out dir> [<COCO instances.json>]
```

Nothing here needs a GPU, and no test is `#[ignore]`d. CI runs the golden test on Linux and Windows.

## Further reading

- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [BENCH.md](../../docs/BENCH.md): what a benchmark is, the conditions, the score and its definitions.
- [API.md](../../docs/API.md), *Vision benchmarks*: the HTTP side, served by folo-server.
- Neighbours: [folo-vision](../folo-vision/README.md) (AP), [folo-data](../folo-data/README.md) (decoding),
  [folo-server](../folo-server/README.md) (storage, scheduling, routes).
