# The `.folo` file format (v1)

A `.folo` file stores model weights, or a full training checkpoint. It is byte-compatible with
[safetensors](https://github.com/huggingface/safetensors): every `.folo` file is a valid safetensors file, and FOLO adds
conventions for the metadata block and the tensor names. "Writers" below means FOLO's writer; "readers" means FOLO's
reader.

Two levels are distinguished: **container parsing** (the safetensors layer: any valid safetensors file parses) and
**model loading** (requires FOLO metadata). A plain safetensors file without `folo.*` metadata can be loaded only
through the explicit import path (`folo import --config <json> --classes <json>` / `ModelFile::import(config, classes)`),
where the caller supplies what the metadata would have said. Loading then follows the same rules below.

## Layout
```
[8 bytes]  N: u64 little-endian, the length of the header in bytes, including any trailing padding
[N bytes]  header: a UTF-8 JSON object whose first byte is `{` (no leading whitespace or BOM), optionally followed
           by ASCII spaces (0x20). Writers pad so that 8 + N is a multiple of 8.
[rest]     data: raw tensor bytes, little-endian, row-major (C order)
```
Header entries:
```jsonc
{
  "__metadata__": { "folo.format": "1", "folo.kind": "weights", "...": "..." },
  "backbone.stem.conv.weight": { "dtype": "F32", "shape": [16, 3, 3, 3], "data_offsets": [0, 1728] }
}
```
Reader acceptance rules (identical to safetensors):
- `N <= 100_000_000` bytes. The header is a JSON object with no duplicate keys. `__metadata__`, if present, maps
  strings to strings.
- `data_offsets` are `[begin, end)` relative to the start of the data section, and `end - begin == numel × dtype size`.
  Taken together, the tensors must cover the data section exactly, with no overlaps and no holes (sorted by `(begin, end)`,
  each tensor starts where the previous one ended; the first starts at 0, and the last ends at the end of the file).
- Zero-sized tensors have `begin == end`. Scalars have `shape: []`.
- Readers do not require any particular order of keys or any element alignment.

Writer conventions (always valid under the rules above):
- Tensors are laid out sorted by **descending dtype size, then by name**, which keeps every tensor naturally aligned
  with no padding between tensors (the data section starts at an 8-aligned offset).

## Dtypes
| dtype | bytes | used for |
|---|---|---|
| `F32` | 4 | parameters, buffers, optimizer state (training) |
| `F16` | 2 | parameters and buffers in exported inference files |
| `I64` | 8 | scalar counters in checkpoints (`*.__step`, `ema.__updates`, ...) |
| `U8`, `U32` | 1, 4 | reserved (masks, indices) |

Readers must support all five. `BF16` and other safetensors dtypes are readable with a clear "unsupported dtype" error
when FOLO cannot use them.

## Metadata (`__metadata__`)
All values are strings; structured values are JSON encoded into a string.

| key | required | value |
|---|---|---|
| `folo.format` | yes | `"1"`. Readers refuse any other major version with a clear message |
| `folo.kind` | yes | `"weights"` (inference / transfer) or `"checkpoint"` (resumable training state) |
| `folo.arch` | yes | family and scale, e.g. `"folo-n"`. Informational; `folo.config` is authoritative |
| `folo.config` | yes | JSON: the full model config as specified in docs/MODEL.md (`scale`, multipliers, `num_classes`, `reg_max`, head type, ...). Readers validate it against the schema of that version, and `num_classes` must equal `len(folo.classes)` |
| `folo.classes` | yes | JSON array of class names, where the index is the class id |
| `folo.imgsz` | yes | training image size, e.g. `"640"` |
| `folo.fused` | no | `"true"` if Conv+BN are fused (an inference-only tensor set) |
| `folo.dtype` | no | `"F32"` (default) or `"F16"`: the floating dtype of parameters and buffers |
| `folo.train` | checkpoint | JSON: the resolved TrainConfig (docs/API.md) with every default filled in (informational; the contracts below are authoritative for resuming) |
| `folo.optim` | checkpoint | JSON: optimizer contract, see below |
| `folo.sched` | checkpoint | JSON: LR / momentum schedule contract, see below |
| `folo.ema` | checkpoint, if EMA | JSON: `{ "decay": 0.9999, "tau": 2000 }`. The effective decay at update `u` is `decay × (1 − exp(−u / tau))`, and `u` is `ema.__updates` |
| `folo.rng` | checkpoint | JSON: `{ "algo": "<generator name from folo_core::rng, e.g. xoshiro256pp>", "seed": <u64>, "streams": { "<stream name>": "<state as lowercase hex of the generator's state words, little-endian>" } }`. Stream names are defined by the trainer (e.g. `data`, `augment`, `init`) |
| `folo.epoch` | checkpoint | number of completed epochs. In v1, checkpoints are written only at epoch boundaries, so no within-epoch loader position is stored |
| `folo.metrics` | no | JSON: best validation metrics `{ "map50", "map50_95", "precision", "recall" }` |
| `folo.best` | checkpoint | JSON: early-stopping state `{ "epoch": <best epoch, -1 if none>, "fitness": <best fitness> }`, see docs/MODEL.md (validation schedule) |
| `folo.created` | no | RFC 3339 timestamp |
| `folo.version` | no | FOLO version that wrote the file |

Unknown `folo.*` keys must be preserved by tools that rewrite a file. Keys outside the `folo.` namespace are ignored.

## Tensor namespaces
Every tensor name falls into exactly one namespace, decided by its first dotted segment:

| namespace | contents |
|---|---|
| *(model)* | anything not starting with a reserved prefix: parameters and buffers from `Module::named_parameters()` / named buffers, e.g. `backbone.stage2.block0.cv1.conv.weight`, `...bn.running_mean`, `...bn.running_var`. BatchNorm `num_batches_tracked` is not stored |
| `ema.` | `ema.<model name>` for **every** model parameter and floating buffer (complete coverage), plus the scalar `ema.__updates: I64` |
| `optim.` | `optim.<model name>.<state>` (`momentum_buffer` for SGD; `exp_avg`, `exp_avg_sq` for AdamW), plus `optim.__step: I64` |
| `sched.` | scheduler scalars, e.g. `sched.__step: I64` |
| `__` (top level) | reserved for future use. Readers ignore unknown `__*` tensors with a warning |

A `weights` file contains only the model namespace. A `checkpoint` contains the model namespace (the live training
weights), a complete `ema.` set when EMA is enabled, and `optim.` / `sched.`.

### EMA
The EMA tracks parameters and the floating BatchNorm buffers (`running_mean`, `running_var`), as Ultralytics does.
Coverage must be complete: a checkpoint with a partial `ema.` set is invalid. Inference and export use the EMA state
when present. Resuming training restores the live model from the model namespace and the EMA from `ema.`
separately.

### Optimizer contract (`folo.optim`)
```jsonc
{ "type": "sgd" | "adamw",
  "groups": [ { "name": "decay", "params": ["backbone.stem.conv.weight", "..."],
                "lr": 0.01, "momentum": 0.937, "nesterov": true, "weight_decay": 0.0005,
                "betas": [0.9, 0.999], "eps": 1e-8 } ],   // the fields that apply to `type`
  "step": 1234 }                                           // number of optimizer.step() calls; equals optim.__step
```
Every model parameter appears in exactly one group. `lr` is the group's base LR, before the scheduler multiplier.
Loaders must reject a checkpoint whose optimizer type or parameter set does not match the model being resumed.
State tensors must be complete: every parameter in a group has each state tensor its optimizer type uses
(`momentum_buffer` for SGD with momentum > 0; `exp_avg` and `exp_avg_sq` for AdamW), with the parameter's shape and dtype
`F32`. Missing or mismatched state is an error on resume. The only exception is a checkpoint with `step == 0`, which may
omit state tensors (they are initialised to zero).

### Scheduler contract (`folo.sched`)
```jsonc
{ "type": "cosine" | "linear",        // decay shape after warmup
  "unit": "iteration",                // the schedule advances once per optimizer.step()
  "epochs": 100, "steps_per_epoch": 250, // total length = epochs × steps_per_epoch iterations
  "lrf": 0.01,                        // final LR = base lr × lrf
  "warmup_iters": 750,                // linear warmup length in iterations (0 = none)
  "warmup_bias_lr": 0.1,              // bias-group LR at iteration 0, ramping to its scheduled value
  "warmup_momentum": 0.8,             // momentum (SGD) / beta1 (AdamW) at iteration 0, ramping to the group value
  "step": 1234 }                      // completed iterations; equals sched.__step
```
The exact formulas (cosine shape, warmup interpolation) are defined in docs/MODEL.md (training recipe). Given this
object and the optimizer groups, the LR and momentum of every group at any iteration are fully determined.

## Loading rules
1. Parse and validate the header (acceptance rules above). Partition tensors by namespace.
2. Choose the source: for inference or export, the EMA weights if present (every `ema.<model name>` tensor with the
   prefix stripped; the counter `ema.__updates` and any other `ema.__*` scalar are EMA state, never model tensors),
   otherwise the model namespace. For resume, the model namespace, with `ema.` / `optim.` / `sched.` routed to their
   owners.
3. **Strict** (the default) compares only the chosen source against the model's parameters and buffers. Missing names,
   unexpected names and shape mismatches are errors that list every offending name. When loading into an existing
   model that has its own class list, strict loading also requires `folo.classes` to be identical (same names, same
   order). Otherwise it fails and suggests partial (transfer) loading.
4. **Partial** (`strict: false`, used for transfer learning) loads a tensor only when the name matches, the shape is
   identical, and the dtype converts. Everything else is skipped and reported. If `folo.classes` differs from the
   target model's classes (in count **or** order), the classification output layers of the head (named in
   docs/MODEL.md) are always skipped, even when the shapes happen to match, so weights for the wrong class order are
   never loaded silently.
5. Dtype conversion on load: `F16 → F32` is allowed (the model computes in F32). `F32 → F16` happens only in export.
6. Files with `folo.fused = "true"` contain a different tensor set (no BatchNorm). They can be loaded for inference
   only; loading them for training is an error that says so.

## Export (`folo export`)
Take the EMA state (or the model namespace), fuse Conv+BN in **F32**, then optionally convert every floating tensor to
`F16`. Write a `weights` file with `folo.fused = "true"` and `folo.dtype` set accordingly. No `ema.`, `optim.` or
`sched.` tensors.

## Implementation notes
Readers memory-map the file where possible. Writers compute all offsets up front from the shapes and dtypes, write
the header, then stream the tensors in layout order.
