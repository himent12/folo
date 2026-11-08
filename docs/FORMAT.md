# The `.folo` file format (v1)

A `.folo` file stores model weights (and optionally a full training checkpoint). It is byte-compatible with
[safetensors](https://github.com/huggingface/safetensors), so any safetensors reader can open FOLO weights. FOLO adds
conventions for the metadata block and the tensor names.

## Layout
```
[8 bytes]  N: u64 little-endian, the length of the header in bytes
[N bytes]  header: UTF-8 JSON object, padded at the end with ASCII spaces (0x20) so that 8 + N is a multiple of 8
[rest]     data: the raw tensor bytes, little-endian, row-major (C order), with no gaps
```
Header entries:
```jsonc
{
  "backbone.stem.conv.weight": { "dtype": "F32", "shape": [16, 3, 3, 3], "data_offsets": [0, 1728] },
  "...": { },
  "__metadata__": { "folo.format": "1", "folo.kind": "weights", "...": "..." }
}
```
- `dtype` is one of `F32`, `F16`, `BF16`, `U8`, `I64`, `U32` (FOLO writes `F32` and `F16`; readers must reject others
  with a clear error).
- `data_offsets` are `[begin, end)` byte offsets relative to the start of the data section. A tensor's byte length must
  equal `numel × dtype size`. Offsets must not overlap, must cover the data section exactly, and are written in
  ascending order. FOLO writes tensors sorted by name. Each tensor starts at an offset that is a multiple of its dtype size.
- A zero-sized tensor has `begin == end`. Scalars have `shape: []`.
- The header may not exceed 100 MB. Duplicate keys are invalid.

## Metadata (`__metadata__`)
safetensors requires every metadata value to be a string, so structured values are JSON encoded into strings.

| key | required | value |
|---|---|---|
| `folo.format` | yes | `"1"` |
| `folo.kind` | yes | `"weights"` (inference) or `"checkpoint"` (resumable training state) |
| `folo.arch` | yes | model family and scale, e.g. `"folo-n"` |
| `folo.config` | yes | JSON: the full model config (scale multipliers, `num_classes`, `reg_max`, head type, ...) as defined in docs/MODEL.md |
| `folo.classes` | yes | JSON array of class names, index = class id |
| `folo.imgsz` | yes | training image size, e.g. `"640"` |
| `folo.fused` | no | `"true"` if Conv+BN are already fused (inference-only file) |
| `folo.train` | no | JSON: the TrainConfig used (docs/API.md) |
| `folo.metrics` | no | JSON: best validation metrics `{ "map50", "map50_95", "precision", "recall" }` |
| `folo.epoch` | checkpoint | completed epochs |
| `folo.created` | no | RFC 3339 timestamp |
| `folo.version` | no | FOLO version that wrote the file |

Unknown `folo.*` keys must be preserved by tools that rewrite a file; keys outside the `folo.` namespace are ignored.

## Tensor names
- Model parameters and buffers use stable dotted paths from `Module::named_parameters()` / named buffers, e.g.
  `backbone.stage2.block0.cv1.conv.weight`, `...bn.running_mean`. BatchNorm `num_batches_tracked` is not stored.
- Checkpoints add, under reserved prefixes:
  - `ema.<param path>`: the EMA weights (inference should prefer these when present); `ema.__updates` is a scalar `I64`.
  - `optim.<param path>.<state>`: optimizer state (e.g. `momentum_buffer`, `exp_avg`, `exp_avg_sq`); `optim.__step` is
    a scalar `I64`.
  - `sched.__epoch` / `sched.__step` scalars as needed.
- A `weights` file never contains `ema.`, `optim.` or `sched.` tensors. `folo export` writes the EMA weights (if
  present) as plain parameter names, optionally fused and converted to `F16`.

## Compatibility rules
- Readers check `folo.format` and refuse a higher major version with a clear message.
- Loading into a model is strict by default: missing or unexpected parameters and shape mismatches are errors that list
  every offending name. A `--partial` / `strict: false` mode loads the intersection and reports the rest (used for
  transfer learning when `num_classes` differs: head classification layers are skipped).
- Files are memory-mapped for reading where possible; writers stream tensors in order and write the header first
  (offsets are computable up front).
