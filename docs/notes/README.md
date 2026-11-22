# Engineering notes

The working record of how FOLO was built: process, lab notebooks and test-drive reports, kept as they were written.
They are not user documentation (that is one folder up, starting at [QUICKSTART](../QUICKSTART.md)), and nothing here
is needed to use FOLO. They are published because the numbers in [BENCHMARKS](../BENCHMARKS.md) and the choices in
[MODEL](../MODEL.md) cite them, negative results included.

Read them as notebooks. Paths such as `D:/folo-data` and `D:/folo-build` are folders on the maintainers' machine (the
one under [BENCHMARKS, Machine](../BENCHMARKS.md#machine)), `#N` is a pull request of this repository, short
hexadecimal names are its commits, and a "lane" is one agent's area of work in an optimisation round. A notebook
states what was true on its date; where a later one disagrees, the later one and BENCHMARKS are current.

| file | what it is |
|---|---|
| [PROCESS](PROCESS.md) | How the maintainers' agents work in T3 Code: orchestrator, implementer, play-tester/reviewer, and writer subagents |
| [BOAT](BOAT.md) | The CPU sandboxes those agents may offload work to |
| [DOGFOOD-1](DOGFOOD-1.md), [-2](DOGFOOD-2.md), [-3](DOGFOOD-3.md) | Reports from agents that used FOLO as a newcomer would (CLI, MCP, Studio, Desktop, the docs), with what got in their way; the fixes are in the [changelog](../../CHANGELOG.md) |
| [DOGFOOD-4](DOGFOOD-4.md) | The stress round of Studio and Desktop (2026-11-14 and 15): 16 agents each used one area as a hard user would and fixed what broke; the lane reports are indexed in [stress/](stress/README.md) |

## Optimisation notebooks

Two rounds ("hill-climbs") in which several agents each took one area, measured against the build before, and wrote
down their method, commands, A/B tables and what did not work. BENCHMARKS has the merged results
([Hill-climb](../BENCHMARKS.md#hill-climb-2026-11-09), [Hill-climb 2](../BENCHMARKS.md#hill-climb-2-2026-11-12)).

| round 1 (2026-11-09) | round 2 (2026-11-12) |
|---|---|
| [GPU inference](hill/gpu-infer.md) | [GPU dispatch cost and the buffer pool](hill/r2-gpu-dispatch.md), [forward GEMM and conv kernels](hill/r2-gpu-gemm.md) |
| [The GPU training step](hill/gpu-train.md) | [The GPU training step](hill/r2-gpu-train.md) |
| [The CPU](hill/cpu.md) | [The CPU](hill/r2-cpu.md) |
| [Trainer data pipeline, host loop and validation](hill/trainer-data.md) | [Trainer start-up, validation and loader caches](hill/r2-trainer.md) |
| [Learning: accuracy per epoch, the `auto` optimizer rule](hill/learn.md) | [Learning: `auto` rules 4 and 5](hill/r2-learn.md) |
| | [Fold-in: the merged build measured again](hill/r2-fold-in.md) |

Round 4 (2026-11-14): [Dense output distillation from a frozen teacher, on the proxy](hill/r4-distill.md).

Round 5 (2026-11-17; round summary in [BENCHMARKS, Hill-climb 5](../BENCHMARKS.md#hill-climb-5-2026-11-17)):
- [Distillation on full COCO: a gated accuracy idea, the fused distillation kernel and the long run](hill/r5-accuracy.md)
- [Int8 inference on the CPU](hill/r5-cpuq8.md)
- [GPU training glue, backward GEMM shapes, BatchNorm in the conv](hill/r5-gputrain.md) (its PR #294 was closed,
  not merged)
- [The trainer's host side: a sync-free AMP optimizer window, a decode-once mosaic window](hill/r5-trainer.md) (its
  PR #298 was closed, not merged)

Wave 6 (2026-11-20):
- [The data loader: faster training samples with the same bytes](hill/r6-loader.md)
- [Fine-tuning defaults on small datasets: the EMA of short runs, the benchmark's selection, starting
  weights](hill/r6-finetune.md) (no default changed)
- [folo-s continued on full COCO for 48 epochs: val2017 AP 0.4162 to 0.4224, and a mixed fine-tuning
  guard](hill/r6-long-s.md) (weights not published)
- [A consensus score at decode: the gate on folo-n v5 and folo-s, shipped as an opt-in](hill/r6-consensus.md)
- [Feature imitation and foreground-gated distillation at the fork gate](hill/r6-distill.md) (no arm passed, no code
  shipped)
- [Why a continued model first gets worse: the warm-up and EMA arms on the proxy and on full
  COCO](hill/r6-continue.md) (no arm passed, no code shipped)
- [Four trainer findings from a research report: one fixed, the rest match the reference or lost the
  gate](hill/r6-trainfix.md) (no default changed)

## Design review logs

The two apps were refined in rounds too: screenshots of every screen, two independent design reviews against a fixed
rubric, fixes, repeat. Their logs live with the apps: [Studio](../../web/notes/HILLCLIMB.md)
([rubric](../../web/notes/REVIEW_RUBRIC.md)) and [Desktop](../../apps/desktop/notes/HILLCLIMB.md)
([rubric](../../apps/desktop/notes/REVIEW_RUBRIC.md)).
