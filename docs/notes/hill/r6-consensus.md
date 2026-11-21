# Wave 6, lane consensus: a score from the cluster NMS throws away (lab notebook)

Lane `consensus` of wave 6 (2026-11-21), CPU only. **Result: the gate written beforehand passes on all four conditions,
and the rule ships as an opt-in, `--score consensus`, off by default.** On the published folo-n v5 it adds +0.0021 COCO
AP (+0.0019 on top of the box fusion). On folo-s it adds +0.0012 (+0.0011), a smaller and weaker result: the rule's
radius was read on these same weights, and folo-s loses on large objects. Scratch, scripts and evaluation files:
`D:/folo-wave6/lanes/consensus/`.

Terms: **AP** is COCO's mAP50-95 (AP_S, AP_M and AP_L are the same for small, medium and large objects). **B0** is the
default post-process (conf 0.001, NMS IoU 0.7, 300 detections per image). **B\*** is the shipped box fusion
(`--fusion dfl --iou 0.65 --max-det 1000`, fusion IoU 0.75).

## The rule

For each box `k` NMS keeps, over the candidates that entered NMS (docs/MODEL.md §9.4):

    M_k  = sum of s_j over candidates j of the same class with IoU(b_k, b_j) >= 0.75   (j = k included)
    s'_k = s_k * min(M_k / s_k, 10)^0.5 / 10^0.5

The detections are then sorted by `s'`. Boxes, classes and which boxes are kept do not change. The one-to-many head
trains several anchors per object, so a real object tends to light up several candidates that agree, and a false
positive is more often one anchor alone.

## What was fixed in advance, and what was not

`D:/folo-wave6/lanes/consensus/GATE.md` was written before any score was read. It fixes the two weights by sha256, the
two post-processes, the bootstrap, and these conditions:

1. folo-n v5, B0 + consensus − B0: at least +0.0010, interval's lower end above 0.
2. folo-n v5, B\* + consensus − B\*: at least +0.0010, interval's lower end above 0.
3. folo-s, B0 + consensus − B0: at least 0 (point estimate).
4. folo-s, B\* + consensus − B\*: at least 0 (point estimate).

**The setting was not fresh for folo-s.** The research report that proposed the rule
(`D:/folo-wave6/research/formula.md`, F3) measured it on folo-n v3 and on the **same published folo-s file** used here.
There, the τ 0.70 fixed in advance gave folo-s +0.0007 [−0.0005, +0.0017], not significant; τ 0.75 gave +0.0012 and was
"read on val". This lane's brief describes that measurement as on "older" weights, which holds for folo-n but not for
folo-s. So conditions 3 and 4 re-measure a number that already informed the choice of τ, and the folo-s gain is
optimistic by an amount not measured. Conditions 1 and 2 are on weights that played no part in the choice, though on
the same 5,000 images. τ, a and R were not changed in this lane.

## The gate

Dumps: `candidate_dump --device cpu`, f32, val2017, conf 0.001. Scoring: pycocotools. Differences: paired bootstrap over
the 5,000 images (`bench/fusion/bootstrap.py`, 1,000 resamples, seed 0, 95% percentile interval). Sanity check before
reading the gate: B0 gives 0.370077 on folo-n v5 (814,519 detections) and 0.416163 on folo-s (678,999; the published
GPU run has 679,000), which match docs/BENCHMARKS.md to four digits.

| condition | without | with | difference [95% interval] | P(≤ 0) | bar | verdict |
|---|---:|---:|---:|---:|---|---|
| 1. folo-n v5, B0 | 0.370077 | 0.372149 | **+0.0021 [+0.0015, +0.0027]** | 0.000 | ≥ +0.0010, lower end > 0 | pass |
| 2. folo-n v5, B\* | 0.372973 | 0.374877 | **+0.0019 [+0.0013, +0.0025]** | 0.000 | ≥ +0.0010, lower end > 0 | pass |
| 3. folo-s, B0 | 0.416163 | 0.417360 | **+0.0012 [+0.0002, +0.0021]** | 0.018 | ≥ 0 | pass |
| 4. folo-s, B\* | 0.418640 | 0.419755 | **+0.0011 [+0.0001, +0.0020]** | 0.022 | ≥ 0 | pass |

Against B0, the fusion and the rule together: folo-n v5 +0.0048 [+0.0038, +0.0058] (the fusion alone +0.0029 [+0.0022,
+0.0035]), folo-s +0.0036 [+0.0025, +0.0048] (the fusion alone +0.0025 [+0.0019, +0.0032]).

By size and threshold:

| configuration | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| folo-n v5, B0 | 0.3701 | 0.5250 | 0.4000 | 0.1789 | 0.4052 | 0.5396 |
| folo-n v5, B0 + consensus | 0.3721 | 0.5269 | 0.4035 | 0.1847 | 0.4099 | 0.5425 |
| folo-n v5, B\* | 0.3730 | 0.5305 | 0.4006 | 0.1829 | 0.4074 | 0.5423 |
| folo-n v5, B\* + consensus | 0.3749 | 0.5318 | 0.4039 | 0.1887 | 0.4118 | 0.5452 |
| folo-s, B0 | 0.4162 | 0.5798 | 0.4519 | 0.2289 | 0.4594 | 0.5845 |
| folo-s, B0 + consensus | 0.4174 | 0.5812 | 0.4544 | 0.2338 | 0.4619 | **0.5812** |
| folo-s, B\* | 0.4186 | 0.5840 | 0.4537 | 0.2312 | 0.4611 | 0.5874 |
| folo-s, B\* + consensus | 0.4198 | 0.5849 | 0.4560 | 0.2359 | 0.4634 | **0.5839** |

**The downside: folo-s AP_L falls**, by 0.0033 on B0 and 0.0035 on B\*, while folo-n's rises. The gain is on small and
medium objects. Why large objects lose on folo-s was not investigated.

## The shipped code against the gate's

The gate is Python on the dumps; the option is Rust in the NMS's last step. `folo val --score consensus --device cpu`
through pycocotools gives **all twelve COCO numbers with a difference of 0.00e+00** from the Python gate, and the same
detection counts, for B0, B0 + consensus and B\* + consensus on both models.

## With the setting off

**Output.** `folo val --save-json` of folo-n v5 from this branch (with the review's fixes) and from `main` at `53d28165`
are byte for byte the same file (sha256 `ba86dad8…d336`). Tests also check this bit for bit on every NMS path. With the
setting on, the same build's four `--score consensus` files (both models, with and without the fusion) are byte for
byte the ones the Rust = Python check above was run on, so the review's fixes did not move a gate number.

**Time.** In the code, off adds one comparison per image, before the rule's branch. The timing below is of the first
version of that step, which also tested an `Option` once per kept detection; the review's fix removed that test, and the
timing was not repeated. One timing, single-threaded, on real
val2017 candidates of folo-n v5 (first 1,500 images, 24 alternated process runs of each binary in one `cpu` lock hold,
fastest of 40 passes per run; the job hit its time limit after 22 complete pairs; 22% busy when it started):

| setting | main, ms per image: median (min–max) | off, this branch | paired ratio, median |
|---|---:|---:|---:|
| val | 0.0768 (0.0729–0.1116) | 0.0763 (0.0725–0.1166) | 1.005 |
| fusion's NMS | 0.0722 (0.0709–0.1177) | 0.0720 (0.0706–0.0939) | 1.006 |
| predict | 0.0033 (0.0029–0.0047) | 0.0033 (0.0029–0.0048) | 1.010 |

All three are ties within this PC's noise. Earlier attempts were noisier and are not evidence either way: the `post`
bench under 36–79% load read new/base 1.113, 1.070, 1.157 and 0.998 with overlapping ranges, and the same single-thread
A/B with 5 passes per run at 26–94% load read 0.989, 0.980 and 1.040.

## What it costs with the setting on

Real val2017 candidates, first 2,000 images, 5 alternated rounds, 8 threads, medians on a shared PC (20–26% busy). Added
milliseconds per image:

| model, setting | median | mean | p99 |
|---|---:|---:|---:|
| folo-n v5, val | 0.040 | 0.104 | 0.810 |
| folo-n v5, fusion's NMS | 0.035 | 0.165 | 1.930 |
| folo-n v5, predict (conf 0.25) | 0.000 | 0.000 | 0.004 |
| folo-s, val | 0.025 | 0.084 | 0.745 |
| folo-s, fusion's NMS | 0.021 | 0.118 | 1.554 |
| folo-s, predict (conf 0.25) | 0.000 | 0.001 | 0.006 |

## What this does not show

- **One split.** Every number is on val2017, the split τ 0.75 was read on. No held-out set was used.
- **Validation's settings only.** The cluster holds only candidates above `conf`, so the same box gets a different score
  at `predict`'s `conf` 0.25 than at 0.001. No gain is claimed at `predict`'s settings.
- **The score scale.** A lonely box is divided by 3.16, so thresholds, best-F1 confidence, precision, recall and the
  confusion matrix mean something else with the option on.
- **The real GPU was not run in this lane** (two other lanes were training on it). The rule runs on the host in the
  step every NMS path ends in. On WARP, Windows' software adapter, the device-pairs test passes with the rule on.
  `gpu_lazy_head`'s fusion test, which also checks that the rule changes scores and nothing else, crashes on WARP on
  `main` as well (segmentation fault, 2 of 2 runs on each side), so that test is left to CI's Windows runner.
- **Not offered** by the server, Studio, ONNX export or the wasm build.
