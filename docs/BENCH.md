# Vision benchmarks

A validation score tells you how a model does on images like the ones it trained on. A benchmark asks the
questions you care about before you rely on a model: does it still work in bad light, in murky water, far away?
Can you trust its confidence? Could it label new images well enough to train its successor?

FOLO lets you make such a benchmark from your own images and run **any vision model** on it: a FOLO model, a
coding agent (Claude Code, Codex) looking at the images, or a model that runs somewhere else and sends its
predictions.

```
labeled images ──► benchmark ──► entries ──► leaderboard, charts
 (a dataset split,   (frozen images     (FOLO models,      (score, five axes,
  or a reviewed       + truth, a plan    agents, external    per condition, per class,
  labeling job)       of conditions)     predictions)        per image)
                                              │
                                              └──► self-improvement loop: the model labels a pool,
                                                   a student trains on those labels, the benchmark
                                                   grades the student
```

This is not the speed benchmark (`folo bench`, [BENCHMARKS.md](BENCHMARKS.md)): that one times a model, this one
grades what a model sees.

## What a benchmark is
- **Images and truth, frozen.** A benchmark copies its images and their true boxes when it is made, from a split of
  a dataset or from the labeled images of a labeling job ([LABELING.md](LABELING.md)). They never change after
  that, whatever happens to the source. Up to 5000 images; when the source has more, a seeded sample.
- **A plan of conditions.** Each condition is a deterministic change to every image that keeps its size (the
  catalog is below), at severities 1 to 5. The plan says which conditions run, at which severities, and how much
  each counts. It can change only while the benchmark has no entries: after that the benchmark is a fixed test.
- **A fingerprint.** A hash of the images, the truth, the plan and the seed. Two entries compare when their
  fingerprints agree.

Keep benchmark images out of training. A benchmark made from a dataset's `val` or `test` split is held out from a
run that trains on its `train` split; one made from the split a model trained on measures memory, not sight.

## Conditions
Every condition takes an image and gives back an image of the same size, the same way every time (the
benchmark's seed, the image and the severity decide where noise falls). Severity 1 is barely visible; severity 5
is severe, and a person can still do the job.

<!-- conditions:begin (crates/folo-bench/src/conditions.rs is the source; its test keeps this table in step) -->
| id | Name | Group | What it does | Parameter | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| `low_light` | Low light | light | Darkens the scene as a short exposure does and adds the sensor noise that comes with it: dusk, indoors, night. | gain in linear light, noise `n` (σ = n·√(signal + 0.004)) | ×0.6 n 0.02 | ×0.4 n 0.03 | ×0.25 n 0.04 | ×0.15 n 0.05 | ×0.08 n 0.06 |
| `overexposure` | Overexposure | light | Brightens the scene as a long exposure does, clipping highlights to white: glare, a bright sky, a flash too close. | gain in linear light, clipped | ×1.3 | ×1.6 | ×2 | ×2.5 | ×3.2 |
| `low_contrast` | Low contrast | light | Pulls every pixel toward the image's mean grey: flat light, an overcast day, a washed-out screen. | share of the deviation from the mean kept | 0.75 | 0.58 | 0.45 | 0.34 | 0.25 |
| `defocus` | Defocus | blur | Blurs the image as a lens out of focus does. | Gaussian σ, px at 640 | 1 | 2 | 3 | 4.5 | 6 |
| `motion_blur` | Motion blur | blur | Smears the image along one direction, as a moving camera or subject does during the exposure. | length, px at 640 | 4 | 8 | 13 | 19 | 26 |
| `sensor_noise` | Sensor noise | noise | Adds grain to every pixel and channel, as a small or hot sensor does. | σ, of 255 | 5 | 12 | 22 | 34 | 50 |
| `haze` | Haze | medium | Scatters light: blends the scene toward a bright veil and softens detail, as fog, smoke or murky water do. | transmission `t`, blur σ px at 640 | t 0.82 σ 0.4 | t 0.66 σ 0.8 | t 0.52 σ 1.2 | t 0.4 σ 1.6 | t 0.3 σ 2 |
| `color_cast` | Colour cast | medium | Absorbs red and tints the scene blue-green, as water does with depth (or tinted light). | red kept in linear light | ×0.8 | ×0.6 | ×0.45 | ×0.33 | ×0.22 |
| `jpeg` | JPEG compression | digital | Re-encodes the image as a low-quality JPEG, with the blocks and smeared colour of strong compression. | quality | 40 | 20 | 12 | 8 | 5 |
| `low_resolution` | Low resolution | digital | Downscales the image and scales it back up, so fine detail is lost: a cheap camera, a digital zoom. | downscale factor at 640 (at least 1) | 1.5× | 2× | 3× | 4× | 6× |
| `distance` | Distance | geometry | Shrinks the whole scene onto a canvas of the same size, so every object is smaller, as if seen from farther away; the true boxes move with it. | scale of the scene | ×0.8 | ×0.65 | ×0.5 | ×0.375 | ×0.25 |
| `occlusion` | Occlusion | geometry | Covers part of every object with a flat patch, as something in front of it would; the true boxes stay. | share of each box covered | 10% | 20% | 30% | 40% | 50% |
<!-- conditions:end -->

The image a contestant gets under a condition is a JPEG at quality 95 of the changed pixels, the same bytes for
every contestant: `GET /api/bench/benchmarks/{id}/images/{image_id}?condition=haze&severity=3`. Unchanged images
are served as their original files.

## The score
An entry gets one number from 0 to 100, the weighted mean of five axes, each between 0 and 1:

| Axis | Weight | What it asks | How it is computed |
|---|---:|---|---|
| Accuracy | 35% | Does it find the objects, with tight boxes? | mAP@0.5:0.95 on the unchanged images ([METRICS.md](METRICS.md)) |
| Robustness | 25% | How much of that survives hard conditions? | the weighted mean, over the plan's conditions, of the accuracy kept |
| Self-labeling | 20% | Could its own labels train a model? | the share of all objects it labels while at least 9 in 10 of its labels are right |
| Calibration | 10% | Can you trust its confidence? | 1 − expected calibration error |
| Localization | 10% | How tight are its correct boxes? | mean IoU of correct detections, rescaled from 0.5..1 to 0..1 |

`score = 100 · Σ weight · axis / Σ weight`, over the axes that were measured. An axis that could not be measured
(calibration for a contestant without confidences, robustness for an entry that ran no conditions) is left out
and the entry is marked **partial**, as is an entry that ran a part of the images or of the plan's conditions:
it ranks on the leaderboard with a mark, because its number compares only loosely with complete entries. (A
benchmark whose plan has no conditions has no robustness axis for anyone; its entries are not partial for that.)

Speed is reported for FOLO models (milliseconds per image on the server's device) and is not part of the score:
it depends on the machine, and an external model's cannot be measured here.

### Definitions
All of this is computed in `crates/folo-bench`; every number is tested against a hand-worked case.

**Matching.** Wherever "right" is used below: on the unchanged images, per image and class, detections in
descending confidence are matched one by one to the still unmatched true box of that class with the highest IoU,
if that IoU is at least 0.5 (the matching of AP at IoU 0.5). A matched detection is right, an unmatched one wrong.

**Accuracy.** `mAP@0.5:0.95` exactly as [METRICS.md](METRICS.md) defines it (COCO's 101-point AP, class mean
over the classes that have true boxes). Its 95% interval is a bootstrap over images: 200 resamples of the images
with replacement (seeded by the benchmark), the 2.5th and 97.5th percentile of their mAP@0.5:0.95. Two entries
whose intervals overlap are not clearly different on this benchmark; more images make the interval narrower.
Also reported: mAP@0.5, mAP@0.75, mAP for small / medium / large objects (COCO's area ranges), per-class AP, and
precision / recall / F1 at the confidence with the best F1.

**Robustness.** For condition `c` at severity `s`, `kept(c, s) = min(1, mAP(c, s) / mAP(unchanged))` with
mAP@0.5:0.95 on both sides. A condition's value is the mean of `kept` over its severities; the axis is the mean of
the conditions' values weighted by the plan's weights. When the unchanged mAP is 0 the axis is 0. A geometric
condition moves the true boxes with the image.

**Self-labeling.** Keep the detections with confidence at least `t`. Their *precision* is the share that is
right, their *recall* the share of all true boxes they match (over all classes together, not a class mean: this
is about a pile of labels, not about each class). The threshold `t*` is the lowest `t` on the grid
`0.00, 0.01, ..., 1.00` with precision at least **0.9** (and at least one detection kept); the axis is the
recall at `t*`, or 0 when no threshold reaches 0.9. A contestant without confidences has one operating point:
the axis is its recall if its precision is at least 0.9, else 0. The verdict is `ready` from 0.5 (it labels at
least half of all objects by itself at that precision), `partial` from 0.25, else `not_ready`. Also reported at
`t*`: the mean IoU of the right labels, labels per image, and the whole precision / recall curve.

This is a measurement on images whose truth is known. Whether training on such labels helps is what a
[self-improvement loop](#self-improvement-loops) measures.

**Calibration.** The detections with confidence at least 0.1 go into ten bins of confidence, `[0, 0.1)` to
`[0.9, 1]`. `ECE = Σ (n_bin / n) · |mean confidence of the bin − share of the bin that is right|`; the axis is
`1 − ECE`. Not measured without confidences, or when no detection reaches 0.1.

**Localization.** The mean IoU of the right detections at or above the best-F1 confidence (all right detections
without confidences) with the true boxes they match; the axis is `(mean IoU − 0.5) / 0.5`, and 0 when nothing is
right.

### Reading it
- The score is a summary. Open the entry: the condition chart shows where a model breaks, the reliability chart
  whether to believe its 0.9, the label chart what a confidence threshold costs in coverage.
- Compare entries of one benchmark (one fingerprint). A score from another benchmark is another test.
- A contestant without confidences has an AP made of one precision / recall point. Compare it with scored
  entries on precision, recall and the self-labeling axis rather than on AP.
- With a few dozen images the intervals are wide. That is what they are for.

## Who can run
- **A FOLO model**: any model of the server, run as validation runs it (so its mAP on the unchanged images is the
  one `folo val` reports on them). Its classes are matched to the benchmark's by name (without case); a
  benchmark class the model does not know scores 0 and is listed on the entry.
- **A coding agent**: Claude Code or Codex, asked for boxes exactly as agent labeling asks
  ([LABELING.md](LABELING.md), same CLIs, same cost per call). Agents give no confidences. By default an agent
  runs the first 48 images, unchanged only: every condition is another call per image.
- **Anything else**: fetch the images, run your model, send the predictions.

```sh
api=http://127.0.0.1:7878/api/bench/benchmarks/$bench
curl -s "$api/images?limit=1000"                       # ids, sizes (and the truth, which your model must not see)
curl -s "$api/images/$image" -o clean.jpg              # the unchanged image
curl -s "$api/images/$image?condition=haze&severity=3" -o haze3.jpg
curl -s "$api/entries" -H 'Content-Type: application/json' -d '{
  "name": "my-detector",
  "contestant": { "kind": "external", "predictions": [
    { "image": "1", "boxes": [ { "class": 0, "score": 0.91, "x1": 12, "y1": 40, "x2": 208, "y2": 331 } ] },
    { "image": "1", "condition": "haze", "severity": 3, "boxes": [] } ] } }'
```

Boxes are in pixels of the image as it was served. Every unchanged image must be answered for; a condition is
either answered for every image at every planned severity or left out. `examples/bench_external.py` does all of
this for a model behind one Python function.

## Designing a plan with an agent
The default plan runs every condition at severities 1, 3 and 5 with equal weight. A benchmark for a camera in a
pond should weigh haze and a green cast, not JPEG artifacts. **Design with an agent** sends the benchmark's
description, its classes and a few of its images to Claude Code or Codex (the connection agent labeling uses),
and gets back a plan drawn from the catalog: which conditions, how severe, how much each counts, and why. It is
a proposal: you see it next to the current plan, change what you disagree with, and save it. One model request,
no tools (a second one only when the first answer cannot be read).

## Self-improvement loops
The self-labeling axis says how good a model's own labels are. A loop tests what they are worth:

1. The benchmark grades the teacher (generation 0).
2. The teacher labels a **pool** of images by itself: it keeps its detections from a confidence threshold up (by
   default the threshold `t*` its benchmark entry found). The pool is a dataset split or a labeling job's images;
   the pool's own labels, if it has any, are never read, and pool images that are also in the benchmark are left
   out.
3. A student trains on those labels alone: the pool is split into train and val, both with the teacher's labels,
   so no true label enters training or the choice of the best checkpoint. The student can be a larger
   architecture than the teacher.
4. The benchmark grades the student, which becomes the teacher of the next generation.

The verdict compares the last generation with the first teacher on mAP@0.5:0.95 of the unchanged images:
**improved** when it is above the teacher's 95% interval, **regressed** when below, **held** otherwise. A loop
that improves over several generations is a model that makes its own training data for a better model: recursive
self-improvement, at the size of your pool and your benchmark. One that holds or regresses says the labels were
not good enough, or the pool not new enough, and the charts say which.

What it does not show: a loop is as honest as its benchmark is held out. If the teacher trained on the benchmark's
images, every generation inherits that.

## In Studio
**Benchmarks** in the sidebar.

## Limits
- Detection only (boxes): no masks, keypoints or classification benchmarks yet.
- Up to 5000 images and 80 classes per benchmark.
- `folo serve --mock` has no benchmarks. Studio's own demo mode (`?mock=1`) simulates them for a look around.
- Not yet in FOLO Desktop, the `folo` CLI or the MCP server: use Studio or the HTTP API.
