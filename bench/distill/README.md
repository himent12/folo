# distill: what a folo-s teacher can give folo-n

Two CPU-only measurements that motivated the dense distillation of hill-climb 4
([docs/notes/hill/r4-distill.md](../../docs/notes/hill/r4-distill.md)). Both read hill-climb 3's candidate dumps of
val2017 (`cargo run -p folo-vision --example candidate_dump`, see [`../fusion/`](../fusion/)): the landed folo-n
(`D:/folo-data/runs-hill3/fusion/dumps/land-val`, = folo-n-coco-v3) and folo-s-coco
(`D:/folo-data/runs-hill3/fusion/dumps/s-val`), both at 640 px. The paths are constants at the top of each script;
they import `coco.py`, `dump.py` and `post.py` from `../fusion/`. Packages: numpy, pycocotools.

| script | what it measures | output |
|---|---|---|
| [`swap.py`](swap.py) | At anchors that are candidates (class-max score > 0.001) in both dumps: the student's scores with the teacher's boxes, halfway boxes, the teacher's scores with the student's boxes; AP of each after the default post-processing (hard NMS 0.7, 300 detections); per-side DFL statistics | stdout, one `AP ...` line per configuration |
| [`rank.py`](rank.py) | Gate step 0 of the rank term (hill-climb 5, [r5-accuracy.md](../../docs/notes/hill/r5-accuracy.md)): on the student's candidates, the teacher's scores (value swap), the student's own scores in the teacher's within-class order over the whole set (rank swap) or per 16 images, the swaps above / below a student score of 0.1; the weighted Pearson correlation of the two models' logits as the rank term computes it | stdout |
| [`nvs.py`](nvs.py) | Per non-crowd truth, each model's top-scoring same-class detection at IoU >= 0.5 ("primary"): recall, edge error and IoU by object size; AP with the other model's primary boxes or scores swapped in on the truths both find | stdout; `nvs_rows.npy` in the working directory |

Run from the repository root, e.g. `python bench/distill/swap.py > swap.log` with numpy and pycocotools importable.
The logs behind the notebook's numbers are `D:/folo-build/hill4/counsel-a-swap.log` and
`D:/folo-build/hill4/counsel-b/nvs.log` (the counsel reports of hill-climb 4, run with the scripts' original paths,
which pointed at the same `fusion/` modules in the counsels' worktrees of origin/main 1436ad68).
