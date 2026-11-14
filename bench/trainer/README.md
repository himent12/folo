# bench/trainer: timing `folo train` and `folo val`

A/B timers for whole `folo` processes: each runs two or more `folo` binaries (say `main` and a branch) alternately,
round by round, a fresh process per run, so that load from other work on a shared machine falls on both. They time
the trainer protocol of docs/BENCHMARKS.md ([Method](../../docs/BENCHMARKS.md#method), "Trainer"; row 10 of the
FOLO targets table), validation, start-up and the first epoch. The results are in the hill-climb sections of
docs/BENCHMARKS.md, for example [The trainer (row 10)](../../docs/BENCHMARKS.md#the-trainer-row-10) and
[Validation](../../docs/BENCHMARKS.md#validation), and in
[docs/notes/hill/r2-trainer.md](../../docs/notes/hill/r2-trainer.md).

| file | what it does |
|---|---|
| [`ab.py`](ab.py) | `ab.py train` / `ab.py val`: per run the per-epoch training img/s, each epoch's duration, the process wall time and the CPU cores used by the run and by everything else; then median, best and spread per binary |
| [`startup.py`](startup.py) | process start to the first `step` record at or past `--until-step` of `folo train --json`, then kills the process; writes every line it received with its arrival time to `<out>/<tag>.log`, prints a JSON row per run, then the median and range per binary |
| [`first_epoch.sh`](first_epoch.sh) | one epoch: `$BASE`, then `$NEW` with a deleted and with a warm GPU pipeline cache, per round; prints epoch 1's `duration_s`, the process wall and the compile timer's line |
| [`cold_start.sh`](cold_start.sh) | first-ever runs: an empty FOLO cache and an empty NVIDIA shader cache (`__GL_SHADER_DISK_CACHE_PATH`) per run, and `$NEW` with its kernel list kept but no pipeline cache |
| [`warm_start.sh`](warm_start.sh) | second starts of several binaries, each with its own cache folder filled by an untimed run; `train` (one epoch) or `predict` (one image) |
| [`slow_write.sh`](slow_write.sh) | the checkpoint writer under a slow disk: sync, async and no delay, 4 epochs each; needs a binary built with `slow_write.patch` |
| [`slow_write.patch`](slow_write.patch) | temporary instrumentation of `crates/folo-vision/src/train/writer/mod.rs` (`FOLO_TMP_WRITE_DELAY_MS`, `FOLO_TMP_SYNC`) for `slow_write.sh`; apply, build, then revert, as the script's header shows |

## What it needs

- Python for `ab.py` and `startup.py` (standard library only); bash with `timeout`, `awk` and `date +%s.%N` for the
  shell scripts.
- The `folo` binaries to compare, built with `cargo build --release -p folo-cli`.
- A GPU: the shell scripts pass `--device gpu:0`, and so do the protocols the Python scripts are used with.

**These scripts are written for the maintainers' Windows machine.** The shell scripts hard-code
`--data D:/folo-bench/data` (`warm_start.sh` also defaults `IMG` to `D:/folo-bench/data/images/val` and `WEIGHTS` to
`D:/folo-weights/folo-n-coco.folo`), and `warm_start.sh` reads the machine's load through `powershell`. `ab.py`'s CPU
accounting uses the Win32 API and leaves those columns empty elsewhere. The Python scripts take the dataset and the
binaries as arguments.

## Run it

Everything after `--` goes to `folo train` / `folo val` unchanged:

```sh
python bench/trainer/ab.py train --bin base=<main>/folo.exe --bin new=<branch>/folo.exe --rounds 3 -- \
    --data <synthetic set> --model folo-n --epochs 2 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 \
    --device gpu:0 [--amp]
python bench/trainer/ab.py val --bin base=... --bin new=... --rounds 3 -- \
    --model <folo-n-coco.folo> --data <coco dir> --device gpu:0
python bench/trainer/startup.py --bin base=... --bin new=... --rounds 3 --until-step 4 -- \
    --data <coco dir> --model folo-n --epochs 1 --batch 16 --imgsz 640 --device gpu:0
BASE=<main folo.exe> NEW=<new folo.exe> bench/trainer/first_epoch.sh ""        # or "--amp"
BASE=... NEW=... OUT=<dir> bench/trainer/cold_start.sh ""
OUT=<dir> bench/trainer/warm_start.sh train base=<path>/folo.exe new=<path>/folo.exe
BIN=<patched folo.exe> bench/trainer/slow_write.sh
```

`ROUNDS` (default 3) sets the rounds of the shell scripts. Output goes to `--out` / `$OUT`, by default under
`runs-bench/` (ignored by git); `cold_start.sh` and `warm_start.sh` need `OUT`. There are no tests in this folder.

[Up: bench/](../README.md)
