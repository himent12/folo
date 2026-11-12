# folo-ui: what FOLO Desktop costs, against the GPUI app

Measured 2026-11-11 on our test machine for the pull request that replaces GPUI 0.2.2 with folo-ui. "folo-ui" is
`desktop/fw-finish` at its final head (the merged `desktop/framework` with the review fixes, plus the finish);
"GPUI" is a reference build of the GPUI app, `folo-desktop-gpui.exe` (`origin/main` at
`ee846c88` with the baseline's off-screen and trace patch, rendering untouched).

## The machine and the load

- Intel i7-14700KF (28 logical cores), 32 GB, NVIDIA RTX 4080 SUPER, Windows 11.
- One 3840 x 2160 display at 150 % (scale 1.5), 240 Hz (Windows reports 239).
- Background load during the runtime measurements: 1 to 11 % of the whole machine (sampled before and after
  each run), no rustc running; the busiest other processes were PowerShell, Chrome and the system, each under
  1 % of the machine. The machine is shared with other agents and its owner, so absolute numbers swing: the two
  builds were measured alternately, three rounds each, and the tables give the median of the three rounds
  (each round's own number is itself a median where the row says so), with the range of the three after it.

## How to rerun

From `apps/desktop`, PowerShell, with a release build of each:

```powershell
# Runtime: start-up, memory, CPU and GPU (about three and a half minutes a run; alternate the builds)
foreach ($i in 1..3) {
    tools\perf.ps1 -Exe <folo-ui exe> -Label folo-ui-$i -Out perf\folo-ui-$i.json
    tools\perf.ps1 -Exe <GPUI exe> -Label gpui-$i -Out perf\gpui-$i.json
}
# Frames (bash; build with --features bench-alloc for the allocation counts)
bash tools/bench-frames.sh <exe built with bench-alloc>
# The renderer
cargo bench -p folo-ui --bench raster
# Build: from an empty target dir, jobs = 8 (.cargo/config.toml)
cargo build --release --locked; cargo build --profile dist --locked
cargo tree -e normal --prefix none | sort -u | wc -l          # crate count, as the GPUI baseline counted it
```

Every launch is off screen (`FOLO_UI_OFFSCREEN=1`) at `--width 1280 --height 800`, with a scratch
`FOLO_DESKTOP_STATE`. `perf.ps1` writes everything it measured, the machine's load included, as JSON.

## Start-up

Twelve launches of `--mock` per run; the first is reported as cold (the files were in the OS cache from earlier
runs) and the median is over the other eleven. Milliseconds since the process was created.

| | folo-ui | GPUI |
| --- | ---: | ---: |
| first render (the root view's first `render`) | 36.9 (36.1-37.5) | 361.6 (315.4-411.2) |
| first frame (the first present) | 70.5 (67.9-138.0) | 441.8 (352.1-477.1) |
| window visible (polled from outside every ~10 ms) | 43.6 (43.4-43.9) | 396.5 (324.3-419.7) |
| cold: first render | 129.4 (121.6-186.1) | 664.7 (646.9-960.2) |
| cold: first frame | 183.4 (182.0-318.9) | 793.3 (684.9-1017.1) |

## Memory

Six seconds after the window appears.

| page | | folo-ui | GPUI |
| --- | --- | ---: | ---: |
| Overview (`--mock`) | working set MB | 64.8 (64.0-65.8) | 89.6 (85.7-90.3) |
| | private MB | 47.7 (47.5-48.4) | 153.2 (146.2-154.6) |
| | threads / handles | 29 / 254 | 73 / 633 |
| | GPU dedicated / shared MB | 0 / 0 | 93.1 / 3.4 |
| Datasets (`--open datasets`) | working set MB | 60.2 (58.9-60.3) | 93.5 (91.6-93.6) |
| | private MB | 43.8 (43.4-43.9) | 159.7 (158.4-159.8) |
| | threads / handles | 37 / 270 | 75 / 638 |
| | GPU dedicated / shared MB | 0 / 0 | 52.1 / 1.9 |
| a dataset (`--open dataset:ds_8f2c`) | working set MB | 58.3 (57.4-58.5) | 93.2 (17.1-93.7) * |
| | private MB | 41.8 (41.8-42.3) | 154.9 (153.4-155.4) |
| | threads / handles | 39 / 274 | 77 / 646 |
| | GPU dedicated / shared MB | 0 / 0 | 48.1 / 2.2 |
| the live run (`--open run:run_a19f`) | working set MB | 69.0 (69.0-69.9) | 99.6 (99.0-100.3) |
| | private MB | 52.8 (52.8-53.6) | 165.5 (156.1-166.6) |
| | threads / handles | 25 / 246 | 69 / 618 |
| | GPU dedicated / shared MB | 0 / 0 | 89.1 / 12.0 |

\* One GPUI round read 17.1 MB: Windows had trimmed that process's working set at that moment (its peak was
92.9 MB).

## CPU and GPU over 30 seconds

Starting six seconds after the window appears. CPU seconds are the process's user + kernel time; GPU load is the
sum of the process's GPU engine counters, averaged over 30 one-second samples. folo-ui has no GPU engine
counters at all: the UI never touches the GPU.

| page | | folo-ui | GPUI |
| --- | --- | ---: | ---: |
| static: Settings (`--mock --open settings`) | CPU s | 0.34 (0.31-0.36) | 0.91 (0.83-1.16) |
| | % of one core | 1.2 | 3.0 |
| | GPU engines avg / max % | none | 0.40 / 0.45 |
| live: a training run (`--mock --open run:run_a19f`) | CPU s | 2.64 (2.33-2.91) | 31.19 (30.94-31.63) |
| | % of one core | 8.8 | 103.0 |
| | GPU engines avg / max % | none | 33.9 / 36.3 |
| live, frozen (`--scenario run-live`: the same page, no new data) | CPU s | 1.08 (0.14-1.20) | 31.84 (30.95-31.88) |
| | % of one core | 3.6 | 103.1 |
| | GPU engines avg / max % | none | 34.1 / 36.4 |

GPUI redraws the whole window at the display's 240 Hz while anything animates (the live run's breathing status
dot), on the GPU. folo-ui re-fades only the dot's opacity group at each refresh and repaints its tiles; a full
rebuild happens only when data arrives. Windows charges CPU time to a thread in clock-tick samples, so a process
that wakes often for a few microseconds reads noisily: the frozen page read 0.14 s in one round and 1.2 s in the
others, with the same frames.

### The live page's frames (folo-ui, `FOLO_UI_TRACE=frames`)

Over 30 s of the `--mock` live run page, by cause (each frame's own phase times, summed; two runs of this branch
just before the review fixes merged, `a4684e41`, with a scratch timestamp on each trace line to tell the causes
apart; the review fixes change no frame path):

| frames | per 30 s | each | total |
| --- | ---: | ---: | ---: |
| full rebuilds by the 10 Hz data pump (only when events arrived) | 188-218 | 3.5-3.9 ms | 0.72-0.85 s |
| a second rebuild right after a pump frame (data fetched on that notify, such as an epoch's samples) | 16-20 pairs | | 0.11-0.13 s |
| chart easing (240 ms at the refresh rate when an epoch adds points; about 40 frames) | 2 bursts | | 0.26-0.28 s |
| the breathing dot's ticks (no rebuild: the opacity group re-faded, its tiles repainted) | 1822-2094 | 0.12-0.14 ms | 0.22-0.28 s |

Every full frame there presented a change. One redraw that did not need to happen is gone: the log under the
fold follows its new lines, and its scroll thumb asked for 800 ms of full rebuilds after each one although it
was clipped away; it now asks for nothing while out of view (48 pairs and several 11-16-frame bursts per 30 s
before, measured the same way). On a page that shows no live data (Settings with the mock's run training), the
pump still rebuilds about 9 times a second, 1.8 ms each, and most of those frames present nothing: the
sidebar's progress bar moves by a fraction of a pixel, which only drawing the frame can tell (about 0.5 s of
CPU per 30 s).

## Frames (`tools/bench-frames.sh`)

Headless, settled, then 100 full rebuilds of an unchanged screen ("full": render, layout, prepaint, paint, and
the renderer finding nothing changed) and 100 animation ticks one 240 Hz refresh apart ("tick"); medians in
microseconds, allocations per full frame (`--features bench-alloc`), heap after the bench. Final build, same day.

| screen | window @ scale | prims | elements | build | layout | prepaint | paint | finish | raster | full (min) | allocs | tick (min) | heap MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overview | 1280x800@1.5 | 962 | 616 | 353 | 194 | 76 | 234 | 49 | 34 | 944 (920) | 4896 | 33 | 43.6 |
| run-live | 1280x800@1.5 | 958 | 1191 | 637 | 248 | 138 | 649 | 90 | 52 | 1821 (1768) | 9564 | 43 | 43.3 |
| dataset | 1280x800@1.5 | 976 | 533 | 234 | 69 | 232 | 204 | 48 | 40 | 829 (800) | 5944 | 0 | 49.0 |
| models | 1280x800@1.5 | 801 | 270 | 162 | 46 | 32 | 96 | 27 | 22 | 386 (376) | 2414 | 0 | 30.8 |
| runs | 1280x800@1.5 | 913 | 415 | 234 | 68 | 52 | 127 | 37 | 27 | 548 (528) | 3188 | 27 | 31.5 |
| playground | 1280x800@1.5 | 826 | 311 | 207 | 85 | 41 | 122 | 28 | 29 | 514 (495) | 2730 | 0 | 41.2 |
| settings | 1280x800@1.5 | 1275 | 531 | 300 | 122 | 67 | 182 | 47 | 42 | 763 (721) | 3754 | 0 | 33.1 |
| overview | 2560x1440@1.5 | 1478 | 569 | 335 | 196 | 72 | 250 | 48 | 63 | 970 (948) | 4682 | 57 | 117.4 |
| run-live | 2560x1440@1.5 | 2181 | 1191 | 646 | 256 | 139 | 824 | 97 | 105 | 2073 (2033) | 9904 | 88 | 115.8 |
| dataset | 2560x1440@1.5 | 2402 | 1483 | 331 | 79 | 1165 | 635 | 141 | 120 | 2487 (2405) | 21649 | 0 | 188.7 |
| models | 2560x1440@1.5 | 837 | 291 | 173 | 50 | 35 | 104 | 29 | 32 | 425 (409) | 2539 | 0 | 99.6 |
| runs | 2560x1440@1.5 | 1001 | 435 | 339 | 86 | 59 | 138 | 56 | 39 | 719 (627) | 3370 | 36 | 100.3 |
| playground | 2560x1440@1.5 | 839 | 328 | 210 | 91 | 43 | 129 | 30 | 42 | 547 (525) | 2768 | 0 | 119.8 |
| settings | 2560x1440@1.5 | 1990 | 531 | 300 | 120 | 68 | 204 | 50 | 71 | 817 (793) | 3754 | 0 | 101.7 |

A tick of 0 means the screen has nothing animating. The heap at 2560 x 1440 includes the 3840 x 2160
framebuffer (33 MB) and the larger image caches of the dataset grid.

## The renderer (`cargo bench -p folo-ui --bench raster`)

A busy synthetic screen, release build, the machine quiet.

| | 1920 x 1200 (6720 primitives) | 3840 x 2160 (24382 primitives) |
| --- | ---: | ---: |
| cold first frame (pool) | 6.27 ms | 23.10 ms |
| full repaint, 1 thread | 4.31 ms | 17.28 ms |
| full repaint, pool of 8 | 1.06 ms | 4.12 ms |
| bin + hash | 0.10 ms | 0.25 ms |
| unchanged frame | 0.10 ms | 0.25 ms |
| small change (200 x 40) | 0.11 ms (3 tiles) | 0.27 ms (4 tiles) |
| scroll | 0.76 ms (70 % of tiles) | 3.36 ms (82 % of tiles) |
| renderer caches | 6.1 MB | 17.8 MB |

## Build

| | folo-ui | GPUI |
| --- | ---: | ---: |
| release exe | 34,071,552 bytes (32.5 MiB) | 41,692,672 bytes (39.8 MiB) |
| dist exe (thin LTO, 1 codegen unit: what ships) | 30,390,272 bytes (29.0 MiB) | 33,925,120 bytes (32.4 MiB) |
| clean `cargo build --release --locked` (empty target dir, jobs = 8, no other rustc) | 140 s | 194 s (188 s in the baseline report, another day) |
| crates: `cargo tree -e normal --prefix none \| sort -u` lines | 215 | 597 |
| distinct crate versions / names | 168 / 162 | 465 / 423 |

Both builds were made from an empty target dir one after the other, the GPUI one from `ee846c88` (without the
baseline's patch; its release exe is 41,679,360 bytes). The exe includes the app's own inference stack (folo-vision on wgpu) in both builds: the UI is not the only GPU
user in the binary, only the only one at run time while no model runs.

## Not measured

- A truly maximized 4K window on the real desktop: every launch here is off screen (or on a private desktop
  nobody sees) at 1280 x 800; the frame bench's 2560 x 1440 rows and the renderer's 3840 x 2160 rows are the
  nearest numbers.
- Linux and macOS: nothing here ran there (CI builds and tests the app headless on Linux; macOS is type-checked
  only).
- The GPUI numbers come from the same machine, the same day and the same tools, but from a build with the
  baseline's patch (off-screen placement and the trace lines), not the shipped GPUI app.
