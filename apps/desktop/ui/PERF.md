# folo-ui: what FOLO Desktop costs, against the GPUI app

Part of folo-ui ([README.md](README.md)). A record of two measurements (the GPUI comparison, then the 2026-11-18
loading and frame work at the end); `apps/desktop/tools/perf.ps1` measures a
build again.

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

## 2026-11-18: loading and frames (pull request `desktop/perf-load`)

Same machine, shared with other agents, so every comparison alternates the two builds and reports the median
over rounds with the range. "Before" is `origin/main` at `c265b78b`; for the launch, it is that build plus the
trace points of this pull request (`FOLO_UI_TRACE=1`: `probe-none`, `server-started`, `first-lists`, `system`,
the image queue's first drain, and the `-frame` lines of the last three), which change nothing else. Release builds,
warm file cache. The tables below are the first measurement, of the pull request before its review fixes; the
fixes were measured again afterwards ("After the review fixes", below), on a busier machine, and the second
round's fixes against `origin/main` at `c39e1862` ("After the second review round", below), which are the
numbers of the branch as it is.

### Launch to data on screen

Off screen (`FOLO_UI_OFFSCREEN=1`), 1280 x 800, nothing on port 7878, so the app starts its own `folo serve`
(`FOLO_BIN` set to the matching build) on a data dir with two datasets (2000 and 500 images) and one run;
eight alternating rounds, milliseconds since the process was created.

| | before | after |
| --- | ---: | ---: |
| first frame | 31.4 (26.2-57.7) | 31.6 (25.4-34.9) |
| probe of port 7878 says nobody is there | 547.1 (540.2-584.3) | 41.9 (30.0-47.2) |
| `folo serve` listening | 731.7 (704.8-1350.0) | 68.8 (63.2-86.2) |
| runs, datasets and models on screen | 751.0 (718.4-1367.5) | 88.2 (80.0-104.7) |
| the system card (CPU, memory, GPUs) on screen | 1212.7 (1141.2-1858.5) | 379.8 (344.4-545.3) |
| the image queue first drained (on this server: the thumbnails on screen) | not traced | 103.2 (96.2-121.1) |

Where it went:

- The probe of `127.0.0.1:7878` waited for its 0.5 s connect timeout: Windows answers a connection to a closed
  port only after its SYN retries. A loopback address is now asked first with `SIO_TCP_INITIAL_RTO` set to no
  retries, which reports the refusal at once (`src/api/http/refused.rs`); anything else goes to the probe as
  before.
- `folo serve` set up sysinfo's CPU-usage counters (about 150 ms) before listening; they are now set up on a
  thread of their own, and the first `/api/system` waits for them and then out sysinfo's 200 ms minimum interval
  since (a reading taken at once said 100 %). `/api/system` itself ran the engine's capabilities (a GPU device,
  about 280 ms), then the adapter list (about 150 ms), then the CPU sample, one after the other; the first call
  now runs the three side by side, once per server, and later calls only read. `folo serve` listening: about 165 ms to 16 ms (six alternating
  rounds); the first `/api/system`: about 1420 to 820 ms (noisy).

### Frames (`--bench-frames 100`, headless, scale 1.5)

Five alternating rounds, microseconds, the median of each round's median; "scroll" is new (40 px wheel steps
over the window's centre, five down then five up), so it has no "before".

| screen | window | full before | full after | paint before | paint after | scroll after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| overview | 1280x800 | 837 (831-891) | 811 (771-1008) | 233 | 207 | 2061 |
| runs | 1280x800 | 740 (721-805) | 600 (566-654) | 281 | 161 | 586 |
| dataset | 1280x800 | 898 (876-932) | 818 (803-831) | 225 | 202 | 1962 |
| run-live | 1280x800 | 2218 (2085-2710) | 1809 (1742-2216) | 758 | 489 | 3596 |
| settings | 1280x800 | 1039 (955-1097) | 851 (826-924) | 206 | 192 | 1994 |
| overview | 2560x1440 | 1093 (1055-1302) | 1008 (950-1036) | 276 | 234 | 985 |
| runs | 2560x1440 | 899 (860-983) | 728 (673-752) | 310 | 189 | 684 |
| dataset | 2560x1440 | 2949 (2852-3295) | 2636 (2487-2888) | 723 | 634 | 7563 |
| run-live | 2560x1440 | 2541 (2365-2622) | 2296 (2166-2322) | 933 | 665 | 5728 |
| settings | 2560x1440 | 1001 (915-1049) | 918 (871-970) | 235 | 221 | 2826 |

Two changes made these:

- Built paths are kept by their input (`ui/src/path_cache.rs`): the charts and sparklines were tessellated
  again every frame. Alone (five rounds): run-live full frame 2145 to 1853 us at 1280 x 800 (paint 743 to
  489), 2306 to 2025 at 2560 x 1440, scroll 5903 to 5645. With the live pump on (the chart's data changes
  every tick) it is a tie.
- The segment heap (`assets/folo-desktop.manifest`): a frame makes thousands of small allocations. Alone
  (five rounds, 1280 x 800): overview 849 to 813 us, dataset 907 to 842, run-live 2387 to 2275, settings
  1070 to 873; at 2560 x 1440 dataset 2790 to 2480 and its scroll 7828 to 7382, run-live a tie. Working set and
  private bytes 5 s after launch equal or slightly lower; start-up unchanged.

Pixels: every scenario of `--list-scenarios` (90) rendered with `--shot` by both builds at 1280 x 800 @ 1 and
1920 x 1200 @ 1.5: 180 of 180 identical (`tools/parity.py --tol 0`). The release exe grew by 32 KB.

Tried and dropped: mimalloc (frames 15 to 23 % shorter, but private bytes 44 to 136 MB); an AVX2 build of the
renderer (a tie); starting the server's GPU device at start-up (about 25 ms on the system card).

### After the review fixes

The review fixes: the refused-at-once check only where the HTTP client connects directly (not through a proxy),
bounded by one 100 ms deadline and with its Winsock start-up released; the first CPU reading after its interval;
`/api/system`'s first parts prepared once per server as blocking tasks instead of threads per request; the path
cache counting what it holds and evicting in one pass. Measured again with three builds alternating: "before"
is `origin/main` at `1f2084ec` plus the trace points, "PR" the pull request before the fixes, "fixed" with them.
The machine was busier than for the tables above (other programs used about a third of the CPU), so every
number is higher; the comparison is within this run.

Launch, as above, eight rounds, ms since the process was created:

| | before | PR | fixed |
| --- | ---: | ---: | ---: |
| first frame | 42.0 (37.3-63.1) | 42.9 (39.1-61.9) | 43.4 (40.0-68.1) |
| probe of port 7878 says nobody is there | 565.9 (556.0-591.8) | 92.2 (54.7-225.9) | 88.5 (53.0-218.3) |
| `folo serve` listening | 767.2 (758.5-852.2) | 127.3 (80.2-279.5) | 128.0 (81.2-247.5) |
| runs, datasets and models on screen | 788.2 (778.5-868.7) | 147.4 (96.5-298.5) | 148.2 (98.3-270.3) |
| the system card on screen | 1318.0 (1278.8-1383.9) | 645.9 (480.9-819.5) | 581.0 (496.8-715.1) |
| the image queue first drained, on screen | 811.0 (797.1-889.6) | 174.9 (113.6-321.5) | 168.2 (117.3-286.9) |

`folo serve` alone, six rounds: ms from spawn to listening and to the answer of a `/api/system` asked at once,
and that answer's CPU reading (`cpu.util`) next to one asked 2 s later:

| | before | PR | fixed |
| --- | ---: | ---: | ---: |
| listening | 231.2 (207.6-467.7) | 32.1 (21.9-136.1) | 29.4 (21.0-45.2) |
| first `/api/system` answered | 1059.0 (775.2-2430.9) | 512.4 (404.3-887.2) | 444.2 (413.2-1321.3) |
| its CPU reading | 0.29 0.08 0.45 0.58 0.48 0.63 | 1.00 1.00 1.00 0.65 0.92 1.00 | 0.37 0.30 0.53 0.77 0.54 0.47 |
| the reading 2 s later | 0.38 0.38 0.54 0.60 0.65 0.36 | 0.36 0.44 0.60 0.54 0.63 0.48 | 0.33 0.37 0.54 0.39 0.53 0.75 |

The probe's check itself, timed in a fresh process: 1.2 to 1.5 ms with the fixes, 1.1 ms without (three
processes each). A second check in a process where nothing else started Winsock takes about 1.2 ms instead of
0.05 ms, since the first one's `WSACleanup` unloads it; the app checks once per connection attempt.

Frames, `--bench-frames 100`, headless, scale 1.5, five alternating rounds, us ("after" is "fixed"):

| screen | window | full before | full after | paint before | paint after | scroll after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| overview | 1280x800 | 2126 (1932-2222) | 1936 (1777-2022) | 597 | 487 | 4205 |
| runs | 1280x800 | 1134 (1026-1306) | 855 (736-1556) | 372 | 216 | 849 |
| dataset | 1280x800 | 1157 (1036-1635) | 1009 (910-1130) | 276 | 237 | 2322 |
| run-live | 1280x800 | 3445 (3024-6872) | 2945 (2447-3424) | 1104 | 774 | 5475 |
| settings | 1280x800 | 1837 (1497-1944) | 1265 (1151-1452) | 331 | 273 | 3292 |
| overview | 2560x1440 | 1648 (1310-2214) | 1838 (1300-2189) | 423 | 431 | 1589 |
| runs | 2560x1440 | 1537 (1240-2097) | 833 (812-2304) | 534 | 217 | 860 |
| dataset | 2560x1440 | 4178 (3900-4336) | 3716 (3343-4254) | 1008 | 907 | 10115 |
| run-live | 2560x1440 | 4155 (3952-4652) | 3840 (3393-4286) | 1493 | 1101 | 9390 |
| settings | 2560x1440 | 1697 (1321-1885) | 2017 (1350-2154) | 405 | 474 | 4627 |

Overview and settings at 2560 x 1440 were called ties here (the ranges cover each other), although settings'
median was 19 % higher; forty alternating rounds of these same binaries, run again later, put both 8 to 9 %
faster ("After the second review round", below). The run-live rows with all three
builds (five rounds, full frame / paint): 1280 x 800 2972 / 1023 before, 2828 / 810 PR, 2490 / 685 fixed;
2560 x 1440 3943 / 1483, 3839 / 1137, 3420 / 997. The path cache's one-pass eviction and exact accounting
cost nothing measurable.

Pixels: all 90 scenarios rendered with `--shot` by "before" and "fixed" at 1280 x 800 @ 1 and 1920 x 1200 @ 1.5:
180 of 180 identical (`tools/parity.py --tol 0`).

### After the second review round

The second round's fixes: `/api/system`'s preparation owned by the server (a request that goes away cancels
nothing), CPU readings closer than sysinfo's 200 ms interval giving the last reading again, the path cache
counting its table at its allocated size and rebuilding it after an eviction, and the refused-at-once check's
Winsock guards made only on success. The branch was rebased onto `origin/main` at `c39e1862` first, so the two
builds differ only by this pull request: "main" is `c39e1862` (for the launch, plus the two trace-point commits
of this pull request, which only print), "fixed" is the branch at `a1ff2cca`. Release builds, warm file cache,
other programs using about a quarter of the CPU.

Launch, as above (fresh data dir, `folo serve` from the matching build), ten alternating rounds, ms since the
process was created. A first batch of ten gave the same medians except for one cold start of the freshly built
`folo.exe` (server listening at 645 ms in its first round); these are the second batch:

| | main | fixed |
| --- | ---: | ---: |
| first frame | 34.8 (33.6-38.9) | 35.1 (33.8-37.8) |
| probe of port 7878 says nobody is there | 549.9 (544.5-560.1) | 45.8 (35.8-65.3) |
| `folo serve` listening | 727.8 (722.6-765.0) | 73.0 (70.0-91.4) |
| runs, datasets and models on screen | 745.8 (740.5-780.0) | 92.1 (87.9-111.2) |
| the system card on screen | 1174.2 (1166.8-1215.4) | 433.2 (428.8-470.8) |
| the next frame after the image queue first drained | 751.9 (748.0-794.5) | 108.0 (102.8-125.1) |

`folo serve` alone, six rounds: ms from spawn to listening and to the answer of a `/api/system` asked at once,
that answer's CPU reading, then eight `/api/system` calls at once (microseconds apart at the probe), and one 2 s
later. `main` has the burst weakness too: it refreshes the counters on every call.

| | main | fixed |
| --- | ---: | ---: |
| listening | 164.4 (151.7-167.5) | 17.1 (13.8-21.1) |
| first `/api/system` answered | 608.2 (596.6-632.1) | 369.1 (364.4-370.8) |
| its CPU reading | 0.08 0.10 0.00 0.19 0.00 1.00 | 0.08 0.06 0.13 0.05 0.06 0.05 |
| burst readings at 1.00 | 28 of 48 | 0 of 48 |
| the reading 2 s later | 0.03 0.04 0.03 0.03 0.04 0.03 | 0.03 0.03 0.03 0.03 0.03 0.03 |

Frames, `--bench-frames 100`, headless, scale 1.5, fifteen alternating rounds, us; "faster" counts the rounds in
which the fixed build's full frame was shorter than main's in the same round:

| screen | window | full main | full fixed | paint main | paint fixed | faster | scroll fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overview | 1280x800 | 897 (828-1006) | 829 (751-951) | 246 | 209 | 14/15 | 1994 |
| runs | 1280x800 | 769 (716-886) | 608 (565-775) | 287 | 162 | 14/15 | 592 |
| dataset | 1280x800 | 890 (826-941) | 819 (781-907) | 221 | 201 | 15/15 | 1939 |
| run-live | 1280x800 | 2663 (2321-3223) | 2298 (2050-2585) | 901 | 612 | 15/15 | 3944 |
| settings | 1280x800 | 1050 (924-1107) | 846 (778-883) | 204 | 187 | 15/15 | 1748 |
| overview | 2560x1440 | 1034 (1004-1172) | 957 (933-1049) | 265 | 224 | 15/15 | 949 |
| runs | 2560x1440 | 993 (804-2448) | 725 (630-2036) | 339 | 186 | 13/15 | 720 |
| dataset | 2560x1440 | 2623 (2516-2919) | 2399 (2295-2609) | 642 | 581 | 15/15 | 7043 |
| run-live | 2560x1440 | 2420 (2260-2711) | 2003 (1876-2115) | 894 | 583 | 15/15 | 5534 |
| settings | 2560x1440 | 928 (881-1034) | 842 (828-896) | 219 | 206 | 15/15 | 2470 |

Settings and overview at 2560 x 1440, which the first round of fixes reported as ties although settings' median
was 19 % higher there, measured again with forty alternating rounds (full frame; the ratio is the median of the
per-round fixed / main ratios, with its quartiles):

| row | build pair | full main | full fixed | ratio | faster | paint main / fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| settings 2560x1440 | `c39e1862` / `a1ff2cca` | 1080 (926-1571) | 962 (862-1216) | 0.888 (0.837-0.924) | 37/40 | 243 / 228 |
| overview 2560x1440 | `c39e1862` / `a1ff2cca` | 1076 (1020-1662) | 993 (948-1339) | 0.927 (0.899-0.944) | 36/40 | 270 / 230 |
| settings 2560x1440 | the first fix round's two binaries | 1024 (890-1698) | 918 (830-1414) | 0.912 (0.789-0.974) | 32/40 | 234 / 218 |
| overview 2560x1440 | the first fix round's two binaries | 1094 (1046-2398) | 1018 (956-1832) | 0.917 (0.874-0.938) | 36/40 | 270 / 233 |

The binaries of that round, run again, are faster too: its +19 % came from five rounds on a busier machine, not
from the build.

Pixels: all 90 scenarios rendered with `--shot` by "main" (`c39e1862`, no trace points) and "fixed" at
1280 x 800 @ 1 and 1920 x 1200 @ 1.5: 180 of 180 identical (`tools/parity.py --tol 0`). The release exe is
31,053,824 bytes on main and 31,145,472 with this pull request.

## Not measured

- A truly maximized 4K window on the real desktop: every launch here is off screen (or on a private desktop
  nobody sees) at 1280 x 800; the frame bench's 2560 x 1440 rows and the renderer's 3840 x 2160 rows are the
  nearest numbers.
- Linux and macOS: nothing here ran there (CI builds and tests the app headless on Linux; macOS is type-checked
  only).
- The GPUI numbers come from the same machine, the same day and the same tools, but from a build with the
  baseline's patch (off-screen placement and the trace lines), not the shipped GPUI app.
