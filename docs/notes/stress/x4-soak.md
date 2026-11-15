# Stress round, lane x4-soak: long sessions and scale

Surface: both apps and the server on a big data dir, and long scripted sessions: what grows, what slows,
what is left behind. Prebuilt binaries of the round (`folo serve --device cpu`, `folo-desktop --server`),
Studio in headless Chrome (playwright-core), Desktop off screen (`FOLO_UI_OFFSCREEN=1`, window messages only).

## The data (how it was built)

- Seed kit copied into the lane's data dir, server on port 18150.
- 50 datasets through the API: one `big-20k` of 20,500 synthetic 64 px images (8 classes), 47 small synthetic
  ones (10-30 images; unicode, 150-character and html-looking names), plus the seed's.
- 300 runs fabricated with the server stopped by copying a finished tiny run (`run_560a`) under new ids
  `run_e000`..`run_e12b` (docs/SERVER.md layout: `runs/<id>/` with `run.json`, `metrics.jsonl`, `log.jsonl`):
  50 completed with their weights hardlinked to the original's (so 100 models, 108 with the seed's), the rest
  failed (`out_of_memory`), metric values varied. 304 runs in all; real disk use about 0.2 GB.

## What was tried

- Time to first content and a screenshot (top and scrolled to the bottom) of every Studio list and page on
  that data: Overview, Datasets, big-20k grid (all and `split=val&class=7`), Runs, a run, Run compare with 3
  and 20 runs, Models, a model, Benchmarks, Playground, New run, the unicode and html-named datasets. All
  first content within about 1 s; nothing cut off or overlapping in the shots.
- The same screens in Desktop off screen (`--open <screen> --shot`): nothing cut off; launch to shot 10 s for
  Runs and 16.5 s for Datasets on the loaded PC. `--open` has no compare screen, so Desktop's Run compare
  was not shot.
- Studio soak, 20 min: 28 cycles over 12 routes, sampled every 30 s with CDP `Performance.getMetrics`
  after a forced GC. JS heap flat at 13.5-15.7 MB, listeners back to 140-180, DOM nodes back to 6-7k.
  Server private bytes 393 -> 430-500 MB, flat after 10 min.
- Desktop soak, 20 min of screen cycling: private bytes 170 -> 230-250 MB plateau (the image cache filling
  its 600-image budget), GDI objects constant at 13, clean exit on WM_CLOSE.
- Desktop idle on Overview: 0.2 CPU-s per 30 s (its 2 s list refresh).
- Server SSE churn: 3 rounds of 300 `GET /api/runs/{id}/events` opened and dropped. Handles 275 -> 636
  while open, back to 327 within one keep-alive (15 s); threads flat. Not a leak.
- Server Playground loop: 200 `POST /api/predict` (multipart, unicode file names). Handles flat at 327,
  private bytes 61 -> 113 MB at the first predict (the model loaded) then flat, no file left in the temp
  dir or the data dir.

## Findings

| id | severity | surface | finding | status |
| --- | --- | --- | --- | --- |
| x4-soak-1 | annoying | Studio Runs | past 200 runs every visit refetches every run's full metrics history | fixed |
| x4-soak-2 | annoying | Desktop dataset grid | a fling through a big grid starts one fetch per tile passed (~700 threads); the tiles in view load seconds late | fixed |

### x4-soak-1 Studio: Runs page refetches every run's metrics on every visit past 200 runs

- Repro: data dir with 304 finished runs. Open `#/runs`, go to `#/datasets`, back to `#/runs`.
- Happened: 312 `GET /api/runs/{id}/metrics` on every visit, though finished runs never change.
- Cause: `web/src/lib/api/metricsCache.ts` evicted the oldest-inserted entry past 200 and a hit did not
  refresh it. The table asks for the same 304 keys in the same order every visit, so FIFO eviction always
  threw out the next key needed: a 0 % hit rate.
- Fix: one entry per run, least recently used evicted first, bound 5000; an active run's entry is replaced as
  it advances. Test `web/src/lib/api/metricsCache.test.ts` (old code: 912 fetches for 3 visits of 304 runs).
- SIBLING: Desktop keeps `run_data` per run without a bound (`apps/desktop/src/store/mod.rs`): it fetches
  once, no thrash.

### x4-soak-2 Desktop: tiles in view load seconds late after a fling through a big grid

- Repro: the 20,500-image dataset, `folo-desktop --server ...`, Ctrl+K "big-20k" Enter, then 60 wheel notches
  of -1200, 30 ms apart.
- Happened: app threads 38 -> 724, server threads 42 -> 194; the tiles in view stay placeholders for 3-8 s.
- Cause: `apps/desktop/src/app/data.rs` `load_cached_image` spawned one background fetch per `image()` call,
  unlimited and unordered: every tile painted for one frame of the fling started an HTTP call on its own pool
  thread (cap 256), plus one thread each for ureq's timed address lookup. The tiles in view, asked last,
  waited behind all of them, and tiles evicted from the cache while waiting were still fetched.
- Fix: `apps/desktop/src/store/image_queue.rs` `LoadQueue`: at most 8 fetches on their way, the newest
  request first, a request whose slot the cache evicted is dropped unfetched; cleared on a new data source.
  Tests: `store/image_queue.rs` and `app/image_tests.rs` (a 3000-tile fling: the newest tiles fetched first,
  at most budget + 8 fetches; the old code fetched all 3000).

## Not a bug

- Studio Datasets page: one `GET /api/datasets/{id}` per dataset per visit (the list has no stats, by design).
- 400 on `ds_7816` in the scale run: the seed's deliberately broken dataset.

## Not done

- Run page left open across ten real CPU runs (Studio and Desktop), start/stop loops of tiny runs.
- Desktop idle CPU on every screen and `--bench-frames` on the big data; Studio `npm run perf`.
- Narrow window widths (1024, 800) on Runs and the big grid.
