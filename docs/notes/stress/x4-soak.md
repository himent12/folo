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
| x4-soak-3 | annoying | Studio Run page | every Run page shown while its run is live stays in memory after you leave it (~1,300 DOM nodes, ~0.6 MB per run) | part fixed (one retainer), rest deferred |

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
  "Newest" is the latest ask: a tile drawn again while it waits (`image()` and `bench_image()` on a loading
  slot call `LoadQueue::touch`) moves ahead of the tiles queued after it (review round 1).
  Tests: `store/image_queue.rs` and `app/image_tests.rs`: a 3000-tile fling starts tiles 0-7 at once, then
  2999 down to 2992, at most budget + 8 fetches (the old code fetched all 3000); 500 tiles, then tiles 20
  and 8 drawn again: 8 and 20 start next, each tile fetched once.
- Also (review round 1, pre-existing): an image that failed to load (server down) stayed a placeholder for
  good; Retry (`retry_all`, also run on a backend switch) now forgets failed image slots
  (`ImageCache::forget_failed`) so they load again. Still deferred: a failed tile looks the same as a loading
  one, with no per-tile Retry (Studio has one, `web/src/lib/detect/DatasetImage.tsx`); same for benchmark
  thumbnails and the inspector.

### x4-soak-3 Studio: Run pages that were live while shown stay in memory

- Repro: `folo serve --mock`, headless Chrome on `#/runs`. Loop: `POST /api/runs` (2 epochs), set
  `location.hash = #/runs/<new id>`, wait until the run ends; forced GC; CDP `Performance.getMetrics`
  (lane script `scripts/live-follow.mjs` with `NOQ=1`, so no `Runtime.queryObjects`; see below).
- Happened: 20 runs of 2 epochs: DOM nodes 3,899 -> 28,808, JS heap 7.5 -> 19.2 MB, while the live document
  stays at about 1,670 elements. Finished runs opened one after another plateau (about 6,100 nodes): only
  pages that were live while shown are kept.
- Cause, part one (fixed): `web/framework/runtime/bind.ts` created the shared `ResizeObserver` of
  `bind:clientWidth` inside the first caller's effect, so its callback closed over that caller's element: the
  first bound element (on the Run page, the run chart) and its whole detached page were kept for good. Fix: the
  callback is the module-level `notifySize`. Test `web/framework/runtime/bind.test.tsx` "bind:clientWidth"
  (WeakRef and a forced GC; the old code keeps the element).
- Measured with that fix: runs of 1 epoch now plateau (about 5,600 nodes; about 7,200 before the fix, one
  whole page more). Runs of 2 epochs still grow about 1,700 nodes per run: a run that crosses an epoch while
  its page is shown leaves that page behind. (A first version of the repro script used CDP
  `Runtime.queryObjects`, whose result arrays stay alive in the inspector and themselves keep every element
  they list: its growth for 1-epoch runs was that artifact. The numbers here are without it.)
- Cause, part two (deferred, not located): a heap snapshot of an unminified build after four 2-epoch runs
  (lane `logs/holders-ep2.txt:355-379`) reaches an old page's chart from an `SVGSVGElement` that the
  document's `SVGDocumentExtensions` still lists: its `$$pointermove` handler, a closure named
  `onpointermove`, -> a computed (`x`) -> its `_targets` -> effects and owners of the left page
  (`web/framework/runtime/core.ts`). The name and shape match the run chart's handler
  (`web/src/lib/charts/LineChart.tsx:335`, bound at `:550`); that mapping is not verified. (The round-0 text
  named a command palette item here: wrong, the palette's handler is on an `<li>`. The palette appears only on
  a separate path, `holders-ep2.txt:54-60`, from its `items` computed to an icon closure, which does not show
  an effect chain.) Why the detached chart's effects stay linked is not found; it sits in the framework's
  reactivity core: left for its owner. Repro scripts: lane dir `scripts/live-follow.mjs` (`NOQ=1`, epochs 2)
  and `scripts/holders.mjs`.

## Not a bug

- Studio Datasets page: one `GET /api/datasets/{id}` per dataset per visit (the list has no stats, by design).
- 400 on `ds_7816` in the scale run: the seed's deliberately broken dataset.

## Not done

- Run page left open across ten real CPU runs (Studio and Desktop), start/stop loops of tiny runs.
- Desktop idle CPU on every screen and `--bench-frames` on the big data; Studio `npm run perf`.
- Narrow window widths (1024, 800) on Runs and the big grid.
- Re-measuring the 20k-grid fling with the fixed Desktop build: the fix is checked by its behaviour tests
  (`store/image_queue.rs`, `app/image_tests.rs`), not by a new thread-count run. The `apps/desktop` gates
  (fmt, test, clippy, build) first ran in review round 2 (all pass).
