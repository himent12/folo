/**
 * Per-Image Error-Analysis Explorer (`GET …/entries/{id}/images`): inspects evaluated benchmark images with
 * client-side error-focus filters (All, Missed GT / FN, False positives / FP, Any error), diagnostic sorts
 * (Highest errors, Benchmark order, Most FN, Most FP, Lowest IoU, Lowest score), canvas layer toggles
 * (Ground Truth, Predicted boxes, Error highlights), TP / FP / FN badges per image, and a large-view modal
 * showing per-box IoU and error cause (TP, class mismatch, localization, duplicate, background FP, missed GT).
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Download from '$icons/download';
import Eye from '$icons/eye';
import EyeOff from '$icons/eye-off';
import FileJson from '$icons/file-json';
import { computed, css, effect, For, MediaQuery, onWindow, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Bench, BenchAccuracy, BenchEntry, BenchEntryImage } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { conf, num } from '$lib/format';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import { dominantAspect } from '$lib/ui/TileGrid';
import { tooltip } from '$lib/ui/tooltip';
import { filterAndSortImages, summarizeGalleryErrors, type ErrorFilter, type ErrorSort, type ImageDiagnostics } from './errorAnalysis';
import { collectBenchErrorRows, exportBenchErrorsCsv, exportBenchErrorsJson } from './errorExport';
import MatchView from './MatchView';
import { bestF1Conf, imagesShownText, imagesText } from '../words';

/** Images per page of the gallery: a page more each time, so 5000 images never load at once. */
export const GALLERY_PAGE = 24;

export interface ImagesSectionProps {
  bench: Bench;
  entry: BenchEntry;
  /** The entry's accuracy: its best F1 (./words bestF1Conf), and the confidence a server before the page's `conf` took
   *  the images at (`best_conf`). */
  accuracy: BenchAccuracy;
}

const SORT_OPTIONS: { value: ErrorSort; label: string }[] = [
  { value: 'worst', label: 'Highest errors' },
  { value: 'fn', label: 'Most missed (FN)' },
  { value: 'fp', label: 'Most false pos (FP)' },
  { value: 'lowest-iou', label: 'Lowest IoU' },
  { value: 'lowest-score', label: 'Lowest score' },
  { value: 'order', label: 'Benchmark order' },
];

const SORT_SENTENCE: Record<ErrorSort, string> = {
  worst: 'The most missed and wrong first.',
  fn: 'Images with the most missed ground truth (FN) first.',
  fp: 'Images with the most false positives (FP) first.',
  'lowest-iou': 'Images with the lowest matched IoU first.',
  'lowest-score': 'Images with the lowest confidence detections first.',
  order: 'In the benchmark’s order.',
};

export default function ImagesSection(props: ImagesSectionProps) {
  const phone = new MediaQuery(media.phone, false);
  const errorFilter = signal<ErrorFilter>('all');
  const errorSort = signal<ErrorSort>('worst');
  const showTruth = signal(true);
  const showPred = signal(true);
  const showErrors = signal(true);

  const images = signal<BenchEntryImage[]>([]);
  const total = signal<number | null>(null);
  /** The confidence the detections were taken at (the page's `conf`); null until the first page is in. */
  const takenAt = signal<number | null>(null);
  const loading = signal(false);
  const error = signal<unknown>(undefined);
  const urls = signal<Record<string, string>>({});
  let seq = 0;

  /** Only 'order' changes the server fetch order; all error-focused sorts build cleanly over the worst-first pages. */
  const serverSort = computed<'worst' | 'order'>(() => (errorSort.get() === 'order' ? 'order' : 'worst'));

  async function load(reset: boolean) {
    const my = ++seq;
    loading.set(true);
    error.set(undefined);
    try {
      const page = await hub.api.benchEntryImages(props.bench.id, props.entry.id, {
        sort: serverSort.peek(),
        offset: reset ? 0 : images.peek().length,
        limit: GALLERY_PAGE,
      });
      if (my !== seq) return;
      images.set(reset ? page.images : [...images.peek(), ...page.images]);
      total.set(page.total);
      takenAt.set(page.conf ?? props.accuracy.best_conf);
      for (const im of page.images) {
        const id = im.image.id;
        if (urls.peek()[id]) continue;
        // A thumbnail that cannot be had stays the tile's well; its counts and its large view still work.
        hub.api.benchImageUrl(props.bench.id, id, { thumb: 384 }).then(
          (u) => urls.set((all) => ({ ...all, [id]: u })),
          () => {},
        );
      }
    } catch (e) {
      if (my === seq) error.set(e);
    } finally {
      if (my === seq) loading.set(false);
    }
  }
  // By id: the page hands down a new benchmark and entry on every `bench` event (and every load of the lists after one),
  // which must not send the gallery back to its first page.
  const benchId = computed(() => props.bench.id);
  const entryId = computed(() => props.entry.id);
  effect(() => {
    void serverSort.get();
    void benchId.get();
    void entryId.get();
    if (hub.mode === 'connecting') return;
    void hub.epoch;
    images.set([]);
    total.set(null);
    takenAt.set(null);
    untrack(() => void load(true));
  });

  const summary = computed(() => summarizeGalleryErrors(images.get()));
  const visible = computed(() => filterAndSortImages(images.get(), errorFilter.get(), errorSort.get(), props.bench.classes));
  const exportErrorCount = computed(
    () => collectBenchErrorRows(visible.get(), errorFilter.get(), props.bench.classes, props.entry.scored).length,
  );
  const exportInput = () => ({
    bench: props.bench,
    entry: props.entry,
    analyzed: visible.peek(),
    filter: errorFilter.peek(),
    sort: errorSort.peek(),
    conf: takenAt.peek(),
    loadedImages: images.peek().length,
    totalImages: total.peek(),
  });
  const filterOptions = computed(() => {
    const s = summary.get();
    const compact = phone.current;
    return [
      { value: 'all' as const, label: 'All', meta: s.total ? num(s.total) : undefined },
      { value: 'fn' as const, label: compact ? 'FN' : 'Missed (FN)', meta: s.total ? num(s.withFn) : undefined },
      { value: 'fp' as const, label: compact ? 'FP' : 'False pos (FP)', meta: s.total ? num(s.withFp) : undefined },
      { value: 'errors' as const, label: compact ? 'Errors' : 'Any error', meta: s.total ? num(s.withErrors) : undefined },
    ];
  });

  /**
   * The tiles' shape: the commonest among the images loaded (within 3:4 to 16:9), as a dataset's gallery has it, so a
   * row of tiles lines up whatever each photo is; an image of another shape is shown whole on the tile's well, its
   * boxes on it (./MatchView). Never the first image's: a portrait photo first made every tile a tall box.
   */
  const ratio = computed(() => (images.get().length ? dominantAspect(images.get().map((im) => im.image)) : 4 / 3));

  // ---- the large view ----
  const openId = signal<string | null>(null);
  const fullUrl = signal<string | null>(null);
  const active = signal<string | null>(null);
  const openIdx = computed(() => {
    const id = openId.get();
    if (id === null) return -1;
    return visible.get().findIndex((v) => v.item.image.id === id);
  });
  const shownEntry = computed(() => {
    const i = openIdx.get();
    return i >= 0 ? visible.get()[i]! : null;
  });
  const shown = computed(() => shownEntry.get()?.item ?? null);

  effect(() => {
    const im = shown.get();
    fullUrl.set(null);
    active.set(null);
    if (!im) return;
    let gone = false;
    hub.api.benchImageUrl(benchId.get(), im.image.id).then(
      (u) => !gone && fullUrl.set(u),
      () => {},
    );
    return () => (gone = true);
  });

  async function step(d: number) {
    const currentId = openId.peek();
    if (currentId === null || loading.peek()) return;
    const list = visible.peek();
    const at = list.findIndex((v) => v.item.image.id === currentId);
    if (at < 0) return;
    const next = at + d;
    if (next < 0) return;
    if (next >= list.length) {
      const t = total.peek();
      if (t !== null && images.peek().length >= t) return;
      await load(false);
      // Never reopen a closed modal or jump if the user switched images while loading.
      if (openId.peek() !== currentId) return;
      const updated = visible.peek();
      const idxNow = updated.findIndex((v) => v.item.image.id === currentId);
      const target = idxNow >= 0 ? updated[idxNow + d] : undefined;
      if (target) openId.set(target.item.image.id);
      return;
    }
    const target = list[next];
    if (target) openId.set(target.item.image.id);
  }

  function onkeydown(e: KeyboardEvent) {
    if (openId.peek() === null || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    if (document.querySelectorAll('dialog[open]').length > 1) return;
    const t = e.target as HTMLElement | null;
    if (t?.closest('input, select, textarea')) return;
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      void step(1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      void step(-1);
    }
  }
  onWindow('keydown', onkeydown);

  /** What the gallery holds, said from the confidence it was taken at and what is loaded (./words imagesShownText). */
  const shownText = () => {
    if (!props.entry.scored) return 'Every box it gave, on the unchanged images.';
    const at = takenAt.get();
    if (at === null) return 'Its detections on the unchanged images.';
    const all = images.get().length >= (total.get() ?? Infinity);
    return imagesShownText(at, bestF1Conf(props.accuracy), all && images.get().every((im) => !im.detections.length));
  };
  const name = (c: number) => props.bench.classes[c] ?? `class ${c}`;

  const layerToggles = () => (
    <div class="layers" role="group" aria-label="Canvas box layers">
      <Button
        size="sm"
        variant={showTruth.get() ? 'secondary' : 'ghost'}
        icon={showTruth.get() ? Eye : EyeOff}
        aria-pressed={showTruth.get()}
        onclick={() => showTruth.set((v) => !v)}
      >
        GT
      </Button>
      <Button
        size="sm"
        variant={showPred.get() ? 'secondary' : 'ghost'}
        icon={showPred.get() ? Eye : EyeOff}
        aria-pressed={showPred.get()}
        onclick={() => showPred.set((v) => !v)}
      >
        Pred
      </Button>
      <Button
        size="sm"
        variant={showErrors.get() ? 'secondary' : 'ghost'}
        icon={showErrors.get() ? Eye : EyeOff}
        aria-pressed={showErrors.get()}
        onclick={() => showErrors.set((v) => !v)}
      >
        Errors
      </Button>
    </div>
  );

  const legend = () => (
    <>
      {showErrors.get() ? (
        <>
          <span class="lg">
            <i class="k right" aria-hidden="true"></i>TP (Right)
          </span>
          <span class="lg">
            <i class="k wrong" aria-hidden="true">
              <b></b>
            </i>
            FP (Wrong)
          </span>
          <span class="lg">
            <i class="k missed" aria-hidden="true"></i>FN (Missed GT)
          </span>
        </>
      ) : (
        <>
          <span class="lg">
            <i class="k plain-pred" aria-hidden="true"></i>Predicted
          </span>
          <span class="lg">
            <i class="k plain-truth" aria-hidden="true"></i>Ground truth
          </span>
        </>
      )}
    </>
  );

  const tally = (d: () => ImageDiagnostics, showIou = false) => (
    <>
      {showErrors.get() ? (
        <>
          <span class="err-pill tp">TP {d().tp}</span>
          {d().fp ? <span class="err-pill fp">FP {d().fp}</span> : null}
          {d().fn ? <span class="err-pill fn">FN {d().fn}</span> : null}
          {!d().fp && !d().fn ? <span class="ok">{d().truthCount ? 'Clean' : 'Empty'}</span> : null}
          {showIou && d().meanIou !== null ? <span class="t-item faint">mIoU {num(d().meanIou, 2)}</span> : null}
        </>
      ) : (
        <span class="t-item faint">
          {d().truthCount} GT · {d().tp + d().fp} pred
        </span>
      )}
    </>
  );

  const isFiltered = computed(() => errorFilter.get() !== 'all');
  const posText = computed(() => {
    const i = openIdx.get();
    if (i < 0) return '';
    if (isFiltered.get()) return `${num(i + 1)} of ${num(visible.get().length)} matching`;
    return `${num(i + 1)} of ${num(total.get() ?? visible.get().length)}`;
  });
  const canStepNext = computed(() => {
    const i = openIdx.get();
    if (i < 0) return false;
    if (i + 1 < visible.get().length) return true;
    const t = total.get();
    return t !== null && images.get().length < t;
  });

  return (
    <>
      <Panel
        title="Images"
        meta={total.get() !== null ? imagesText(total.get()!) : undefined}
        id="images"
        actions={() => <Select size="sm" label="Sort evaluated images" bind:value={errorSort} options={SORT_OPTIONS} />}
      >
        <div class="head">
          <p class="what">
            {shownText()} {SORT_SENTENCE[errorSort.get()]}
          </p>
          <div class="explorer-bar">
            <Segmented size="sm" label="Filter images by error" bind:value={errorFilter} options={filterOptions.get()} />
            <div class="explorer-right">
              {layerToggles()}
              <div class="export-acts" role="group" aria-label="Export error analysis">
                <span use:tooltip={exportErrorCount.get() === 0 ? 'No failure cases in the current filter' : undefined}>
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={Download}
                    disabled={exportErrorCount.get() === 0}
                    data-bench-error-export="csv"
                    aria-label="Export errors CSV"
                    onclick={() => exportBenchErrorsCsv(exportInput())}
                  >
                    Export errors CSV
                  </Button>
                </span>
                <span use:tooltip={exportErrorCount.get() === 0 ? 'No failure cases in the current filter' : undefined}>
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={FileJson}
                    disabled={exportErrorCount.get() === 0}
                    data-bench-error-export="json"
                    aria-label="Export errors JSON"
                    onclick={() => exportBenchErrorsJson(exportInput())}
                  >
                    Export errors JSON
                  </Button>
                </span>
              </div>
            </div>
          </div>
          <p class="legend">{legend()}</p>
        </div>
        {error.get() && !images.get().length ? (
          <ErrorState error={error.get()} compact title="Could not load its images" retry={() => load(true)} />
        ) : (
          <>
            {!loading.get() && images.get().length > 0 && visible.get().length === 0 ? (
              <div class="filter-empty" role="status">
                <p>No loaded images match this error filter ({num(images.get().length)} inspected).</p>
                <div class="filter-empty-actions">
                  <Button size="sm" variant="secondary" onclick={() => errorFilter.set('all')}>
                    Show all
                  </Button>
                  {total.get() !== null && images.get().length < total.get()! ? (
                    <Button size="sm" variant="primary" loading={loading.get()} onclick={() => load(false)}>
                      Load {Math.min(GALLERY_PAGE, total.get()! - images.get().length)} more
                    </Button>
                  ) : null}
                </div>
              </div>
            ) : (
              <ul class="grid" aria-label={`Images of ${props.entry.name}, ${SORT_OPTIONS.find((o) => o.value === errorSort.get())?.label.toLowerCase() ?? 'worst first'}`}>
                <For each={visible.get()} key={(entry) => entry.item.image.id}>
                  {(entry) => {
                    const im = () => entry().item;
                    const d = () => entry().diag;
                    return (
                      <li>
                        <button
                          type="button"
                          class="tile"
                          aria-label={`${im().image.file}: ${d().tp} of ${d().truthCount} found, ${d().fn} missed, ${d().fp} wrong. Open`}
                          onclick={() => openId.set(im().image.id)}
                        >
                          <span class="img" style:aspect-ratio={ratio.get()}>
                            <MatchView
                              src={urls.get()[im().image.id] ?? null}
                              width={im().image.width}
                              height={im().image.height}
                              truth={im().image.boxes}
                              detections={im().detections}
                              classes={props.bench.classes}
                              showTruth={showTruth.get()}
                              showPred={showPred.get()}
                              showErrors={showErrors.get()}
                              alt=""
                            />
                          </span>
                          <span class="cap num">{tally(d)}</span>
                        </button>
                      </li>
                    );
                  }}
                </For>
                {loading.get() ? (
                  <For each={Array.from({ length: images.get().length ? 4 : 8 }, (_, k) => k)} key={(k) => k}>
                    {() => (
                      <li aria-hidden="true">
                        <span class="tile sk">
                          <span class="img" style:aspect-ratio={ratio.get()}>
                            <Skeleton h="100%" r="0" image />
                          </span>
                          <span class="cap">
                            <Skeleton w="60%" h="12px" />
                          </span>
                        </span>
                      </li>
                    )}
                  </For>
                ) : null}
              </ul>
            )}
            {total.get() !== null ? (
              <div class="foot">
                <span class="count num" role="status" aria-live="polite">
                  {isFiltered.get()
                    ? `${num(visible.get().length)} matching of ${num(images.get().length)} loaded (${num(total.get()!)} total)`
                    : `${num(images.get().length)} of ${num(total.get()!)}`}
                </span>
                {images.get().length < total.get()! ? (
                  <Button size="sm" loading={loading.get() && images.get().length > 0} onclick={() => load(false)}>
                    Show {Math.min(GALLERY_PAGE, total.get()! - images.get().length)} more
                  </Button>
                ) : null}
              </div>
            ) : null}
          </>
        )}
      </Panel>

      <Dialog
        open={shown.get() !== null}
        onclose={() => openId.set(null)}
        title={shown.get()?.image.file ?? 'Image'}
        size="lg"
        class="viewer"
        footer={() => (
          <>
            <span class="v-pos num" aria-live="polite">
              {posText.get()}
            </span>
            <Button icon={ChevronLeft} disabled={openIdx.get() <= 0} onclick={() => step(-1)}>
              Previous
            </Button>
            <Button iconRight={ChevronRight} loading={loading.get()} disabled={!canStepNext.get()} onclick={() => step(1)}>
              Next
            </Button>
          </>
        )}
      >
        {shownEntry.get() ? (
          <div class="v-body">
            {/* The image at its own shape, as large as the column and 64% of the viewport's height allow: a portrait photo
                takes the width that height gives it, centred, never between bands of the well. */}
            <div
              class="v-img"
              style={{
                'aspect-ratio': `${shown.get()!.image.width} / ${shown.get()!.image.height}`,
                '--ar': shown.get()!.image.width / shown.get()!.image.height,
              }}
            >
              <MatchView
                src={fullUrl.get()}
                width={shown.get()!.image.width}
                height={shown.get()!.image.height}
                truth={shown.get()!.image.boxes}
                detections={shown.get()!.detections}
                classes={props.bench.classes}
                showTruth={showTruth.get()}
                showPred={showPred.get()}
                showErrors={showErrors.get()}
                full
                active={active.get()}
                alt={shown.get()!.image.file}
              />
            </div>
            <div class="v-side">
              <p class="v-tally num">{tally(() => shownEntry.get()!.diag, true)}</p>
              {layerToggles()}
              <p class="legend">{legend()}</p>
              <ul class="v-list" aria-label={`Boxes of ${shown.get()!.image.file}`}>
                <For each={shownEntry.get()!.diag.boxes} key={(r) => r.key} fallback={<li class="v-none">No true box and no detection.</li>}>
                  {(r) => {
                    const isPredRow = () => r().key.startsWith('d');
                    const canHighlight = () => (isPredRow() ? showPred.get() : showTruth.get());
                    const swatchClass = () => (showErrors.get() ? r().outcome : isPredRow() ? 'plain-pred' : 'plain-truth');
                    return (
                      <li
                        tabindex={0}
                        class:on={active.get() === r().key}
                        onpointerenter={() => canHighlight() && active.set(r().key)}
                        onpointerleave={() => active.set(null)}
                        onfocus={() => canHighlight() && active.set(r().key)}
                        onblur={() => active.set(null)}
                      >
                        <i class={`k ${swatchClass()}`} aria-hidden="true">
                          {showErrors.get() && r().outcome === 'wrong' ? <b></b> : null}
                        </i>
                        <div class="v-info">
                          <span class="v-cls">{name(r().cls)}</span>
                          <span class="v-what" class:mismatch={showErrors.get() && r().cause === 'class-mismatch'}>
                            {showErrors.get() ? r().detail : isPredRow() ? 'Predicted' : 'Ground truth'}
                          </span>
                        </div>
                        <span class="v-sc num">{r().score !== null && props.entry.scored ? conf(r().score!) : ''}</span>
                      </li>
                    );
                  }}
                </For>
              </ul>
            </div>
          </div>
        ) : null}
      </Dialog>
    </>
  );
}

css`
  .head {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }
  .what {
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .explorer-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
    min-width: 0;
    max-width: 100%;
  }
  .explorer-bar :global(.seg button) {
    flex-shrink: 0;
  }
  .explorer-right,
  .layers,
  .export-acts {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .explorer-right {
    gap: var(--space-2);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .lg,
  .t-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .err-pill {
    display: inline-flex;
    align-items: center;
    padding: 0 6px;
    border-radius: var(--radius-xs);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .err-pill.tp {
    background: color-mix(in srgb, var(--success-solid) 14%, transparent);
    color: var(--success-text);
  }
  .err-pill.fp {
    background: color-mix(in srgb, var(--danger-solid) 14%, transparent);
    color: var(--danger-text);
  }
  .err-pill.fn {
    background: color-mix(in srgb, var(--warning-solid) 16%, transparent);
    color: var(--warning-text);
  }
  /* The legend's keys: the boxes as drawn, each its line style and its status ink. */
  .k {
    position: relative;
    flex-shrink: 0;
    width: 14px;
    height: 10px;
    border: 2px solid var(--c);
    border-radius: 1px;
  }
  .k.right {
    --c: var(--success-solid);
  }
  .k.wrong {
    --c: var(--danger-solid);
  }
  .k.missed {
    --c: var(--warning-solid);
    border-style: dashed;
  }
  .k.plain-pred {
    --c: var(--accent-solid);
  }
  .k.plain-truth {
    --c: var(--info-solid);
    border-style: dashed;
  }
  .k b {
    position: absolute;
    left: -4px;
    top: -4px;
    width: 8px;
    height: 8px;
    border-radius: 2px;
    background:
      linear-gradient(45deg, transparent 40%, var(--bg-panel) 40% 60%, transparent 60%),
      linear-gradient(-45deg, transparent 40%, var(--bg-panel) 40% 60%, transparent 60%),
      var(--danger-solid);
  }
  .filter-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    padding: var(--space-8) var(--space-4);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    text-align: center;
  }
  .filter-empty-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: var(--space-2);
  }
  /* Tiles at least 200 px wide, as many as fit (an intrinsic grid: the panel's width decides). */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: var(--space-3);
  }
  .tile {
    display: grid;
    gap: var(--space-2);
    width: 100%;
    text-align: left;
    border-radius: var(--radius-md);
  }
  .img {
    display: block;
    width: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
    background: var(--bg-inset);
  }
  @media (hover: hover) {
    .tile:not(.sk):hover .img {
      box-shadow: 0 0 0 1px var(--line-strong);
    }
  }
  .cap {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    min-height: 20px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
  }
  .ok {
    color: var(--text-secondary);
  }
  .faint {
    color: var(--text-tertiary);
  }
  .foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    margin-top: var(--space-4);
  }
  .count {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .v-body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 296px;
    gap: var(--space-4);
    align-items: start;
  }
  .v-img {
    width: min(100%, calc(64dvh * var(--ar)));
    margin: 0 auto;
    border-radius: var(--radius-md);
    overflow: hidden;
  }
  .v-side {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
  }
  .v-tally {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
  }
  .v-list {
    display: grid;
    max-height: 44dvh;
    overflow-y: auto;
  }
  .v-list li {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) 40px;
    align-items: center;
    gap: var(--space-2);
    min-height: 36px;
    padding: var(--space-1) var(--space-1);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    border-radius: var(--radius-xs);
  }
  .v-list li.on,
  .v-list li:focus-visible {
    background: var(--bg-hover);
  }
  .v-info {
    display: grid;
    min-width: 0;
  }
  .v-cls {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .v-what {
    color: var(--text-secondary);
  }
  .v-what.mismatch {
    color: var(--warning-text);
  }
  .v-sc {
    text-align: right;
    color: var(--text-tertiary);
  }
  .v-none {
    color: var(--text-tertiary);
  }
  .v-pos {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (min-width: 768px) {
    :global(dialog.lg.viewer) {
      width: min(1140px, calc(100vw - 48px));
    }
  }
  @media (max-width: 1023px) {
    .v-body {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  /* Phones: two tiles a row. */
  @media (max-width: 767px) {
    .grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-2);
    }
  }
`;
