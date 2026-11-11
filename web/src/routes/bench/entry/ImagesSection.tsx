// fw port of ImagesSection.svelte.
/**
 * What it gets wrong (`GET …/entries/{id}/images`): the entry's images, the worst first by default (most missed + wrong)
 * or in the benchmark's order, each tile the image with truth and detections overlaid (./MatchView: right solid,
 * wrong solid with a cross, missed dashed) and its counts in words under it; a page of GALLERY_PAGE at a time, "Show
 * more" loading the next. A tile opens a large view with the boxes, the legend, every box with its class and
 * confidence, and previous / next (the arrow keys too), which loads the next page when it reaches the end.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import { computed, css, effect, For, onWindow, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Bench, BenchEntry, BenchEntryImage } from '$lib/api/types';
import { matchImage } from '$lib/bench/match';
import { conf, num } from '$lib/format';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Skeleton from '$lib/ui/Skeleton';
import { dominantAspect } from '$lib/ui/TileGrid';
import MatchView, { type Outcome } from './MatchView';
import { imagesText } from '../words.fw';

/** Images per page of the gallery: a page more each time, so 5000 images never load at once. */
export const GALLERY_PAGE = 24;

export interface ImagesSectionProps {
  bench: Bench;
  entry: BenchEntry;
  bestConf: number;
}

/** What a tile says under its image: what it found right, missed and got wrong, in words. */
function counts(im: BenchEntryImage) {
  const found = im.image.boxes.length - im.missed;
  return { found, missed: im.missed, wrong: im.wrong, boxes: im.image.boxes.length };
}
type Counts = ReturnType<typeof counts>;

const ORDER = ['wrong', 'missed', 'right'];
const SORTS = [
  { value: 'worst' as const, label: 'Worst first' },
  { value: 'order' as const, label: 'In order' },
];

export default function ImagesSection(props: ImagesSectionProps) {
  const sort = signal<'worst' | 'order'>('worst');
  const images = signal<BenchEntryImage[]>([]);
  const total = signal<number | null>(null);
  const loading = signal(false);
  const error = signal<unknown>(undefined);
  const urls = signal<Record<string, string>>({});
  let seq = 0;

  async function load(reset: boolean) {
    const my = ++seq;
    loading.set(true);
    error.set(undefined);
    try {
      const page = await hub.api.benchEntryImages(props.bench.id, props.entry.id, {
        sort: sort.peek(),
        offset: reset ? 0 : images.peek().length,
        limit: GALLERY_PAGE,
      });
      if (my !== seq) return;
      images.set(reset ? page.images : [...images.peek(), ...page.images]);
      total.set(page.total);
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
    void sort.get();
    void benchId.get();
    void entryId.get();
    if (hub.mode === 'connecting') return;
    void hub.epoch;
    images.set([]);
    total.set(null);
    untrack(() => void load(true));
  });

  /**
   * The tiles' shape: the commonest among the images loaded (within 3:4 to 16:9), as a dataset's gallery has it, so a
   * row of tiles lines up whatever each photo is; an image of another shape is shown whole on the tile's well, its
   * boxes on it (./MatchView). Never the first image's: a portrait photo first made every tile a tall box.
   */
  const ratio = computed(() => (images.get().length ? dominantAspect(images.get().map((im) => im.image)) : 4 / 3));

  // ---- the large view ----
  const open = signal<number | null>(null);
  const fullUrl = signal<string | null>(null);
  const active = signal<string | null>(null);
  const shown = computed(() => (open.get() === null ? null : (images.get()[open.get()!] ?? null)));
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
    const at = open.peek();
    if (at === null) return;
    const next = at + d;
    if (next < 0) return;
    if (next >= images.peek().length) {
      const t = total.peek();
      if (t !== null && images.peek().length >= t) return;
      await load(false);
    }
    if (next < images.peek().length) open.set(next);
  }
  function onkeydown(e: KeyboardEvent) {
    if (open.peek() === null) return;
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
  /** The large view's list: every detection (right or wrong) and every true box nothing found. */
  const rows = computed(() => {
    const s = shown.get();
    if (!s) return [];
    const m = matchImage(s.image.boxes, s.detections);
    return [
      ...m.detections.map((d) => ({
        key: `d${d.index}`,
        outcome: (d.truth === null ? 'wrong' : 'right') as Outcome,
        cls: d.det.class,
        score: d.det.score as number | null,
        iou: d.truth === null ? null : d.iou,
      })),
      ...s.image.boxes.flatMap((b, i) =>
        m.found[i] === null ? [{ key: `t${i}`, outcome: 'missed' as Outcome, cls: b.class, score: null, iou: null }] : [],
      ),
    ].sort((a, b) => ORDER.indexOf(a.outcome) - ORDER.indexOf(b.outcome));
  });
  const name = (c: number) => props.bench.classes[c] ?? `class ${c}`;
  const shownCounts = computed(() => (shown.get() ? counts(shown.get()!) : { found: 0, missed: 0, wrong: 0, boxes: 0 }));

  const legend = () => (
    <>
      <span class="lg">
        <i class="k right" aria-hidden="true"></i>Right
      </span>
      <span class="lg">
        <i class="k wrong" aria-hidden="true">
          <b></b>
        </i>
        Wrong (no true box)
      </span>
      <span class="lg">
        <i class="k missed" aria-hidden="true"></i>Missed (nothing found it)
      </span>
    </>
  );

  const tally = (c: () => Counts) => (
    <>
      {!c().missed && !c().wrong ? (
        <span class="ok">{c().boxes ? `All ${c().boxes} found` : 'Nothing to find, nothing found'}</span>
      ) : (
        <>
          {c().missed ? (
            <span class="t-item">
              <i class="k missed" aria-hidden="true"></i>
              {c().missed} missed
            </span>
          ) : null}
          {c().wrong ? (
            <span class="t-item">
              <i class="k wrong" aria-hidden="true">
                <b></b>
              </i>
              {c().wrong} wrong
            </span>
          ) : null}
          <span class="t-item faint">
            {c().found} of {c().boxes} found
          </span>
        </>
      )}
    </>
  );

  return (
    <>
      <Panel
        title="Images"
        meta={total.get() !== null ? imagesText(total.get()!) : undefined}
        id="images"
        actions={() => <Segmented size="sm" label="Order of the images" bind:value={sort} options={SORTS} />}
      >
        <div class="head">
          <p class="what">
            {props.entry.scored ? `Its detections from confidence ${conf(props.bestConf)} up (its best F1)` : 'Every box it gave'}, on the unchanged
            images. {sort.get() === 'worst' ? 'The most missed and wrong first.' : 'In the benchmark’s order.'}
          </p>
          <p class="legend">{legend()}</p>
        </div>
        {error.get() && !images.get().length ? (
          <ErrorState error={error.get()} compact title="Could not load its images" retry={() => load(true)} />
        ) : (
          <>
            <ul class="grid" aria-label={`Images of ${props.entry.name}, ${sort.get() === 'worst' ? 'worst first' : 'in order'}`}>
              <For each={images.get()} key={(im) => im.image.id}>
                {(im, i) => {
                  const c = computed(() => counts(im()));
                  return (
                    <li>
                      <button
                        type="button"
                        class="tile"
                        aria-label={`${im().image.file}: ${c.get().found} of ${c.get().boxes} found, ${c.get().missed} missed, ${c.get().wrong} wrong. Open`}
                        onclick={() => open.set(i())}
                      >
                        <span class="img" style:aspect-ratio={ratio.get()}>
                          <MatchView
                            src={urls.get()[im().image.id] ?? null}
                            width={im().image.width}
                            height={im().image.height}
                            truth={im().image.boxes}
                            detections={im().detections}
                            classes={props.bench.classes}
                            alt=""
                          />
                        </span>
                        <span class="cap num">{tally(() => c.get())}</span>
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
            {total.get() !== null ? (
              <div class="foot">
                <span class="count num">
                  {num(images.get().length)} of {num(total.get()!)}
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
        open={open.get() !== null}
        onclose={() => open.set(null)}
        title={shown.get()?.image.file ?? 'Image'}
        size="lg"
        class="viewer"
        footer={() => (
          <>
            <span class="v-pos num">{open.get() !== null ? `${num(open.get()! + 1)} of ${num(total.get() ?? images.get().length)}` : ''}</span>
            <Button icon={ChevronLeft} disabled={!open.get()} onclick={() => step(-1)}>
              Previous
            </Button>
            <Button
              iconRight={ChevronRight}
              disabled={open.get() === null || (total.get() !== null && open.get()! >= total.get()! - 1)}
              onclick={() => step(1)}
            >
              Next
            </Button>
          </>
        )}
      >
        {shown.get() ? (
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
                full
                active={active.get()}
                alt={shown.get()!.image.file}
              />
            </div>
            <div class="v-side">
              <p class="v-tally num">{tally(() => shownCounts.get())}</p>
              <p class="legend">{legend()}</p>
              <ul class="v-list" aria-label={`Boxes of ${shown.get()!.image.file}`}>
                <For each={rows.get()} key={(r) => r.key} fallback={<li class="v-none">No true box and no detection.</li>}>
                  {(r) => (
                    <li onpointerenter={() => active.set(r().key)} onpointerleave={() => active.set(null)}>
                      <i class={`k ${r().outcome}`} aria-hidden="true">
                        {r().outcome === 'wrong' ? <b></b> : null}
                      </i>
                      <span class="v-cls">{name(r().cls)}</span>
                      <span class="v-what">
                        {r().outcome === 'right' ? `right · IoU ${num(r().iou, 2)}` : r().outcome === 'wrong' ? 'wrong' : 'missed'}
                      </span>
                      <span class="v-sc num">{r().score !== null && props.entry.scored ? conf(r().score!) : ''}</span>
                    </li>
                  )}
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
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }
  .what {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
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
    gap: 2px var(--space-3);
    min-height: 20px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
  }
  .ok {
    color: var(--text-secondary);
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
    grid-template-columns: minmax(0, 1fr) 260px;
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
    gap: 2px var(--space-3);
  }
  .v-list {
    display: grid;
    max-height: 48dvh;
    overflow-y: auto;
  }
  .v-list li {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) auto 40px;
    align-items: center;
    gap: var(--space-2);
    min-height: 28px;
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
  }
  .v-cls {
    color: var(--text-primary);
  }
  .v-what {
    color: var(--text-secondary);
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
      width: min(1100px, calc(100vw - 48px));
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
