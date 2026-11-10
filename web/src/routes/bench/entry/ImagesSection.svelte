<script lang="ts" module>
  /** Images per page of the gallery: a page more each time, so 5000 images never load at once. */
  export const GALLERY_PAGE = 24;
</script>

<script lang="ts">
  /**
   * What it gets wrong (`GET …/entries/{id}/images`): the entry's images, the worst first by default (most missed + wrong)
   * or in the benchmark's order, each tile the image with truth and detections overlaid (./MatchView: right solid,
   * wrong solid with a cross, missed dashed) and its counts in words under it; a page of GALLERY_PAGE at a time, "Show
   * more" loading the next. A tile opens a large view with the boxes, the legend, every box with its class and
   * confidence, and previous / next (the arrow keys too), which loads the next page when it reaches the end.
   */
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { hub } from '$lib/api/index.svelte';
  import type { Bench, BenchEntry, BenchEntryImage } from '$lib/api/types';
  import { matchImage } from '$lib/bench/match';
  import { conf, num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { dominantAspect } from '$lib/ui/TileGrid.svelte';
  import MatchView, { type Outcome } from './MatchView.svelte';
  import { imagesText } from '../words';

  let { bench, entry, bestConf }: { bench: Bench; entry: BenchEntry; bestConf: number } = $props();

  let sort = $state<'worst' | 'order'>('worst');
  let images = $state.raw<BenchEntryImage[]>([]);
  let total = $state<number | null>(null);
  let loading = $state(false);
  let error = $state.raw<unknown>(undefined);
  let urls = $state<Record<string, string>>({});
  let seq = 0;

  async function load(reset: boolean) {
    const my = ++seq;
    loading = true;
    error = undefined;
    try {
      const page = await hub.api.benchEntryImages(bench.id, entry.id, { sort, offset: reset ? 0 : images.length, limit: GALLERY_PAGE });
      if (my !== seq) return;
      images = reset ? page.images : [...images, ...page.images];
      total = page.total;
      for (const im of page.images) {
        const id = im.image.id;
        if (urls[id]) continue;
        // A thumbnail that cannot be had stays the tile's well; its counts and its large view still work.
        hub.api.benchImageUrl(bench.id, id, { thumb: 384 }).then(
          (u) => (urls = { ...urls, [id]: u }),
          () => {},
        );
      }
    } catch (e) {
      if (my === seq) error = e;
    } finally {
      if (my === seq) loading = false;
    }
  }
  $effect(() => {
    void sort;
    void entry.id;
    if (hub.mode === 'connecting') return;
    void hub.epoch;
    images = [];
    total = null;
    void load(true);
  });

  /** What a tile says under its image: what it found right, missed and got wrong, in words. */
  function counts(im: BenchEntryImage) {
    const found = im.image.boxes.length - im.missed;
    return { found, missed: im.missed, wrong: im.wrong, boxes: im.image.boxes.length };
  }
  /**
   * The tiles' shape: the commonest among the images loaded (within 3:4 to 16:9), as a dataset's gallery has it, so a
   * row of tiles lines up whatever each photo is; an image of another shape is shown whole on the tile's well, its
   * boxes on it (./MatchView). Never the first image's: a portrait photo first made every tile a tall box.
   */
  const ratio = $derived(images.length ? dominantAspect(images.map((im) => im.image)) : 4 / 3);

  // ---- the large view ----
  let open = $state<number | null>(null);
  let fullUrl = $state<string | null>(null);
  let active = $state<string | null>(null);
  const shown = $derived(open === null ? null : (images[open] ?? null));
  $effect(() => {
    const im = shown;
    fullUrl = null;
    active = null;
    if (!im) return;
    let gone = false;
    hub.api.benchImageUrl(bench.id, im.image.id).then(
      (u) => !gone && (fullUrl = u),
      () => {},
    );
    return () => (gone = true);
  });
  async function step(d: number) {
    if (open === null) return;
    const next = open + d;
    if (next < 0) return;
    if (next >= images.length) {
      if (total !== null && images.length >= total) return;
      await load(false);
    }
    if (next < images.length) open = next;
  }
  function onkeydown(e: KeyboardEvent) {
    if (open === null) return;
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
  /** The large view's list: every detection (right or wrong) and every true box nothing found. */
  const rows = $derived.by(() => {
    if (!shown) return [];
    const m = matchImage(shown.image.boxes, shown.detections);
    return [
      ...m.detections.map((d) => ({ key: `d${d.index}`, outcome: (d.truth === null ? 'wrong' : 'right') as Outcome, cls: d.det.class, score: d.det.score, iou: d.truth === null ? null : d.iou })),
      ...shown.image.boxes.flatMap((b, i) => (m.found[i] === null ? [{ key: `t${i}`, outcome: 'missed' as Outcome, cls: b.class, score: null, iou: null }] : [])),
    ].sort((a, b) => ['wrong', 'missed', 'right'].indexOf(a.outcome) - ['wrong', 'missed', 'right'].indexOf(b.outcome));
  });
  const name = (c: number) => bench.classes[c] ?? `class ${c}`;
</script>

{#snippet legend()}
  <span class="lg"><i class="k right" aria-hidden="true"></i>Right</span>
  <span class="lg"><i class="k wrong" aria-hidden="true"><b></b></i>Wrong (no true box)</span>
  <span class="lg"><i class="k missed" aria-hidden="true"></i>Missed (nothing found it)</span>
{/snippet}

{#snippet tally(c: ReturnType<typeof counts>)}
  {#if !c.missed && !c.wrong}
    <span class="ok">{c.boxes ? `All ${c.boxes} found` : 'Nothing to find, nothing found'}</span>
  {:else}
    {#if c.missed}<span class="t-item"><i class="k missed" aria-hidden="true"></i>{c.missed} missed</span>{/if}
    {#if c.wrong}<span class="t-item"><i class="k wrong" aria-hidden="true"><b></b></i>{c.wrong} wrong</span>{/if}
    <span class="t-item faint">{c.found} of {c.boxes} found</span>
  {/if}
{/snippet}

<Panel title="Images" meta={total !== null ? imagesText(total) : undefined} id="images">
  {#snippet actions()}
    <Segmented
      size="sm"
      label="Order of the images"
      bind:value={sort}
      options={[
        { value: 'worst', label: 'Worst first' },
        { value: 'order', label: 'In order' },
      ]}
    />
  {/snippet}
  <div class="head">
    <p class="what">{entry.scored ? `Its detections from confidence ${conf(bestConf)} up (its best F1)` : 'Every box it gave'}, on the unchanged images. {sort === 'worst' ? 'The most missed and wrong first.' : 'In the benchmark’s order.'}</p>
    <p class="legend">{@render legend()}</p>
  </div>
  {#if error && !images.length}
    <ErrorState {error} compact title="Could not load its images" retry={() => load(true)} />
  {:else}
    <ul class="grid" aria-label="Images of {entry.name}, {sort === 'worst' ? 'worst first' : 'in order'}">
      {#each images as im, i (im.image.id)}
        {@const c = counts(im)}
        <li>
          <button type="button" class="tile" aria-label="{im.image.file}: {c.found} of {c.boxes} found, {c.missed} missed, {c.wrong} wrong. Open" onclick={() => (open = i)}>
            <span class="img" style:aspect-ratio={ratio}>
              <MatchView src={urls[im.image.id] ?? null} width={im.image.width} height={im.image.height} truth={im.image.boxes} detections={im.detections} classes={bench.classes} alt="" />
            </span>
            <span class="cap num">{@render tally(c)}</span>
          </button>
        </li>
      {/each}
      {#if loading}
        {#each Array.from({ length: images.length ? 4 : 8 }, (_, k) => k) as k (k)}
          <li aria-hidden="true"><span class="tile sk"><span class="img" style:aspect-ratio={ratio}><Skeleton h="100%" r="0" image /></span><span class="cap"><Skeleton w="60%" h="12px" /></span></span></li>
        {/each}
      {/if}
    </ul>
    {#if total !== null}
      <div class="foot">
        <span class="count num">{num(images.length)} of {num(total)}</span>
        {#if images.length < total}<Button size="sm" loading={loading && images.length > 0} onclick={() => load(false)}>Show {Math.min(GALLERY_PAGE, total - images.length)} more</Button>{/if}
      </div>
    {/if}
  {/if}
</Panel>

<svelte:window {onkeydown} />

<Dialog open={open !== null} onclose={() => (open = null)} title={shown?.image.file ?? 'Image'} size="lg" class="viewer">
  {#if shown}
    {@const c = counts(shown)}
    <div class="v-body">
      <!-- The image at its own shape, as large as the column and 64% of the viewport's height allow: a portrait photo
           takes the width that height gives it, centred, never between bands of the well. -->
      <div class="v-img" style:aspect-ratio="{shown.image.width} / {shown.image.height}" style:--ar={shown.image.width / shown.image.height}>
        <MatchView src={fullUrl} width={shown.image.width} height={shown.image.height} truth={shown.image.boxes} detections={shown.detections} classes={bench.classes} full {active} alt={shown.image.file} />
      </div>
      <div class="v-side">
        <p class="v-tally num">{@render tally(c)}</p>
        <p class="legend">{@render legend()}</p>
        <ul class="v-list" aria-label="Boxes of {shown.image.file}">
          {#each rows as r (r.key)}
            <!-- svelte-ignore a11y_no_noninteractive_element_interactions (pointing at a row only brings its box forward; the list is read as text) -->
            <li onpointerenter={() => (active = r.key)} onpointerleave={() => (active = null)}>
              <i class="k {r.outcome}" aria-hidden="true">{#if r.outcome === 'wrong'}<b></b>{/if}</i>
              <span class="v-cls">{name(r.cls)}</span>
              <span class="v-what">{r.outcome === 'right' ? `right · IoU ${num(r.iou, 2)}` : r.outcome === 'wrong' ? 'wrong' : 'missed'}</span>
              <span class="v-sc num">{r.score !== null && entry.scored ? conf(r.score) : ''}</span>
            </li>
          {:else}
            <li class="v-none">No true box and no detection.</li>
          {/each}
        </ul>
      </div>
    </div>
  {/if}
  {#snippet footer()}
    <span class="v-pos num">{open !== null ? `${num(open + 1)} of ${num(total ?? images.length)}` : ''}</span>
    <Button icon={ChevronLeft} disabled={!open} onclick={() => step(-1)}>Previous</Button>
    <Button iconRight={ChevronRight} disabled={open === null || (total !== null && open >= total - 1)} onclick={() => step(1)}>Next</Button>
  {/snippet}
</Dialog>

<style>
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
</style>
