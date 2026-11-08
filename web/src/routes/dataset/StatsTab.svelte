<script lang="ts">
  /** Dataset statistics: summary numbers, class balance, health findings and the two histograms. */
  import ArrowDown from '@lucide/svelte/icons/arrow-down';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Info from '@lucide/svelte/icons/info';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { Dataset, DatasetStats } from '$lib/api/types';
  import Histogram from '$lib/charts/Histogram.svelte';
  import { classColor } from '$lib/design/tokens';
  import { num } from '$lib/format';
  import { href } from '$lib/router.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Stat from '$lib/ui/Stat.svelte';
  import { countBinLabel, healthChecks, imageCount, reviewCount, sizeBinLabel, TINY_SIZE, type Severity } from './health';

  let { dataset, stats, onclass }: { dataset: Dataset; stats: DatasetStats; onclass: (cls: number) => void } = $props();

  const images = $derived(imageCount(dataset));
  const health = $derived(healthChecks(dataset, stats));
  const issues = $derived(reviewCount(health));
  const sizeEdges = $derived(stats.box_size_hist.edges);
  // Unequal size bins keep their true widths up to 60 % of the side; the open last bin (≥ 50 %) is drawn up to there.
  const sizeMax = $derived(Math.min(0.6, sizeEdges[sizeEdges.length - 1] ?? 1));
  const pctEdge = (v: number) => `${+(v * 100).toFixed(1)}%`;
  const balance = $derived(
    stats.class_counts
      .map((count, cls) => ({ cls, count, name: dataset.classes[cls] ?? `class ${cls}` }))
      .sort((a, b) => b.count - a.count || a.cls - b.cls),
  );
  const maxCount = $derived(Math.max(1, ...stats.class_counts));
  const ratio = $derived.by(() => {
    const present = stats.class_counts.filter((c) => c > 0);
    return present.length >= 2 ? Math.max(...present) / Math.min(...present) : 1;
  });
  const share = (x: number) => (stats.boxes ? (x / stats.boxes) * 100 : 0);
  const icons = { danger: CircleAlert, warning: TriangleAlert, info: Info, ok: CircleCheck } satisfies Record<Severity, unknown>;
  const perLast = $derived(stats.boxes_per_image_hist.counts.length - 1);

  function toHealth() {
    const el = document.getElementById('ds-health');
    if (!el) return;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
    // Move focus to the panel's heading so keyboard and screen-reader users land there too.
    const title = document.getElementById('ds-health-title');
    title?.setAttribute('tabindex', '-1');
    title?.focus({ preventScroll: true });
  }
</script>

<div class="stats">
  <Panel class="summary" pad="none">
    <div class="tiles">
      <Stat label="Images" value={num(images)}>
        {#snippet sub()}{num(dataset.splits.train)} / {num(dataset.splits.val)} / {num(dataset.splits.test)} <span class="split-key">train / val / test</span>{/snippet}
      </Stat>
      <Stat label="Boxes" value={num(stats.boxes)} sub="{dataset.classes.length} classes" />
      <Stat label="Boxes per image" value={images ? (stats.boxes / images).toFixed(1) : '—'} unit="mean" sub="over all images" />
      <Stat label="Empty images" value={num(stats.empty_images)} sub={images ? `${((stats.empty_images / images) * 100).toFixed(1)}% of images` : undefined} />
      {#if stats.broken_labels > 0}
        <!-- Points at the Health finding that explains what to do about them; the whole tile is the hit area. -->
        <div class="tile-link">
          <Stat label="Broken labels" value={num(stats.broken_labels)} tone="danger">
            {#snippet sub()}<span class="sr-only">Skipped in training.</span><button type="button" class="see" onclick={toHealth}>See Health<ArrowDown size={12} strokeWidth={2} aria-hidden="true" /></button>{/snippet}
          </Stat>
        </div>
      {:else}
        <Stat label="Broken labels" value="0" sub="all labels parse" />
      {/if}
      <Stat label="Imbalance" value={ratio.toFixed(1)} unit="×" sub="largest / smallest class" />
    </div>
  </Panel>

  <div class="two" class:issues>
    <Panel title="Class balance" meta="{dataset.classes.length} classes · by boxes" pad="none" id="ds-balance">
      <ul class="bars">
        {#each balance as b (b.cls)}
          <li>
            <button type="button" class:absent={b.count === 0} onclick={() => onclass(b.cls)} aria-label="{b.name}: {num(b.count)} boxes, {share(b.count).toFixed(1)}%. Show images">
              <span class="sw" style:background={classColor(b.cls)}></span>
              <span class="cname truncate">{b.name}</span>
              <span class="track"><span class="fill" style:width="{(b.count / maxCount) * 100}%" style:background={classColor(b.cls)}></span></span>
              <span class="count num">{num(b.count)}</span>
              <span class="pct num">{share(b.count).toFixed(1)}<small>%</small></span>
              <span class="go" aria-hidden="true"><ArrowRight size={12} /></span>
            </button>
          </li>
        {/each}
      </ul>
    </Panel>

    <Panel title="Health" meta={issues ? `${issues} to review` : 'looks good'} pad="none" id="ds-health">
      <ul class="health">
        {#each health as h (h.id)}
          {@const Icon = icons[h.severity]}
          <li class={h.severity}>
            <span class="hicon"><Icon size={16} strokeWidth={1.75} aria-label={h.severity === 'ok' ? 'OK' : h.severity} /></span>
            <span class="htext">
              <span class="htitle">{h.title}</span>
              <span class="hdetail">{h.detail}</span>
              {#if h.action}
                <a class="haction" href={href(h.action.path)}>{h.action.label}<ArrowRight size={12} strokeWidth={2} aria-hidden="true" /></a>
              {/if}
            </span>
          </li>
        {/each}
      </ul>
    </Panel>
  </div>

  <div class="two even">
    <Panel title="Box size" meta="% of image side, √area" id="ds-size">
      <Histogram
        edges={sizeEdges}
        counts={stats.box_size_hist.counts}
        binLabel={(lo, hi) => sizeBinLabel(lo, hi)}
        label="Histogram of box sizes relative to the image side (square root of the area share), with bins drawn at their true widths and a marker at the 5 % tiny-box threshold"
        color="var(--chart-1)"
        height={180}
        proportional
        xMax={sizeMax}
        edgeLabel={pctEdge}
        densityPer={0.01}
        marker={{ x: TINY_SIZE, label: 'tiny' }}
      />
      <p class="caption">Bins keep their true widths and bar heights are boxes per 1 % step, so a bar's area is its count. Left of the dashed line: tiny boxes, under 5 % of the side.</p>
    </Panel>
    <Panel title="Boxes per image" meta="images" id="ds-per">
      <Histogram
        edges={stats.boxes_per_image_hist.edges}
        counts={stats.boxes_per_image_hist.counts}
        binLabel={(lo, hi, i) => countBinLabel(lo, hi, i === perLast)}
        label="Histogram of the number of boxes per image"
        color="var(--chart-1)"
        unit="images"
        height={180}
      />
    </Panel>
  </div>
</div>

<style>
  .stats {
    display: grid;
    gap: var(--space-4);
  }
  .stats :global(.summary) {
    overflow: hidden;
  }
  /* 1 px gaps over a hairline-coloured background draw the dividers in any column count. */
  .tiles {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
  }
  .tiles > :global(*) {
    padding: var(--space-4);
    background: var(--bg-panel);
  }
  .two {
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
    gap: var(--space-4);
  }
  /* With findings to review, Health comes straight after the summary (first in the row, and first when stacked). */
  .two.issues {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }
  .two.issues > :global(#ds-health) {
    order: -1;
  }
  .two.even {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .tile-link {
    position: relative;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .tiles > .tile-link:hover {
    background: linear-gradient(var(--bg-hover), var(--bg-hover)), var(--bg-panel);
  }
  /* A neutral link: the red count above it carries the alarm. */
  .see {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .tile-link:hover .see {
    color: var(--text-primary);
  }
  .split-key {
    display: block;
    color: var(--text-tertiary);
  }
  .caption {
    margin-top: var(--space-2);
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    color: var(--text-tertiary);
  }
  .haction {
    justify-self: start;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .haction:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .see::after {
    content: '';
    position: absolute;
    inset: 0;
  }
  .see:focus-visible {
    outline: none;
  }
  .tile-link:has(.see:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  .tile-link:hover .see {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .bars {
    padding: 0 var(--space-2) var(--space-2);
  }
  .bars button {
    width: 100%;
    display: grid;
    grid-template-columns: 10px minmax(72px, 112px) minmax(0, 1fr) 56px 48px 12px;
    align-items: center;
    gap: var(--space-3);
    height: 32px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    text-align: left;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .bars button:hover {
    background: var(--bg-hover);
  }
  .bars:hover button:not(:hover) .fill {
    opacity: 0.45;
  }
  .sw {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
  /* Class marks keep the palette hue in both themes; a hairline edge holds pale hues on white. */
  :global([data-theme='light']) .sw,
  :global([data-theme='light']) .fill {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .absent .sw {
    opacity: 0.4;
  }
  .cname {
    font-weight: var(--weight-medium);
  }
  .track {
    height: 8px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    display: block;
    height: 100%;
    border-radius: 2px;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  .count {
    text-align: right;
  }
  .pct {
    text-align: right;
    color: var(--text-secondary);
  }
  .pct small {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .go {
    color: var(--text-tertiary);
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  .bars button:hover .go,
  .bars button:focus-visible .go {
    opacity: 1;
  }
  .health {
    display: grid;
    padding: 0 var(--space-2) var(--space-2);
  }
  .health li {
    display: flex;
    gap: var(--space-3);
    padding: var(--space-2) var(--space-2) var(--space-3);
  }
  .health li + li {
    border-top: 1px solid var(--line-subtle);
    padding-top: var(--space-3);
  }
  .hicon {
    flex-shrink: 0;
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius-md);
  }
  .danger .hicon {
    background: var(--danger-muted);
    color: var(--danger-text);
  }
  .warning .hicon {
    background: var(--warning-muted);
    color: var(--warning-text);
  }
  .info .hicon {
    background: var(--info-muted);
    color: var(--info-text);
  }
  .ok .hicon {
    background: var(--success-muted);
    color: var(--success-text);
  }
  .htext {
    display: grid;
    gap: 2px;
    min-width: 0;
    padding-top: 4px;
  }
  .htitle {
    font-weight: var(--weight-medium);
  }
  .hdetail {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 1199px) {
    .tiles {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .two,
    .two.issues {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (pointer: coarse) {
    .bars button {
      height: 44px;
    }
    .haction {
      min-height: 44px;
    }
  }
  @media (max-width: 767px) {
    .tiles {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .tiles > :global(*) {
      padding: var(--space-3) var(--space-4);
    }
    .two.even {
      grid-template-columns: minmax(0, 1fr);
    }
    .bars button {
      grid-template-columns: 10px minmax(56px, 88px) minmax(0, 1fr) 48px 44px;
      gap: var(--space-2);
    }
    .go {
      display: none;
    }
  }
</style>
