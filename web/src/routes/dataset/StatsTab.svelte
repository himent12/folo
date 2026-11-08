<script lang="ts">
  /**
   * Dataset statistics. Findings to review come first, and only when there are any; then four numbers, the class
   * balance as the focal chart, and the two histograms in ink.
   */
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Info from '@lucide/svelte/icons/info';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { Dataset, DatasetStats } from '$lib/api/types';
  import Histogram from '$lib/charts/Histogram.svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { num } from '$lib/format';
  import { href } from '$lib/router.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { countBinLabel, healthChecks, imageCount, reviewCount, sizeBinLabel, TINY_SIZE, type Severity } from './health';

  let { dataset, stats, onclass }: { dataset: Dataset; stats: DatasetStats; onclass: (cls: number) => void } = $props();

  const images = $derived(imageCount(dataset));
  const health = $derived(healthChecks(dataset, stats));
  const issues = $derived(reviewCount(health));
  const findings = $derived(health.filter((h) => h.severity !== 'ok'));
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
  const pct = (x: number) => (images ? `${((x / images) * 100).toFixed(1)}% of images` : '');
</script>

<div class="stats">
  {#if issues > 0}
    <Panel title="To review" meta={String(issues)} pad="none" id="ds-health">
      <ul class="health">
        {#each findings as h (h.id)}
          {@const Icon = icons[h.severity]}
          <li class={h.severity}>
            <span class="hicon"><Icon size={16} strokeWidth={1.75} aria-label={h.severity} /></span>
            <span class="htext">
              <span class="htitle">{h.title}</span>
              <span class="hdetail">{h.detail}</span>
            </span>
            {#if h.action}
              <a class="haction" href={href(h.action.path)}>{h.action.label}<ArrowRight size={12} strokeWidth={2} aria-hidden="true" /></a>
            {/if}
          </li>
        {/each}
      </ul>
    </Panel>
  {/if}

  <!-- 1 px gaps over a hairline-coloured background draw the dividers in any column count. -->
  <section class="kpis" aria-label="Summary">
    <div class="kpi">
      <span class="k-label">Images</span>
      <span class="k-value num">{num(images)}</span>
      <span class="k-sub num" title="train / val / test">{num(dataset.splits.train)} / {num(dataset.splits.val)} / {num(dataset.splits.test)}<span class="hide-sm"> train / val / test</span></span>
    </div>
    <div class="kpi">
      <span class="k-label">Boxes</span>
      <span class="k-value num">{num(stats.boxes)}</span>
      <span class="k-sub num">{images ? (stats.boxes / images).toFixed(1) : '—'} per image · {dataset.classes.length} classes</span>
    </div>
    <div class="kpi">
      <span class="k-label">Empty images</span>
      <span class="k-value num">{num(stats.empty_images)}</span>
      <span class="k-sub num">{pct(stats.empty_images)}</span>
    </div>
    <div class="kpi">
      <span class="k-label">Broken labels</span>
      <span class="k-value num" class:bad={stats.broken_labels > 0}>{num(stats.broken_labels)}</span>
      <span class="k-sub">{stats.broken_labels > 0 ? 'skipped in training' : 'all labels parse'}</span>
    </div>
  </section>

  <Panel title="Class balance" meta="boxes per class · {ratio.toFixed(1)}× largest / smallest" pad="none" id="ds-balance">
    <ul class="bars">
      {#each balance as b (b.cls)}
        <li>
          <button type="button" class:absent={b.count === 0} onclick={() => onclass(b.cls)} aria-label="{b.name}: {num(b.count)} boxes, {share(b.count).toFixed(1)}%. Show images">
            <span class="sw" style:background={classInk(b.cls)}></span>
            <span class="cname truncate">{b.name}</span>
            <span class="track"><span class="fill sw" style:width="{(b.count / maxCount) * 100}%" style:background={classInk(b.cls)}></span></span>
            <span class="count num">{num(b.count)}</span>
            <span class="pct num">{share(b.count).toFixed(1)}<small>%</small></span>
            <span class="go" aria-hidden="true"><ArrowRight size={12} /></span>
          </button>
        </li>
      {/each}
    </ul>
  </Panel>

  <div class="two">
    <Panel title="Box size" meta="√area as % of image side" id="ds-size">
      <Histogram
        edges={sizeEdges}
        counts={stats.box_size_hist.counts}
        binLabel={(lo, hi) => sizeBinLabel(lo, hi)}
        label="Histogram of box sizes relative to the image side (square root of the area share), with bins drawn at their true widths and a marker at the 5 % tiny-box threshold"
        color="var(--text-tertiary)"
        hoverColor="var(--text-primary)"
        height={180}
        proportional
        xMax={sizeMax}
        edgeLabel={pctEdge}
        densityPer={0.01}
        marker={{ x: TINY_SIZE, label: 'tiny' }}
      />
    </Panel>
    <Panel title="Boxes per image" meta="images" id="ds-per">
      <Histogram
        edges={stats.boxes_per_image_hist.edges}
        counts={stats.boxes_per_image_hist.counts}
        binLabel={(lo, hi, i) => countBinLabel(lo, hi, i === perLast)}
        label="Histogram of the number of boxes per image"
        color="var(--text-tertiary)"
        hoverColor="var(--text-primary)"
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
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  :global([data-theme='dark']) .kpis {
    border-color: var(--line-subtle);
  }
  .kpi {
    display: grid;
    grid-template-rows: 16px 32px 16px;
    gap: 6px;
    padding: var(--space-4);
    background: var(--bg-panel);
    min-width: 0;
  }
  .k-label,
  .k-sub {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .k-value {
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .k-value.bad {
    color: var(--danger-text);
  }
  .two {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
  }
  .bars {
    padding: 0 var(--space-2) var(--space-2);
  }
  .bars button {
    width: 100%;
    display: grid;
    grid-template-columns: 10px minmax(72px, 160px) minmax(0, 1fr) 64px 48px 12px;
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
    font-size: var(--type-sm-size);
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
  }
  .health li {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .hicon {
    flex-shrink: 0;
    display: grid;
    place-items: center;
    height: 20px;
  }
  .danger .hicon {
    color: var(--danger-text);
  }
  .warning .hicon {
    color: var(--warning-text);
  }
  .info .hicon {
    color: var(--text-tertiary);
  }
  .htext {
    flex: 1;
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .htitle {
    font-weight: var(--weight-medium);
  }
  .hdetail {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .haction {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 20px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    white-space: nowrap;
  }
  .haction:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  @media (pointer: coarse) {
    .bars button {
      height: 44px;
    }
    .haction {
      min-height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .kpis {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 767px) {
    .kpi {
      gap: 2px;
      padding: var(--space-3);
    }
    .two {
      grid-template-columns: minmax(0, 1fr);
    }
    .health li {
      flex-wrap: wrap;
    }
    .haction {
      flex-basis: 100%;
      margin-left: 28px;
    }
    .hide-sm {
      display: none;
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
