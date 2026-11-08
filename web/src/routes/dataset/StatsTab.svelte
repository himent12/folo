<script lang="ts">
  /**
   * Dataset statistics. Findings to review come first, and only when there are any; then four numbers, the class
   * balance as the focal chart, and the two histograms in ink. On phones the findings start collapsed under a one-line
   * summary, so the class balance stays on the first screen.
   */
  import IssueList from './IssueList.svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import ArrowUp from '@lucide/svelte/icons/arrow-up';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Info from '@lucide/svelte/icons/info';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { tick } from 'svelte';
  import type { Dataset, DatasetStats, IssueCounts, IssueSeverity } from '$lib/api/types';
  import Histogram from '$lib/charts/Histogram.svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { num } from '$lib/format';
  import { href } from '$lib/router.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import { countBinLabel, healthChecks, imageCount, reviewCount, sizeBinLabel, TINY_SIZE, type Severity } from './health';

  let {
    dataset,
    stats,
    counts,
    onclass,
  }: { dataset: Dataset; stats: DatasetStats; counts?: IssueCounts; onclass: (cls: number) => void } = $props();

  const images = $derived(imageCount(dataset));
  const health = $derived(healthChecks(dataset, stats, counts));
  const issues = $derived(reviewCount(health));
  const findings = $derived(health.filter((h) => h.severity !== 'ok'));
  const notes = $derived(findings.length - issues);
  // One list of the check's issues, under the first finding they back; the others move its filter to their severity.
  const host = $derived(findings.find((h) => h.issues));
  let listed = $state<IssueSeverity | null>(null);
  const listSeverity = $derived(listed ?? host?.issues ?? 'error');
  async function showIssues(sev: IssueSeverity) {
    if (!host) return;
    listed = sev;
    toggled[host.id] = true;
    await tick();
    document.getElementById('ds-issues')?.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }
  // Findings start open on wide screens and collapsed on phones; once toggled, a finding keeps its own state.
  const phone = new MediaQuery('max-width: 767px', false);
  let toggled = $state<Record<string, boolean>>({});
  const sizeEdges = $derived(stats.box_size_hist.edges);
  // Unequal size bins keep their true widths up to 60% of the side; the open last bin (≥ 50%) is drawn up to there.
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
  const splitLine = $derived(`${num(dataset.splits.train)} train · ${num(dataset.splits.val)} val · ${num(dataset.splits.test)} test`);
  const classCount = $derived(`${dataset.classes.length} ${dataset.classes.length === 1 ? 'class' : 'classes'}`);
  const pct = (x: number) => (images ? `${((x / images) * 100).toFixed(1)}% of images` : '');
</script>

<div class="stats">
  {#if issues > 0}
    <Panel pad="none" id="ds-health">
      <h2 class="hsum num">
        {issues} to review{#if notes > 0}<span class="hnotes"><span aria-hidden="true">·</span> {notes} {notes === 1 ? 'note' : 'notes'}</span>{/if}
      </h2>
      <ul class="health">
        {#each findings as h (h.id)}
          {@const Icon = icons[h.severity]}
          <li class={h.severity}>
            <details open={toggled[h.id] ?? !phone.current} ontoggle={(e) => (toggled[h.id] = e.currentTarget.open)}>
              <summary>
                <span class="hicon"><Icon size={16} strokeWidth={1.75} aria-label={h.severity} /></span>
                <span class="htitle">{h.title}</span>
                <span class="hchev" aria-hidden="true"><ChevronDown size={14} strokeWidth={1.75} /></span>
              </summary>
              <div class="hbody">
                <p class="hdetail">{h.detail}</p>
                {#if h.issues && h.id === host?.id}
                  <!-- The evidence: each file and line with the checker's message and its fix. -->
                  <div class="hlist" id="ds-issues">
                    <IssueList datasetId={dataset.id} path={dataset.path} bind:severity={() => listSeverity, (v) => (listed = v)} {counts} />
                  </div>
                {:else if h.issues}
                  <button type="button" class="haction" onclick={() => h.issues && showIssues(h.issues)}
                    >List the {h.issues === 'error' ? 'errors' : 'warnings'}<ArrowUp size={12} strokeWidth={2} aria-hidden="true" /></button
                  >
                {/if}
                {#if h.action}
                  <a class="haction" href={href(h.action.path)}>{h.action.label}<ArrowRight size={12} strokeWidth={2} aria-hidden="true" /></a>
                {/if}
              </div>
            </details>
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
      <span class="k-sub num"><span class="wide">{splitLine}</span><span class="narrow">{classCount}</span></span>
    </div>
    <div class="kpi">
      <span class="k-label">Boxes</span>
      <span class="k-value num">{num(stats.boxes)}</span>
      <span class="k-sub num">{images ? (stats.boxes / images).toFixed(1) : '—'} per image<span class="wide">{` · ${classCount}`}</span></span>
    </div>
    <div class="kpi">
      <span class="k-label">Empty images</span>
      <span class="k-value num">{num(stats.empty_images)}</span>
      <span class="k-sub num">{pct(stats.empty_images)}</span>
    </div>
    <div class="kpi">
      <span class="k-label">Broken labels</span>
      <span class="k-value num" class:bad={stats.broken_labels > 0}>{num(stats.broken_labels)}</span>
      <span class="k-sub">{stats.broken_labels > 0 ? 'not trained on' : 'all labels parse'}</span>
    </div>
  </section>
  <!-- Phones: the split counts get a full line of their own instead of truncating inside the Images cell. -->

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
    <Panel title="Box size" meta="how big objects are next to the image" id="ds-size">
      {#snippet actions()}
        <!-- The precise reading, one step away: the plain line above is what most readers need. -->
        <button
          type="button"
          class="how"
          aria-label="How box size is measured"
          use:tooltip={{ text: "Size = √(box area ÷ image area): 10% is a box about a tenth of the image's side. Bars span their real size range, so a bar's area is its count.", side: 'bottom' }}
          ><Info size={14} aria-hidden="true" /></button
        >
      {/snippet}
      <Histogram
        edges={sizeEdges}
        counts={stats.box_size_hist.counts}
        binLabel={(lo, hi) => sizeBinLabel(lo, hi)}
        label="Histogram of box sizes relative to the image side (square root of the area share), with bins drawn at their true widths and a marker at the 5% tiny-box threshold"
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
    <Panel title="Boxes per image" meta="how many images have each count" id="ds-per">
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
  .how {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .how:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .how {
      width: 44px;
      height: 44px;
    }
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
  /* The panel header is the severity summary: a 40 px row like every panel title. */
  .hsum {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 40px;
    padding: 0 var(--space-4);
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    letter-spacing: var(--type-md-tracking);
    font-weight: var(--weight-medium);
  }
  .hnotes {
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .health {
    display: grid;
  }
  .health li {
    border-top: 1px solid var(--line-subtle);
  }
  summary {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 40px;
    padding: 0 var(--space-4);
    cursor: pointer;
    list-style: none;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary:hover {
    background: var(--bg-hover);
  }
  summary:focus-visible {
    outline-offset: -2px;
  }
  .health li:last-child details:not([open]) summary {
    border-radius: 0 0 calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px);
  }
  .hchev {
    display: flex;
    margin-left: auto;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  details[open] .hchev {
    transform: rotate(180deg);
  }
  /* The detail lines up under the title: 16 px padding + 16 px icon + 12 px gap. */
  .hbody {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: 0 var(--space-4) var(--space-3) 44px;
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
  .htitle {
    min-width: 0;
    padding: var(--space-2) 0;
    font-weight: var(--weight-medium);
  }
  .hdetail {
    flex: 1;
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* The issue list takes two thirds of the row beside the detail. */
  .hlist {
    flex: 2 1 0;
    min-width: 0;
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
  .narrow {
    display: none;
  }
  @media (pointer: coarse) {
    .bars button {
      height: 44px;
    }
    summary {
      min-height: 44px;
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
    .hbody {
      flex-wrap: wrap;
    }
    .haction,
    .hlist {
      flex-basis: 100%;
    }
    /* Phones: Images spans the grid so the split counts stay inside their tile; the other three share one row. */
    .kpis {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .kpi:first-child {
      grid-column: 1 / -1;
    }
    .kpi:not(:first-child) .wide {
      display: none;
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
