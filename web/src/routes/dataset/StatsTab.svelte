<script lang="ts">
  /**
   * Dataset statistics. Health comes first, only when there are findings: one compact line across the page (each
   * severity's icon and a short summary of its findings, units named; Details at its right end) that opens to every
   * finding with its next step, in two columns from the laptop tier. Open, the line is a plain heading ("Health ·
   * 4 checks"): the findings under it say what the summary said, so it is never said twice. A finding backed by `folo dataset check` issues opens them under itself: the one issue
   * list, filtered to that finding's severity (and inside it, by check type). Then four numbers, the class balance as
   * the focal chart, and the two histograms in ink.
   */
  import IssueList from './IssueList.svelte';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Info from '@lucide/svelte/icons/info';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { Dataset, DatasetStats, IssueCounts, IssueSeverity } from '$lib/api/types';
  import Histogram from '$lib/charts/Histogram.svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { num } from '$lib/format';
  import { href } from '$lib/router.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import { swatchStyle } from './swatch';
  import { bySeverity, countBinLabel, groupSummary, healthChecks, imageCount, sizeBinLabel, TINY_SIZE, type Severity } from './health';

  let {
    dataset,
    stats,
    counts,
    onclass,
  }: { dataset: Dataset; stats: DatasetStats; counts?: IssueCounts; onclass: (cls: number) => void } = $props();

  const images = $derived(imageCount(dataset));
  const health = $derived(healthChecks(dataset, stats, counts));
  const groups = $derived(bySeverity(health));
  const findings = $derived(health.filter((h) => h.severity !== 'ok'));
  // The line is the point; the evidence is one tap away.
  let healthOpen = $state(false);

  // The finding whose check issues are open under it (one at a time: the list is the evidence of the finding it sits
  // under, never a second copy elsewhere).
  let evidence = $state<string | null>(null);
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
  const sevWord: Record<Severity, string> = { danger: 'Error', warning: 'Warning', info: 'Note', ok: 'OK' };
  const perLast = $derived(stats.boxes_per_image_hist.counts.length - 1);
  const classCount = $derived(`${dataset.classes.length} ${dataset.classes.length === 1 ? 'class' : 'classes'}`);
  const pct = (x: number) => (images ? `${((x / images) * 100).toFixed(1)}% of images` : '');
  const evidenceNoun = (sev: IssueSeverity) => (sev === 'error' ? 'files' : sev === 'warning' ? 'warnings' : 'images');

  /**
   * Sets `wrap` on a line that would not fit its box on one line, so it can break between its parts instead of being
   * cut off; measured on every resize, without the class.
   */
  function wrapWhenNarrow(node: HTMLElement) {
    const ro = new ResizeObserver(() => {
      node.classList.remove('wrap');
      node.classList.toggle('wrap', node.scrollWidth > node.clientWidth + 1);
    });
    ro.observe(node);
    return { destroy: () => ro.disconnect() };
  }
</script>

<div class="stats">
  {#if groups.length}
    <!-- One line at rest: each severity's icon and what its findings are; it opens to every finding and its next step. -->
    <section class="health" id="ds-health" aria-labelledby="ds-health-title">
      <details bind:open={healthOpen}>
        <summary>
          <span class="h-title" id="ds-health-title">Health</span>
          {#if healthOpen}
            <span class="h-count num">{findings.length} {findings.length === 1 ? 'check' : 'checks'}</span>
          {:else}
            <span class="h-sum">
              {#each groups as g (g.severity)}
                {@const Icon = icons[g.severity]}
                <span class="h-sev {g.severity}">
                  <Icon size={14} strokeWidth={2} aria-hidden="true" /><span class="sr-only">{g.label}:</span>{groupSummary(g)}
                </span>
              {/each}
            </span>
          {/if}
          <span class="h-more" aria-hidden="true">{healthOpen ? 'Hide' : 'Details'}<ChevronDown size={14} strokeWidth={1.75} /></span>
        </summary>
        <ul class="hitems">
          {#each findings as h (h.id)}
            {@const Icon = icons[h.severity]}
            <li class={h.severity}>
              <span class="hicon"><Icon size={16} strokeWidth={1.75} aria-hidden="true" /></span>
              <div class="htext">
                <p class="htitle"><span class="sr-only">{sevWord[h.severity]}: </span>{h.title}</p>
                <p class="hdetail">{h.detail}</p>
                {#if h.issues || h.action}
                  <p class="hacts">
                    {#if h.issues}
                      {@const open = evidence === h.id}
                      <button
                        type="button"
                        class="haction"
                        aria-expanded={open}
                        aria-controls="ds-ev-{h.id}"
                        onclick={() => (evidence = open ? null : h.id)}
                        >{open ? 'Hide' : 'Show'} the {evidenceNoun(h.issues)}<ChevronDown class="ev-chev" size={12} strokeWidth={2} aria-hidden="true" /></button
                      >
                    {/if}
                    {#if h.action}
                      <a class="haction" href={href(h.action.path)}>{h.action.label}<ArrowRight size={12} strokeWidth={2} aria-hidden="true" /></a>
                    {/if}
                  </p>
                {/if}
              </div>
            </li>
            {#if h.issues && evidence === h.id}
              <!-- The evidence: every file and line the check flagged, with its fix, in a full-width row of its own under
                   its finding's row (the grid packs the next finding into the cell beside it), never a card that
                   spans both columns among half-width ones. -->
              <li class="evidence" id="ds-ev-{h.id}">
                <IssueList datasetId={dataset.id} path={dataset.path} severity={h.issues} severityFilter={false} {counts} />
              </li>
            {/if}
          {/each}
        </ul>
      </details>
    </section>
  {/if}

  <!-- 1 px gaps over a hairline-coloured background draw the dividers in any column count. -->
  <section class="kpis" aria-label="Summary">
    <div class="kpi">
      <span class="k-label">Images</span>
      <span class="k-value num">{num(images)}</span>
      <!-- The split counts break between train and the rest where one line would cut them off; tablets and phones show
           the class count instead. -->
      <span class="k-sub num splits" use:wrapWhenNarrow
        ><span class="wide"><span class="seg">{num(dataset.splits.train)} train<span class="sep">{' · '}</span></span><span class="seg"
            >{num(dataset.splits.val)} val · {num(dataset.splits.test)} test</span
          ></span
        ><span class="narrow">{classCount}</span></span
      >
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

  <Panel title="Class balance" meta="boxes per class · {ratio.toFixed(1)}× largest / smallest" pad="none" id="ds-balance">
    <ul class="bars">
      {#each balance as b (b.cls)}
        <li>
          <button type="button" class:absent={b.count === 0} onclick={() => onclass(b.cls)} aria-label="{b.name}: {num(b.count)} boxes, {share(b.count).toFixed(1)}%. Show images">
            <span class="sw" style={swatchStyle(b.cls, b.count === 0)}></span>
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
    <Panel title="Box size" meta="% of the image side" id="ds-size">
      {#snippet actions()}
        <!-- The precise reading, one step away: the plain line above is what most readers need. -->
        <button
          type="button"
          class="how"
          aria-label="How box size is measured"
          use:tooltip={{ text: "Size = √(box area ÷ image area): 10% is a box about a tenth of the image's side. One bar per size range, finer near the 5% tiny-box line; a bar's height is its number of boxes.", side: 'bottom' }}
          ><Info size={14} aria-hidden="true" /></button
        >
      {/snippet}
      <Histogram
        edges={stats.box_size_hist.edges}
        counts={stats.box_size_hist.counts}
        binLabel={(lo, hi) => sizeBinLabel(lo, hi)}
        tickLabel={(lo, hi) => sizeBinLabel(lo, hi).replace('%', '')}
        label="Histogram of box sizes relative to the image side (square root of the area share), one bar per size range, with a marker at the 5% tiny-box threshold"
        color="var(--text-tertiary)"
        hoverColor="var(--text-primary)"
        height={180}
        marker={{ x: TINY_SIZE, label: 'tiny' }}
        yLabel="Boxes"
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
        yLabel="Images"
      />
    </Panel>
  </div>

</div>

<style>
  .stats {
    display: grid;
    gap: var(--space-4);
  }
  /* Health: full width like the panels under it, the summary right after the title and Details at the right end. */
  .health {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
  }
  /* Baseline: when the summaries wrap at the laptop tier, the title and Details stay on the first line. */
  summary {
    display: flex;
    align-items: baseline;
    gap: var(--space-2) var(--space-4);
    min-height: 40px;
    padding: var(--space-2) var(--space-3) var(--space-2) var(--space-4);
    border-radius: calc(var(--radius-lg) - 1px);
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
  details[open] summary {
    border-radius: calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px) 0 0;
  }
  .h-title {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .h-count {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .h-sum {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-4);
    min-width: 0;
  }
  .h-sev {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .h-sev :global(svg),
  .hicon {
    flex-shrink: 0;
  }
  .h-sev.danger {
    color: var(--text-primary);
  }
  .danger .hicon,
  .h-sev.danger :global(svg) {
    color: var(--danger-text);
  }
  .warning .hicon,
  .h-sev.warning :global(svg) {
    color: var(--warning-text);
  }
  .info .hicon,
  .h-sev.info :global(svg) {
    color: var(--text-tertiary);
  }
  .h-more {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .h-more :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  details[open] .h-more :global(svg) {
    transform: rotate(180deg);
  }
  /* The evidence: each finding with its severity icon, its body at a 60-character measure, its next step under it. */
  .hitems {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4) var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .hitems li {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr);
    gap: var(--space-3);
    align-items: start;
  }
  /* From the laptop tier, the findings in two columns (row by row, so severity order still reads left to right, top to
     bottom). Open evidence is a full-width row of its own under its finding's row: dense packing moves the next
     finding up beside the opened one, so the cards keep their two-column grid. */
  @media (min-width: 1024px) {
    .hitems {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-auto-flow: row dense;
      gap: var(--space-4) var(--space-8);
    }
  }
  .hicon {
    display: grid;
    place-items: center;
    height: 20px;
  }
  .htext {
    display: grid;
    justify-items: start;
    gap: 2px;
    min-width: 0;
  }
  /* Full width, its left edge on the findings' text (past the 16 px icon and its 12 px gap). */
  .hitems li.evidence {
    display: block;
    grid-column: 1 / -1;
    min-width: 0;
    margin-top: calc(-1 * var(--space-2));
    padding-left: calc(16px + var(--space-3));
  }

  .haction :global(.ev-chev) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .haction[aria-expanded='true'] :global(.ev-chev) {
    transform: rotate(180deg);
  }
  .htitle {
    font-weight: var(--weight-medium);
  }
  .hdetail {
    max-width: 60ch;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .hacts {
    display: flex;
    flex-wrap: wrap;
    gap: 0 var(--space-4);
    margin-top: 2px;
  }
  .haction {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    min-height: 24px;
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
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    background-clip: padding-box;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
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
  .kpi {
    display: grid;
    grid-template-rows: 16px 32px minmax(16px, auto);
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
  /* Wrapped: train on the first line, val and test on the second (the row grows; the other cells keep their top). */
  .splits:global(.wrap) {
    white-space: normal;
  }
  .splits:global(.wrap) .seg {
    display: block;
  }
  .splits:global(.wrap) .sep {
    display: none;
  }
  .k-value.bad {
    color: var(--danger-text);
  }
  .two {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
  }
  /* The focal panel: the bars a step taller than the list rows around them. */
  .bars {
    padding: 0 var(--space-2) var(--space-2);
  }
  .bars button {
    width: 100%;
    display: grid;
    grid-template-columns: 10px minmax(72px, 160px) minmax(0, 1fr) 64px 48px 12px;
    align-items: center;
    gap: var(--space-3);
    height: 36px;
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
  .cname {
    font-weight: var(--weight-medium);
  }
  .track {
    height: 12px;
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
  .narrow {
    display: none;
  }
  @media (pointer: coarse) {
    .how {
      width: 44px;
      height: 44px;
    }
    .bars button,
    summary {
      min-height: 44px;
    }
    .haction {
      min-height: 44px;
    }
    .hacts {
      margin-top: calc(var(--space-2) * -1);
      margin-bottom: calc(var(--space-2) * -1);
    }
  }
  /* Tablets and phones: the histograms stack, each at full width; Health takes the full width too. The KPI captions
     take their short form (the split counts are in the page header's meta line), so every cell stays one line. */
  @media (max-width: 1023px) {
    .two {
      grid-template-columns: minmax(0, 1fr);
    }
    .wide {
      display: none;
    }
    .narrow {
      display: inline;
    }
  }
  @media (max-width: 767px) {
    /* Phones, at rest: the title and Details on the first line, the severity summaries wrapping under them. Open, the
       heading and its count fit the one line. */
    details:not([open]) summary {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      grid-template-areas:
        'title more'
        'sum sum';
      gap: var(--space-1) var(--space-3);
      padding-block: var(--space-2) var(--space-3);
    }
    .h-title {
      grid-area: title;
    }
    .h-more {
      grid-area: more;
    }
    .h-sum {
      grid-area: sum;
      gap: var(--space-1) var(--space-3);
    }
    .h-sev {
      white-space: normal;
    }
    /* A compact 2 × 2 grid: no hero tile, so the class balance starts on the first screen. */
    .kpi {
      grid-template-rows: 16px 28px minmax(16px, auto);
      gap: 2px;
      padding: var(--space-3);
    }
    .k-value {
      font-size: var(--type-xl-size);
      line-height: 28px;
      letter-spacing: var(--type-xl-tracking);
    }
    .kpis {
      grid-template-columns: repeat(2, minmax(0, 1fr));
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
