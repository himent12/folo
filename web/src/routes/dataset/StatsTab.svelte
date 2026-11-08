<script lang="ts">
  /**
   * Dataset statistics. A compact health summary comes first: one line per severity (errors, warnings, notes) naming
   * its findings, each opening to the evidence. Then four numbers, the class balance as the focal chart, the two
   * histograms in ink, and last the `folo dataset check` issues in their own section with severity and kind filters.
   */
  import IssueList from './IssueList.svelte';
  import ArrowDown from '@lucide/svelte/icons/arrow-down';
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
  import { bySeverity, countBinLabel, healthChecks, imageCount, reviewCount, sizeBinLabel, TINY_SIZE, type Severity } from './health';

  let {
    dataset,
    stats,
    counts,
    onclass,
  }: { dataset: Dataset; stats: DatasetStats; counts?: IssueCounts; onclass: (cls: number) => void } = $props();

  const images = $derived(imageCount(dataset));
  const health = $derived(healthChecks(dataset, stats, counts));
  const issues = $derived(reviewCount(health));
  const groups = $derived(bySeverity(health));
  const notes = $derived(groups.find((g) => g.severity === 'info')?.items.length ?? 0);
  const ok = $derived(health.find((h) => h.severity === 'ok'));
  const summary = $derived(
    [issues ? `${issues} to review` : 'nothing to fix', notes ? `${notes} ${notes === 1 ? 'note' : 'notes'}` : ''].filter(Boolean).join(' · '),
  );
  // Each severity line starts folded: the summary is the point, the evidence one tap away.
  let open = $state<Partial<Record<Severity, boolean>>>({});

  // The check's issues get a section of their own (where the server ran the check, or labels are broken); its filter
  // starts on the most severe level that has issues, and a finding backed by them moves it there.
  const checkShown = $derived(stats.broken_labels > 0 || (!!counts && counts.error + counts.warning + counts.info > 0));
  let listed = $state<IssueSeverity | null>(null);
  const listSeverity = $derived<IssueSeverity>(
    listed ?? (stats.broken_labels > 0 || (counts?.error ?? 0) > 0 ? 'error' : (counts?.warning ?? 0) > 0 ? 'warning' : 'info'),
  );
  function showIssues(sev: IssueSeverity) {
    listed = sev;
    document.getElementById('ds-check')?.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }
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
  const showLabel = (sev: IssueSeverity) => (sev === 'error' ? 'Show the files' : sev === 'warning' ? 'Show the warnings' : 'Show the notes');
</script>

<div class="stats">
  <Panel title="Health" meta={summary} pad="none" id="ds-health">
    <ul class="health">
      {#each groups as g (g.severity)}
        {@const Icon = icons[g.severity]}
        <li class={g.severity}>
          <details open={open[g.severity] ?? false} ontoggle={(e) => (open[g.severity] = e.currentTarget.open)}>
            <summary>
              <span class="hicon"><Icon size={16} strokeWidth={1.75} aria-hidden="true" /></span>
              <span class="hlabel">{g.label}</span>
              <!-- Folded: the findings' titles. Open: the titles head the evidence below, so the line just counts them. -->
              <span class="htitles truncate">{open[g.severity] ? `${g.items.length} ${g.items.length === 1 ? 'finding' : 'findings'}` : g.items.map((h) => h.title).join(' · ')}</span>
              <span class="hchev" aria-hidden="true"><ChevronDown size={14} strokeWidth={1.75} /></span>
            </summary>
            <ul class="hitems">
              {#each g.items as h (h.id)}
                <li>
                  <p class="htitle">{h.title}</p>
                  <p class="hdetail">{h.detail}</p>
                  {#if h.issues || h.action}
                    <p class="hacts">
                      {#if h.issues}
                        {@const sev = h.issues}
                        <button type="button" class="haction" onclick={() => showIssues(sev)}
                          >{showLabel(sev)}<ArrowDown size={12} strokeWidth={2} aria-hidden="true" /></button
                        >
                      {/if}
                      {#if h.action}
                        <a class="haction" href={href(h.action.path)}>{h.action.label}<ArrowRight size={12} strokeWidth={2} aria-hidden="true" /></a>
                      {/if}
                    </p>
                  {/if}
                </li>
              {/each}
            </ul>
          </details>
        </li>
      {:else}
        {#if ok}
          <li class="ok">
            <div class="okrow">
              <span class="hicon"><CircleCheck size={16} strokeWidth={1.75} aria-hidden="true" /></span>
              <span class="hlabel">{ok.title}</span>
              <span class="htitles truncate">{ok.detail}</span>
            </div>
          </li>
        {/if}
      {/each}
    </ul>
  </Panel>

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
    <Panel title="Box size" meta="how big objects are next to the image" id="ds-size">
      {#snippet actions()}
        <!-- The precise reading, one step away: the plain line above is what most readers need. -->
        <button
          type="button"
          class="how"
          aria-label="How box size is measured"
          use:tooltip={{ text: "Size = √(box area ÷ image area): 10% is a box about a tenth of the image's side. Bars span their real size range and their height is boxes per 1 percentage point of size, so a bar's area is its count.", side: 'bottom' }}
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
        yLabel="Boxes per 1 percentage point of size"
        note="bar area is the count"
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

  {#if checkShown}
    <!-- The evidence: every file and line the check flagged, with the checker's message and its fix. -->
    <Panel title="Dataset check findings" meta="from folo dataset check" id="ds-check">
      <IssueList datasetId={dataset.id} path={dataset.path} bind:severity={() => listSeverity, (v) => (listed = v)} {counts} />
    </Panel>
  {/if}
</div>


<style>
  .stats {
    display: grid;
    gap: var(--space-4);
  }
  /* "Show the files" scrolls the check's section to just under the top bar. */
  .stats :global(#ds-check) {
    scroll-margin-top: calc(var(--size-topbar) + var(--space-4));
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
  /* One 40 px line per severity: icon, the level as a word, its findings' titles; it opens to the evidence. */
  .health > li {
    border-top: 1px solid var(--line-subtle);
  }
  summary,
  .okrow {
    display: grid;
    grid-template-columns: 16px 72px minmax(0, 1fr) 14px;
    align-items: center;
    gap: var(--space-3);
    min-height: 40px;
    padding: 0 var(--space-4);
  }
  .okrow {
    grid-template-columns: 16px auto minmax(0, 1fr);
  }
  summary {
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
  .health > li:last-child details:not([open]) summary {
    border-radius: 0 0 calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px);
  }
  .hicon {
    display: grid;
    place-items: center;
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
  .ok .hicon {
    color: var(--success-text);
  }
  .hlabel {
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .htitles {
    color: var(--text-secondary);
  }
  /* Errors keep their weight in the summary: the titles in ink, not grey. */
  .danger .htitles {
    color: var(--text-primary);
  }
  .info .hlabel,
  .ok .htitles {
    color: var(--text-secondary);
  }
  details[open] .htitles {
    color: var(--text-tertiary);
  }
  .hchev {
    display: flex;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  details[open] .hchev {
    transform: rotate(180deg);
  }
  /* The evidence lines up under the titles column: 16 px padding + 16 px icon + 72 px label + two 12 px gaps. */
  .hitems {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-1) var(--space-4) var(--space-4) calc(var(--space-16) * 2);
  }
  .hitems li {
    display: grid;
    gap: 2px;
    max-width: 72ch;
  }
  .htitle {
    font-weight: var(--weight-medium);
  }
  .hdetail {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .hacts {
    display: flex;
    flex-wrap: wrap;
    gap: 0 var(--space-4);
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
    .hacts {
      margin-top: calc(var(--space-2) * -1);
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
    /* Phones: the level's word shrinks to fit, and the evidence starts under the icon's right edge. */
    summary {
      grid-template-columns: 16px auto minmax(0, 1fr) 14px;
      gap: var(--space-2);
    }
    .okrow {
      gap: var(--space-2);
    }
    .hitems {
      padding-left: 40px;
    }
    .kpi {
      grid-template-rows: 16px 28px 16px;
    }
    .k-value {
      font-size: var(--type-xl-size);
      line-height: 28px;
      letter-spacing: var(--type-xl-tracking);
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
    .bars button {
      grid-template-columns: 10px minmax(56px, 88px) minmax(0, 1fr) 48px 44px;
      gap: var(--space-2);
    }
    .go {
      display: none;
    }
  }
</style>
