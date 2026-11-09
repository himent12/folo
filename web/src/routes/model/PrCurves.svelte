<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  /**
   * Precision-recall curves with the per-class table as their legend. Every class is a 1 px context line
   * (`--chart-context`, >= 3:1 on the panel) and the mean over all classes a heavier line in ink; many classes are a
   * percentile band instead, so 80 classes read as one shape instead of spaghetti. Hovering a table row gives its curve
   * the class hue, clicking keeps it highlighted (the selection rule: bg.active and a 2 px accent bar), and a line
   * above the table says so and offers Clear. The legend row above the plot is the readout. Phones hide the precision
   * and recall columns: each row has two controls there, the class name traces its curve (`aria-pressed`) and the
   * chevron opens a details row with those two at their operating point (`aria-expanded`), with "View curve"
   * (`onviewcurve`). `view` shows only the curves or only the table: the model page renders them as two panels (the
   * curves beside the Details rail, the table in a row of its own; on a phone, two sections) that share `selected` and
   * `hovered` (bindable), so a class picked or pointed at in the table is traced in the other panel's curves, and
   * `onback` puts a way back to the table in the curves' legend row. `fill` lets the plot grow to the panel's height
   * (a panel stretched to its row), from 300 px.
   */
  import type { Evaluation } from '$lib/api/evaluation';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { classInk, classStroke } from '$lib/design/classInk.svelte';
  import { metric, num } from '$lib/format';
  import Input from '$lib/ui/Input.svelte';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import Search from '@lucide/svelte/icons/search';
  import { classMetrics, matchesClass, prContext, prEnvelope, trimPrTail } from './pr';
  import { media } from '$lib/design/tokens';

  let {
    classes,
    ev,
    map50,
    view = 'all',
    onviewcurve,
    onback,
    fill = false,
    selected = $bindable([]),
    hovered = $bindable(null),
    allCurves = $bindable(false),
    modeInHeader = false,
  }: {
    classes: readonly string[];
    ev: Evaluation;
    map50: number | null;
    view?: 'all' | 'curves' | 'classes';
    /** A details row's "View curve": the class is traced, then the page shows the curves. */
    onviewcurve?: (cls: number) => void;
    /** Back to the class table, offered in the curves' legend row (after "View curve"). */
    onback?: () => void;
    /** The plot takes the height left in its panel (at least 300 px). */
    fill?: boolean;
    /** Classes traced in their hue (clicked rows), shared between a curves view and a table view. */
    selected?: number[];
    /** The class under the pointer or keyboard focus in the table. */
    hovered?: number | null;
    /** Many classes: every curve instead of the 10th–90th percentile band. */
    allCurves?: boolean;
    /** The band / all-curves switch sits in the panel's header (the page renders it), not at the legend row's end. */
    modeInHeader?: boolean;
  } = $props();
  /** The plot's box when it fills its panel: the chart is drawn at that height. */
  let plotH = $state(0);

  // Phones get a shorter plot so the table (the legend) starts on the first screen.
  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = $derived(phoneQuery.current);

  // Few classes: a leaderboard by AP. Many: by frequency.
  const many = $derived(classes.length > 10);
  const order = $derived(ev.perClass.map((c) => c).sort((a, b) => (many ? b.instances - a.instances : b.ap50 - a.ap50)));

  const toggle = (i: number) => (selected = selected.includes(i) ? selected.filter((x) => x !== i) : [...selected, i]);
  /** Phones: the classes whose details row is open (apart from tracing). */
  let opened = $state<number[]>([]);
  const disclose = (i: number) => (opened = opened.includes(i) ? opened.filter((x) => x !== i) : [...opened, i]);
  function viewCurve(i: number) {
    if (!selected.includes(i)) selected = [...selected, i];
    onviewcurve?.(i);
  }
  /** Classes drawn in their hue, in legend order: the selection, then the hovered row. */
  const lit = $derived(hovered !== null && !selected.includes(hovered) ? [...selected, hovered] : selected);

  const name = (i: number) => classes[i] ?? `class ${i}`;
  // Curves end at their last recall point with precision > 0; the zero padding up to recall 1 is not data.
  // Many classes: by default a band (10th–90th percentile of the classes' precision) instead of 80 lines; every
  // curve is one click away. Few classes: every curve, as context lines.
  const banded = $derived(many && !allCurves);
  const envelope = $derived(many ? prEnvelope(classes.map((_, i) => trimPrTail(ev.curves[i] ?? []))) : null);
  const series = $derived<Series[]>([
    ...(banded && envelope
      ? [{ id: 'band', label: 'P10–P90 of classes', color: 'var(--text-tertiary)', points: envelope.upper, lower: envelope.lower }]
      : classes
          .map((_, i) => i)
          .filter((i) => !lit.includes(i))
          .map((i) => ({ id: `c${i}`, label: name(i), color: 'var(--chart-context)', context: true, points: trimPrTail(ev.curves[i] ?? []) }))),
    { id: 'mean', label: 'All classes', color: 'var(--text-primary)', points: trimPrTail(ev.mean), emphasis: true },
    ...lit.map((i) => ({ id: `c${i}`, label: name(i), color: classStroke(i), points: trimPrTail(ev.curves[i] ?? []), emphasis: true })),
  ]);

  // Cursor on the shared recall axis (bound to the chart); the legend reads each curve's precision there.
  let cursor = $state<number | null>(null);
  function at(points: { x: number; y: number }[]): string {
    const pts = trimPrTail(points);
    const c = cursor;
    if (c === null || !pts.length) return '—';
    const last = pts[pts.length - 1]!;
    if (c > last.x + 1e-9) return '—';
    let best = pts[0]!;
    for (const p of pts) if (Math.abs(p.x - c) < Math.abs(best.x - c)) best = p;
    return metric(best.y);
  }

  const meanAp = $derived(map50 ?? ev.perClass.reduce((a, c) => a + c.ap50, 0) / Math.max(1, ev.perClass.length));
  const apOf = $derived(new Map(ev.perClass.map((c) => [c.class, c.ap50])));

  // Phones show four columns: a selected (highlighted) class also opens a details row with all five metrics.
  const COLS_NARROW = 4;

  // Large class sets: a filter and 16 rows until the reader asks for all of them.
  const ROWS_MAX = 16;
  let query = $state('');
  let all = $state(false);
  const listed = $derived(order.filter((c) => matchesClass(name(c.class), query)));
  const rows = $derived(all || query ? listed : listed.slice(0, ROWS_MAX));
</script>

<div class="pr">
  <div class="part" class:off={view === 'classes'}>
    <!-- At rest the legend states each curve's AP50 (a summary of the whole curve); under the cursor it switches to a
         readout, marked by the cursor's own glyph: the precision at that recall. -->
    <div class="legend" class:reading={cursor !== null}>
      {#if cursor === null}
        <span class="what">AP50 <span class="what-sub">area under each curve</span></span>
      {:else}
        <span class="what at num"><span class="cur" aria-hidden="true"></span>Precision at recall {cursor.toFixed(2)}</span>
      {/if}
      <span class="item"><span class="line mean" aria-hidden="true"></span>All classes <span class="num">{cursor === null ? metric(meanAp) : at(ev.mean)}</span></span>
      {#each lit as i (i)}
        <span class="item"><span class="line" style:background={classStroke(i)} aria-hidden="true"></span>{name(i)} <span class="num">{cursor === null ? metric(apOf.get(i)) : at(ev.curves[i] ?? [])}</span></span>
      {/each}
      {#if banded}<span class="item band-key"><span class="swatch-band" aria-hidden="true"></span>10th–90th percentile of classes</span>{/if}
      <!-- Phones' Curves section has no table beside it: the hint and Clear come along. Elsewhere they head the table. -->
      {#if view === 'curves' && onback}
        <button type="button" class="clear back" onclick={onback}><ArrowLeft size={14} aria-hidden="true" />Back to Classes</button>
      {/if}
      {#if view === 'curves'}
        {#if selected.length}
          <button type="button" class="clear" onclick={() => (selected = [])}>Clear</button>
        {:else if hovered === null && narrow}
          <span class="hint">Tap a class in Classes to trace its curve.</span>
        {/if}
      {/if}
      {#if many && !modeInHeader}
        <button type="button" class="clear mode" aria-pressed={allCurves} onclick={() => (allCurves = !allCurves)}>{allCurves ? 'Show band' : 'All curves'}</button>

      {/if}
    </div>
    <div class="plot" class:fill bind:clientHeight={plotH}>
      <LineChart
        {series}
        height={narrow ? 220 : fill ? Math.max(300, plotH - 24) : 300}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        xFormat={(x) => x.toFixed(1)}
        valueFormat={(y) => y.toFixed(3)}
        xTicks={6}
        yTicks={5}
        legend={false}
        xLabel="Recall"
        yLabel="Precision"
        bind:cursor
        empty="No correct detections yet: precision is 0 at every recall."
        label="Precision-recall curves at IoU 0.5 for {classes.length} classes and the mean over all classes"
      />
    </div>
  </div>

  <div class="part" class:off={view === 'curves'}>
    <!-- The rows are the curves' controls: say so, and how many are traced, with a way back. -->
    <div class="tools">
      {#if many}
        <div class="filter">
          <Input bind:value={query} icon={Search} size="sm" type="search" placeholder="Find a class" aria-label="Find a class" />
        </div>
      {/if}
      <p class="pick num" aria-live="polite">
        {#if selected.length}
          <span>{selected.length} {selected.length === 1 ? 'class' : 'classes'} traced</span>
          <button type="button" class="clear" onclick={() => (selected = [])}>Clear</button>
        {:else}
          <span class="hint">{narrow ? 'Tap a class to trace its curve, the arrow for its details' : 'Select classes to trace their curves'}</span>
        {/if}
      </p>
    </div>
    <div class="scroll">
      <table class="table dense" class:fixed={narrow}>
        <!-- Phones: one column definition for the header and every row (fixed layout), so the headings sit over their
             numbers and the class column ends a clear gutter before AP50. -->
        {#if narrow}
          <colgroup>
            <col />
            <col class="c-ap" />
            <col class="c-ap95" />
            <col class="c-inst" />
          </colgroup>
        {/if}
        <!-- Wider screens: bounded columns (the class 224 px, each number 96 px) and an empty last column that takes
             what is left, so 3 or 80 classes read as one compact block on the left, never figures strewn across the
             panel. -->
        <thead>
          <tr>
            <th scope="col" class="cls-h" style:width={narrow ? undefined : '224px'}>Class</th>
            <th scope="col" class="r" style:width={narrow ? undefined : '96px'}>AP50</th>
            <th scope="col" class="r" style:width={narrow ? undefined : '96px'}>AP50-95</th>
            <th scope="col" class="r hide-sm" style:width={narrow ? undefined : '96px'}>Precision</th>
            <th scope="col" class="r hide-sm" style:width={narrow ? undefined : '96px'}>Recall</th>
            <th scope="col" class="r" style:width={narrow ? undefined : '96px'}>Instances</th>
            {#if !narrow}<th class="rest" aria-hidden="true"></th>{/if}
          </tr>
        </thead>
        <tbody>
          {#each rows as c (c.class)}
            {@const on = selected.includes(c.class)}
            {@const open = narrow && opened.includes(c.class)}
            <tr
              class="link"
              class:on
              class:hot={hovered === c.class}
              onpointerenter={() => (hovered = c.class)}
              onpointerleave={() => (hovered = null)}
              onclick={() => toggle(c.class)}
            >
              <td>
                <!-- The class name traces its curve (a toggle). Phones: the chevron beside it is a separate control that
                     opens the details row. -->
                <div class="cls-cell">
                  <button
                    type="button"
                    class="cls"
                    data-cls={c.class}
                    aria-pressed={on}
                    aria-label="Trace {name(c.class)}"
                    onclick={(e) => (e.stopPropagation(), toggle(c.class))}
                    onfocus={() => (hovered = c.class)}
                    onblur={() => (hovered = null)}
                  >
                    <span class="sw" style:background={classInk(c.class)} aria-hidden="true"></span>
                    <span class="truncate">{name(c.class)}</span>
                  </button>
                  {#if narrow}
                    <button
                      type="button"
                      class="disc"
                      data-cls={c.class}
                      aria-expanded={open}
                      aria-controls="pr-detail-{c.class}"
                      aria-label="{name(c.class)} details"
                      onclick={(e) => (e.stopPropagation(), disclose(c.class))}
                    >
                      <span class="chev" class:open aria-hidden="true"><ChevronDown size={14} /></span>
                    </button>
                  {/if}
                </div>
              </td>
              <td class="r">{metric(c.ap50)}</td>
              <td class="r muted">{metric(c.ap50_95)}</td>
              <td class="r hide-sm muted">{metric(c.precision)}</td>
              <td class="r hide-sm muted">{metric(c.recall)}</td>
              <td class="r muted">{num(c.instances)}</td>
              {#if !narrow}<td class="rest" aria-hidden="true"></td>{/if}
            </tr>
            {#if open}
              <tr class="detail" class:on id="pr-detail-{c.class}">
                <td colspan={COLS_NARROW}>
                  <dl>
                    {#each classMetrics(c) as d (d.label)}
                      <div><dt>{d.label}</dt><dd class="num">{d.value}</dd></div>
                    {/each}
                  </dl>
                  <p class="ctx">{prContext(ev.bestF1Conf)}</p>
                  {#if onviewcurve}
                    <button type="button" class="view" onclick={() => viewCurve(c.class)}>View curve<ArrowRight size={14} aria-hidden="true" /></button>
                  {/if}
                </td>
              </tr>
            {/if}
          {:else}
            <tr><td colspan={narrow ? COLS_NARROW : 7} class="none">No class matches “{query.trim()}”</td></tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if !query && listed.length > ROWS_MAX}
      <button type="button" class="more" aria-expanded={all} onclick={() => (all = !all)}>
        {all ? `Show the first ${ROWS_MAX}` : `Show all ${listed.length}`}
      </button>
    {/if}
    <p class="foot">
      Evaluated on {num(ev.images)} validation images at IoU 0.50. Precision and recall {prContext(ev.bestF1Conf, false)}; instances are
      ground-truth boxes per class.
    </p>
  </div>
</div>

<style>
  .pr {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  /* The two halves lay out as direct children of .pr (the foot keeps pinning to the bottom); `view` hides one. */
  .part {
    display: contents;
  }
  .part.off {
    display: none;
  }
  .swatch-band {
    width: 14px;
    height: 8px;
    border-radius: 2px;
    background: var(--text-tertiary);
    opacity: 0.35;
  }
  .mode {
    margin-left: auto;
  }
  .what {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 13em;
    color: var(--text-secondary);
  }
  .what-sub {
    color: var(--text-tertiary);
  }
  /* The readout state: the cursor's dashed rule as its key, values in place of the AP summary. */
  .what.at {
    color: var(--text-primary);
  }
  .cur {
    width: 1px;
    height: 12px;
    background: repeating-linear-gradient(to bottom, var(--chart-cursor) 0 2px, transparent 2px 4px);
    flex-shrink: 0;
  }
  .legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    min-height: 24px;
    padding: 0 var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .item {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    white-space: nowrap;
  }
  .item .num {
    color: var(--text-primary);
  }
  .line {
    width: 12px;
    height: 2px;
    border-radius: 1px;
    flex-shrink: 0;
  }
  .line.mean {
    background: var(--text-primary);
  }
  .hint {
    color: var(--text-tertiary);
  }
  .clear {
    height: 20px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
  }
  .clear:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  /* Class curves are 1 px hairlines; the mean and the highlighted classes 2 px. */
  .plot {
    padding: var(--space-2) var(--space-4) var(--space-4);
  }
  /* Filling a stretched panel: the box takes the height left (its 8 + 16 px padding around the chart), and the chart,
     drawn at that height, sits out of flow so it never feeds back into the box it measures. */
  .plot.fill {
    flex: 1;
    position: relative;
    min-height: 324px;
  }
  .plot.fill > :global(*) {
    position: absolute;
    inset: var(--space-2) var(--space-4) var(--space-4);
  }
  .plot :global(path.line) {
    stroke-width: 1;
  }
  .plot :global(path.line.emphasis) {
    stroke-width: 2;
  }
  .tools {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-4);
    padding: 0 var(--space-4) var(--space-2);
  }
  .filter {
    width: 240px;
    max-width: 100%;
  }
  .pick {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .scroll {
    overflow-x: auto;
    border-top: 1px solid var(--line-subtle);
  }
  /* The selection rule: a neutral bg.active fill and a 2 px accent bar on the row's leading edge. Hover is a lighter
     neutral step, and adds to the fill of a selected row, so the two stay distinct. */
  tr.hot {
    background: var(--bg-hover);
  }
  tr.on:not(.detail) {
    background: var(--bg-active);
  }
  tr.on.hot {
    background: linear-gradient(var(--bg-hover), var(--bg-hover)), var(--bg-active);
  }
  tr.on td:first-child {
    box-shadow: inset 2px 0 0 var(--accent-line);
  }
  /* The heading starts where the names do, past the swatch (8 px) and its gap (8 px). */
  .cls-h {
    padding-left: calc(var(--space-4) + var(--space-4));
  }
  .rest {
    padding: 0;
  }
  .cls-cell {

    display: flex;
    align-items: center;
    min-width: 0;
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    max-width: 200px;
    text-align: left;
  }
  .disc {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    margin-left: auto;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .disc:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .back {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    margin-left: calc(var(--space-2) * -1);
  }
  .view {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 28px;
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .view:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .cls:focus-visible {
    outline-offset: 2px;
  }
  .chev {
    display: inline-flex;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .chev.open {
    transform: rotate(180deg);
  }
  /* Phones: every metric of an opened class, under its row (precision and recall have no column there). Indented to
     the class names (past the swatch), like the Class heading. One layer in both themes: a raised well under a hairline,
     never a second selection fill; a traced class's leading bar runs on through it. */
  tr.detail td {
    height: auto;
    padding: var(--space-2) var(--space-4) var(--space-3) calc(var(--space-4) + var(--space-4));
    background: var(--bg-raised);
    box-shadow: inset 0 1px 0 var(--line-subtle);
  }
  tr.detail.on td {
    box-shadow:
      inset 2px 0 0 var(--accent-line),
      inset 0 1px 0 var(--line-subtle);
  }
  .detail dl {
    display: grid;
    grid-template-columns: repeat(2, max-content);
    gap: var(--space-2) var(--space-8);
  }
  .ctx {
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .detail dt {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .detail dd {
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .none {
    text-align: center;
    color: var(--text-tertiary);
  }
  .more {
    display: block;
    width: 100%;
    height: var(--control-lg);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .more:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .more:focus-visible {
    outline-offset: -2px;
  }
  .foot {
    margin-top: auto;
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .more {
      height: 44px;
    }
    .clear {
      height: 44px;
    }
    /* Touch: every class row is a 44 px target (fine pointers keep the dense 28 px rows). */
    tbody td {
      height: 44px;
    }
    .cls {
      min-height: 44px;
    }
    .disc {
      width: 44px;
      height: 44px;
      margin-right: calc(var(--space-3) * -1);
    }
    .view {
      min-height: 44px;
      margin-top: 0;
    }
  }
  @media (max-width: 767px) {
    /* The trace toggle takes the class cell's width, so every row's chevron sits on one edge. */
    .cls {
      flex: 1;
      max-width: none;
    }
    .fixed {
      table-layout: fixed;
    }
    /* Each number column is as wide as its heading plus a 12 px gutter on both sides (16 at the table's edge). */
    .c-ap {
      width: 64px;
    }
    .c-ap95 {
      width: 80px;
    }
    .c-inst {
      width: 88px;
    }
    .legend,
    .plot,
    .tools {
      padding-inline: var(--space-3);
    }
  }
</style>
