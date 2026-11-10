<script lang="ts" module>
  import { FROM_SCRATCH, origin } from '$lib/lineage';

  /**
   * The weights a run started from, by name: a model id resolves through the models list, a path shows its file name,
   * none is null ("From scratch"). Fine-tuned vs from scratch often explains most of a gap between two runs.
   */
  export function startFrom(weights: string | null, models: readonly { id: string; name: string }[] | undefined): string | null {
    const o = origin(weights, models ?? []);
    return o.kind === 'scratch' ? null : o.label;
  }
</script>

<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import { fade } from 'svelte/transition';
  import { motion } from '$lib/ui/motion';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import { untrack } from 'svelte';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Epoch, Run, RunMetrics } from '$lib/api/types';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { fitFloor, fitFloorNote, niceDomain, unitDomain, unitFitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, withLabelRow } from '$lib/charts/domain';
  import { nearest } from '$lib/charts/downsample';
  import { axisReason, AXIS_REASON, runEnd, runX } from '$lib/charts/runAxis';
  import { metric } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import InfoRegion from '$lib/ui/InfoRegion.svelte';
  import { provideInfoSlot } from '$lib/ui/infoSlot.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import { media } from '$lib/design/tokens';
  import AddRun from './compare/AddRun.svelte';
  import RunChips from './compare/RunChips.svelte';
  import ManageSheet from './compare/ManageSheet.svelte';
  import CompareTable from './compare/CompareTable.svelte';

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'Compare' });

  const MAX = 4;
  // Each run once: a link naming one twice (`?ids=a,a`) would key two rows alike, which Svelte refuses.
  const ids = $derived([...new Set((router.route.query.get('ids') ?? '').split(',').filter(Boolean))].slice(0, MAX));
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  // Names for runs fine-tuned from a model ("Start from").
  const models = new Resource((s) => hub.api.models(s));
  const metrics = new Resource(async (s) => {
    const list = ids;
    // A run deleted since the link was made has no metrics (and no chip): it leaves the others' alone.
    const gone = (e: unknown) => {
      if (e instanceof ApiError && e.code === 'run_not_found') return null;
      throw e;
    };
    const out = await Promise.all(list.map((id) => hub.api.runMetrics(id, s).then((m) => [id, m] as const, gone)));
    return new Map<string, RunMetrics>(out.filter((x) => x !== null));
  });

  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = $derived(phoneQuery.current);
  let cursor = $state<number | null>(null);

  const chosen = $derived(ids.map((id) => runs.data?.find((r) => r.id === id)).filter((r): r is Run => !!r));
  // Runs planned for very different lengths (30 vs 150 epochs), or one run that trained over twice as long as every
  // other (150 epochs against 38, 61 and 23), would crush the short ones into the start of an epoch axis: then the axis
  // defaults to "By progress" (each run over its planned epochs, ../lib/charts/runAxis), and the header says why. Every
  // run is divided by its plan, so a run that failed or stopped early ends where it ended. Chosen over fitting the axis
  // to the shorter runs: no run's line is cut off, and the same toggle switches back to "By epoch".
  let xChoice = $state<Axis | null>(null);
  const reason = $derived(axisReason(chosen));
  const x = $derived<Axis>(xChoice ?? (reason ? 'progress' : 'epoch'));
  const xAuto = $derived(xChoice === null && x === 'progress');
  const X_OPTIONS: { value: Axis; label: string }[] = [
    { value: 'epoch', label: 'By epoch' },
    { value: 'progress', label: 'By progress' },
  ];
  /** Phones: the x axis as a compact select ("X axis": Epoch, Progress), so the metric's select keeps its whole name. */
  const X_PHONE: { value: Axis; label: string }[] = [
    { value: 'epoch', label: 'Epoch' },
    { value: 'progress', label: 'Progress' },
  ];
  // Phones: the axis ⓘ opens in place under the control row (ui/infoSlot).
  const axisInfo = provideInfoSlot('cmp-axis-info');
  /** Phones: the progress axis explained in the ⓘ beside its control, with why it was chosen when it was. */
  // Two lines at most (ui/InfoButton HELP_MAX); how the axis works is its fold.
  const progressWhy = $derived(`Each run over % of its plan${xAuto && reason ? `; chosen because ${AXIS_REASON[reason]}` : ''}.`);
  const PROGRESS_HOW = 'Each run is drawn over its own planned epochs, so a short plan and a long one compare by the shape of their curves.';
  // The x axis is named in the chart's header, as on every chart (DESIGN_DIRECTION §2.4), never under the plot: the
  // normalised axis names its unit; chosen for you, it also says why.
  const xQual = $derived(x === 'progress' ? `by progress, % of planned epochs${xAuto && reason ? ` · ${AXIS_REASON[reason]}` : ''}` : 'by epoch');
  let hint = $state<string | null>(null);
  // Colours follow the run, not its position: removing one run never repaints the others.
  let slots = $state<Record<string, number>>({});
  $effect(() => {
    const cur = ids;
    untrack(() => {
      const next: Record<string, number> = {};
      for (const id of cur) if (slots[id] !== undefined) next[id] = slots[id];
      for (const id of cur) {
        if (next[id] !== undefined) continue;
        let k = 0;
        while (Object.values(next).includes(k)) k++;
        next[id] = k;
      }
      slots = next;
    });
  });
  type Axis = 'epoch' | 'progress';
  // Run slots: sky and amber first (the most distinct pair, for the common two-run comparison), then magenta and teal.
  // No four hues stay apart for every colour-vision type, so slots 3 and 4 also get a line pattern.
  const RUN_SLOTS = [1, 2, 3, 4];
  const SLOT_PATTERN = [undefined, undefined, 'dash', 'dot'] as const;
  const slotOf = (id: string) => slots[id] ?? Math.max(0, ids.indexOf(id));
  const color = (id: string) => `var(--chart-${RUN_SLOTS[slotOf(id)] ?? 1})`;
  // Keys (chips, table header) are fills: the exact palette hue with the swatch edge, never the light theme's darker
  // line ink, so a light amber run reads amber, not brown (design/README.md, chart.fill).
  const fill = (id: string) => `var(--chart-fill-${RUN_SLOTS[slotOf(id)] ?? 1})`;

  function setIds(next: string[]) {
    router.go(`/runs/compare${next.length ? `?ids=${next.join(',')}` : ''}`, { replace: true });
  }

  const curve = (f: (e: Epoch) => number) =>
    chosen.map((r, i): Series => {
      const eps = metrics.data?.get(r.id)?.epochs ?? [];
      return {
        id: r.id,
        label: r.name,
        color: color(r.id),
        pattern: SLOT_PATTERN[slotOf(r.id)],
        // Each line ends in its run's status mark (live, stopped, failed), so a line that stops short says why.
        end: runEnd(r.status) ?? undefined,
        points: eps.map((e) => ({ x: runX(e.epoch, r, x), y: f(e) })),
      };
    });

  type MetricId = 'm95' | 'm50' | 'loss' | 'rec';
  /** The one caveat that changes how to read the plot: one wording on every screen. */
  const DATA_CAVEAT = 'Different datasets: compare the shape of the curves, not their values.';
  const METRICS: { id: MetricId; label: string; f: (e: Epoch) => number; unit?: boolean }[] = [
    { id: 'm95', label: 'mAP50-95', f: (e) => e.val.map50_95, unit: true },
    { id: 'm50', label: 'mAP50', f: (e) => e.val.map50, unit: true },
    { id: 'loss', label: 'Train loss', f: (e) => e.train_loss.total },
    { id: 'rec', label: 'Recall', f: (e) => e.val.recall, unit: true },
  ];
  // One chart, one metric at a time: four small charts repeated the same legend four times.
  let metricId = $state<MetricId>('m95');
  const metricDef = $derived(METRICS.find((m) => m.id === metricId)!);
  const series = $derived(curve(metricDef.f));
  const extent = $derived.by((): [number, number] | null => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series)
      for (const p of s.points) {
        if (!Number.isFinite(p.y)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    return Number.isFinite(lo) ? [lo, hi] : null;
  });
  // Runs that all scored low get a zoomed axis whose caption names the full 0–1 scale (./domain UNIT_ZOOM_BELOW).
  const zoomed = $derived(!!metricDef.unit && !!extent && unitTop(extent[1]) < UNIT_ZOOM_BELOW);
  // The plot itself is at least 360 px tall from 768 px up, so four curves that climb together still separate; phones
  // a 210 px plot. LineChart's margins add 8 px on top (24 under a y caption) and 22 px below.
  const plotPx = $derived(narrow ? 210 : 360);
  // The y axis: "Fit" (the default) starts under the values the runs spend most of their length in, so the differences
  // people came to compare fill the plot (round 26: 0–0.6 with the runs between 0.45 and 0.61, in its top quarter);
  // "From 0" shows the whole climb. The top is fitted either way (./domain). On a phone, by progress, the default is
  // From 0: its 210 px plot has no end labels, and a fitted floor cut the first ~18 % of every curve, the four lines
  // entering from the bottom edge within a few px of each other with no cue (round 27), where a mark per entry would
  // only pile up. A choice made by hand holds at every width.
  type YAxis = 'fit' | 'zero';
  let yChoice = $state<YAxis | null>(null);
  const yAxis = $derived<YAxis>(yChoice ?? (narrow && x === 'progress' ? 'zero' : 'fit'));
  const setY = (v: YAxis) => (yChoice = v);
  const Y_OPTIONS: { value: YAxis; label: string }[] = [
    { value: 'fit', label: 'Axis: Fit' },
    { value: 'zero', label: 'Axis: From 0' },
  ];
  const ys = $derived(series.flatMap((s) => s.points.map((p) => p.y)));
  const lastYs = $derived(series.map((s) => s.points.at(-1)?.y ?? NaN));
  // Where the lines carry direct labels (768 px up) the top leaves a label row over the highest end (`withLabelRow`;
  // `unitFitDomain`'s on the fitted span).
  const fit = $derived.by(() => {
    if (!extent) return undefined;
    if (!metricDef.unit) return niceDomain(yAxis === 'zero' ? Math.min(0, extent[0]) : extent[0], extent[1]);
    if (yAxis === 'fit') return unitFitDomain(ys, lastYs, extent[1], plotPx, !narrow);
    return unitDomain(narrow ? extent[1] : withLabelRow(extent[1], plotPx), plotPx);
  });
  /** A fitted axis's floor, said where the axis is named ("y from 0.40"), as a zoomed axis says its full scale. */
  const floorNote = $derived(fitFloorNote(fit, metricDef.unit ? 2 : 0));
  // Phones have no header qualifier: the floor goes in the y caption, after the zoom note.
  const yCaption = $derived([zoomed ? UNIT_ZOOM_NOTE : '', narrow && floorNote ? floorNote : ''].filter(Boolean).join(' · ') || undefined);
  const height = $derived(plotPx + (yCaption ? 24 : 8) + 22);
  // The chips are the legend: each shows its run's value under the cursor, or its latest value at rest.
  function valueOf(id: string): number | null {
    const pts = series.find((s) => s.id === id)?.points ?? [];
    const first = pts[0];
    const last = pts[pts.length - 1];
    if (!first || !last) return null;
    if (cursor === null) return last.y;
    if (cursor < first.x - 0.5 || cursor > last.x + 0.5) return null;
    return pts[nearest(pts, cursor)]?.y ?? null;
  }
  // Hover, focus or tap a run chip to bring its line forward: 3 px in its hue over a halo, painted last with a bold end
  // label; the others become context curves, 1 px in chart.context (>= 3:1, never faded), while their chips keep their
  // hue. Unpinning restores every hue.
  let hovered = $state<string | null>(null);
  let pinned = $state<string | null>(null);
  // Only runs still in the comparison count (a removed run must not leave every line faded).
  const highlight = $derived([hovered, pinned].find((x) => x !== null && ids.includes(x)) ?? null);
  // At rest the chips read each run's latest value: "Latest · ep 37" when the runs share that epoch, else per run.
  const lastEps = $derived(new Set(chosen.map((r) => metrics.data?.get(r.id)?.epochs.at(-1)?.epoch ?? null)));
  const latestNote = $derived.by(() => {
    const [only] = lastEps;
    return lastEps.size === 1 && only !== null && only !== undefined ? `Latest · ep ${only}` : "Latest · each run's last epoch";
  });
  // Recall is read at each epoch's own best-F1 confidence (docs/API.md, Operating point): the readout says so.
  const readoutAt = $derived(
    (cursor === null ? latestNote : x === 'epoch' ? `Epoch ${Math.round(cursor)}` : `${Math.round(cursor)}% through`) +
      (metricId === 'rec' ? ' · at best-F1 conf' : ''),
  );
  const xDomain = $derived<[number, number]>(x === 'epoch' ? [0, Math.max(1, ...chosen.map((r) => r.epochs))] : [0, 100]);

  // mAP is only comparable on the same data: no winner across datasets.
  const sameData = $derived(new Set(chosen.map((r) => r.config.dataset_id)).size <= 1);

  const startOf = (r: Run) => startFrom(r.config.weights, models.data);
  const setX = (v: Axis) => (xChoice = v);

  let managing = $state(false);

  const candidates = $derived((runs.data ?? []).filter((r) => !ids.includes(r.id) && r.status !== 'queued'));
</script>

{#snippet lede()}<span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>{/snippet}

<!-- Phones lead with the plot: the explanation sits in the disclosure under the legend. -->
<Page title="Compare runs" sub={narrow ? undefined : lede}>
  {#if runs.error && !runs.data}
    <!-- The page's states sit in the chart's frame at the chart's height (DESIGN_DIRECTION 8), the shape the page will
         have, never a frameless block in the corner of an empty canvas. -->
    <section class="compare c-state" aria-label="Curves" style:min-height="{height}px">
      <ErrorState error={runs.error} retry={() => runs.refresh()} compact />
    </section>
  {:else if ids.length === 0}
    <section class="compare c-state" aria-label="Curves" style:min-height="{height}px">
      <EmptyState icon={GitCompare} title="Pick runs to compare" description="Add two to four runs, or select them on the Runs page.">
        {#snippet actions()}
          <!-- One action: pick runs here when there are any, else go and make some. -->
          {#if candidates.length}{@render adder()}{:else}<Button href={href('/runs')}>Go to Runs</Button>{/if}
        {/snippet}
      </EmptyState>
    </section>

  {:else}
    <section class="compare" aria-label="Curves">
      {#if narrow}
        <!-- Phones: one control row, the runs (two columns of compact keys whose values follow a tap or scrub on the
             plot), the one caveat that changes how to read it right above the plot, then the plot. Opening, removing and
             adding runs sit in a sheet (Edit); the longer explanation sits in the disclosure. -->
        <!-- One row of three compact groups: the metric (the row's remaining width, its whole name at 16 px touch text),
             the x axis as a select of two short words, and its ⓘ in a 44 px slot of its own, so no control's target
             lies over its neighbour. -->
        <header class="c-row">
          <span class="c-metric"><Select label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} /></span>
          <!-- The x axis and its ⓘ move together: on the narrowest phones both wrap under the metric, never the ⓘ alone. -->
          <span class="c-xi">
            <span class="c-x"><Select label="X axis" bind:value={() => x, setX} options={X_PHONE} /></span>
            <!-- The normalised axis, and why it was chosen, one tap away beside its control: the ticks say "%", so the
                 line of words under the controls went (it pushed the plot down). -->
            {#if x === 'progress'}<span class="i-slot"><InfoButton chart label="About the progress axis" text={progressWhy} how={PROGRESS_HOW} /></span>{/if}
          </span>
        </header>
        <!-- The ⓘ opens here, under the controls, never as a popover over the legend and plot (ui/infoSlot). -->
        <InfoRegion slot={axisInfo} />
        {@render chips()}
        {#if !sameData && chosen.length > 1}<p class="caveat">{DATA_CAVEAT}</p>{/if}
        {@render plot()}
        <details class="about">
          <summary>
            <ChevronRight size={14} aria-hidden="true" />
            <span>How to read this</span>
          </summary>
          <div class="about-body">
            <!-- The y axis's choice: the control row has no room for it on a phone. -->
            <div class="about-axis">
              <p>{floorNote ? `The y axis starts at ${fitFloor(fit)}, under where the runs spend most of their length.` : 'The y axis starts at 0.'}</p>
              <Segmented size="sm" label="Y axis" value={yAxis} onchange={setY} options={[
                { value: 'fit', label: 'Fit' },
                { value: 'zero', label: 'From 0' },
              ]} />
            </div>
            {#if x === 'progress'}<p>By progress puts each run over its planned epochs, so a short plan and a long one compare by shape.</p>{/if}
            <p>Each line ends in its run's status: a lime dot while it trains, a square where it was stopped, a cross
              where it failed.</p>
            <p>Colours follow the run, not its rank. Tap a run to keep its line in front; Edit shows how each run
              started (fine-tuned, or from scratch). The table's baseline stays until you make another run the baseline.</p>
            <p>Tap or drag across the chart to read every run at one point; tap it again or anywhere else to let go.</p>
          </div>
        </details>
        <ManageSheet bind:managing {chosen} {startOf} {fill} {SLOT_PATTERN} {slotOf} {ids} {setIds} {MAX} {candidates} {adder} />
      {:else}
        {@render chips()}
        <!-- One line, and only when it changes how to read the plot; the axis names itself (and why it was chosen). -->
        {#if !sameData && chosen.length > 1}<p class="caveat">{DATA_CAVEAT}</p>{/if}
        <header class="c-head">
          <span class="c-what">
            <Segmented label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
            <!-- The header qualifier: what the x axis is ("by epoch", "by progress, % of planned epochs"), and where a
                 fitted y axis starts ("y from 0.40"). -->
            <span class="qual">{floorNote ? `${xQual} · ${floorNote}` : xQual}</span>
          </span>
          <span class="c-ctl">
            <Select size="sm" label="Y axis" value={yAxis} onchange={setY} options={Y_OPTIONS} />
            <Segmented size="sm" label="X axis" value={x} onchange={setX} options={X_OPTIONS} />
          </span>
        </header>
        {@render plot()}
      {/if}
    </section>

    {#if chosen.length}
      <CompareTable {chosen} {ids} {narrow} {datasets} {metrics} {sameData} {startOf} />
    {/if}
  {/if}
</Page>


{#snippet chips()}
  <RunChips
    {hint}
    {readoutAt}
    bind:managing
    {chosen}
    {valueOf}
    {startOf}
    bind:pinned
    bind:hovered
    {fill}
    {SLOT_PATTERN}
    {slotOf}
    {ids}
    {setIds}
    {narrow}
    {MAX}
    {candidates}
    {adder}
  />
{/snippet}


{#snippet plot()}
  {#if metrics.error}
    <ErrorState error={metrics.error} compact retry={() => metrics.refresh()} title="Could not load these runs' metrics" />
  {:else if !metrics.data || !runs.data}
    <ChartSkeleton {height} legend={false} />
  {:else}
    <!-- A metric swap fades the new chart in over 160 ms (0 under reduced motion); within a metric the domain eases. -->
    {#key metricId}
    <div class="swap" in:fade={{ duration: motion(160) }}>
    <LineChart
      {series}
      {height}
      {xDomain}
      xInteger={x === 'epoch'}
      yDomain={fit?.domain}
      yTickValues={fit?.ticks}
      yDecimals={fit?.decimals}
      curve="monotone"
      valueFormat={(v) => metric(v)}
      xFormat={(v) => (x === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
      xTitle={(v) => (x === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
      yLabel={yCaption}
      bind:cursor
      bind:hint
      legend={false}
      {highlight}
      recede
      endDots
      endLabels
      label="{metricDef.label} for the compared runs"
    />
    </div>
    {/key}
  {/if}
{/snippet}

{#snippet adder()}
  <AddRun {candidates} {ids} {setIds} />
{/snippet}

<style>
  .compare {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  /* Nothing to draw yet (no runs picked, or the runs could not load): the chart panel's frame at the plot's height,
     the message centred where the curves will be. */
  .compare.c-state {
    place-content: center;
    justify-items: center;
  }
  /* The one caveat that changes how to read the plot, always in view, on one line above it. */
  .caveat {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    /* Compact groups 8 px apart (controls, legend, caveat, plot), so the plot starts within ~200 px of the controls. */
    .compare {
      gap: var(--space-2);
      padding: var(--space-3);
    }
  }
  /* The axis ⓘ's region sits on the frame's own edges; the section's gap spaces it. */
  .compare {
    --info-inset: 0;
    --info-after: 0;
  }
  /* Phones: metric picker and x axis share one row above the legend. The metric takes what is left (at least 140 px,
     so "mAP50-95" shows whole at 16 px touch text); the x axis is as wide as its longest word; the ⓘ has a real 44 px
     slot, its touch target inside it rather than over the select beside it. On the narrowest phones (320 px) the x axis
     and its ⓘ wrap to a second line rather than overflow. */
  .c-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .c-metric {
    flex: 1 1 140px;
    min-width: 140px;
    display: flex;
  }
  .c-metric > :global(.select) {
    flex: 1;
  }
  .c-xi {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .c-x {
    display: flex;
  }
  .i-slot {
    flex: 0 0 44px;
    display: grid;
    place-items: center;
    height: 44px;
  }
  .about {
    border-top: 1px solid var(--line-subtle);
  }
  .about summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    list-style: none;
    cursor: pointer;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .about summary::-webkit-details-marker {
    display: none;
  }
  .about summary :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .about[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .about summary:focus-visible {
    outline-offset: -2px;
  }
  .about-body {
    display: grid;
    gap: var(--space-2);
    padding-bottom: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* From 768 px up one layout whatever the qualifier's length: the metric tabs left, the x-axis control right on the
     tabs' row; a long qualifier ("by progress, % of planned epochs · plans differ in length") wraps under the tabs,
     never pushing the control onto a row of its own. Phones have their own control row (.c-row). */
  .c-head {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: start;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  /* The chart's controls, right: the y axis's choice, then the x axis's. */
  .c-ctl {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .about-axis {
    display: grid;
    gap: var(--space-2);
    justify-items: start;
  }
  .c-what {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    min-width: 0;
  }
  .c-what .qual {
    min-height: var(--control-md);
    display: inline-flex;
    align-items: center;
  }
  /* The chart's header qualifier, as on every chart: tertiary, 12 px. */
  .qual {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* The fading wrapper of a swapped chart: it takes the chart's place in the layout. */
  .swap {
    min-width: 0;
  }
</style>
