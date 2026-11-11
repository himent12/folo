// fw port of RunCompare.svelte.
import GitCompare from '$icons/git-compare-arrows';
import ChevronRight from '$icons/chevron-right';
import { computed, css, effect, fade, Key, MediaQuery, signal, untrack } from 'fw';
import { origin } from '$lib/lineage';
import { motion } from '$lib/ui/motion';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import { ApiError, hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Epoch, Run, RunMetrics } from '$lib/api/types';
import LineChart, { type Series } from '$lib/charts/LineChart';
import { fitFloor, fitFloorNote, niceDomain, unitDomain, unitFitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, withLabelRow } from '$lib/charts/domain';
import { nearest } from '$lib/charts/downsample';
import { axisReason, AXIS_REASON, runEnd, runX } from '$lib/charts/runAxis';
import { metric } from '$lib/format';
import { href, router } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import Page from '$lib/shell/Page';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import InfoButton from '$lib/ui/InfoButton';
import InfoRegion from '$lib/ui/InfoRegion';
import { provideInfoSlot } from '$lib/ui/infoSlot';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import { media } from '$lib/design/tokens';
import AddRun from './compare/AddRun';
import RunChips from './compare/RunChips';
import ManageSheet from './compare/ManageSheet';
import CompareTable from './compare/CompareTable';

/**
 * The weights a run started from, by name: a model id resolves through the models list, a path shows its file name,
 * none is null ("From scratch"). Fine-tuned vs from scratch often explains most of a gap between two runs.
 */
export function startFrom(weights: string | null, models: readonly { id: string; name: string }[] | undefined): string | null {
  const o = origin(weights, models ?? []);
  return o.kind === 'scratch' ? null : o.label;
}

type Axis = 'epoch' | 'progress';
type MetricId = 'm95' | 'm50' | 'loss' | 'rec';
type YAxis = 'fit' | 'zero';

const MAX = 4;
const X_OPTIONS: { value: Axis; label: string }[] = [
  { value: 'epoch', label: 'By epoch' },
  { value: 'progress', label: 'By progress' },
];
/** Phones: the x axis as a compact select ("X axis": Epoch, Progress), so the metric's select keeps its whole name. */
const X_PHONE: { value: Axis; label: string }[] = [
  { value: 'epoch', label: 'Epoch' },
  { value: 'progress', label: 'Progress' },
];
const PROGRESS_HOW = 'Each run is drawn over its own planned epochs, so a short plan and a long one compare by the shape of their curves.';
// Run slots: sky and amber first (the most distinct pair, for the common two-run comparison), then magenta and teal.
// No four hues stay apart for every colour-vision type, so slots 3 and 4 also get a line pattern.
const RUN_SLOTS = [1, 2, 3, 4];
const SLOT_PATTERN = [undefined, undefined, 'dash', 'dot'] as const;
/** The one caveat that changes how to read the plot: one wording on every screen. */
const DATA_CAVEAT = 'Different datasets: compare the shape of the curves, not their values.';
const METRICS: { id: MetricId; label: string; f: (e: Epoch) => number; unit?: boolean }[] = [
  { id: 'm95', label: 'mAP50-95', f: (e) => e.val.map50_95, unit: true },
  { id: 'm50', label: 'mAP50', f: (e) => e.val.map50, unit: true },
  { id: 'loss', label: 'Train loss', f: (e) => e.train_loss.total },
  { id: 'rec', label: 'Recall', f: (e) => e.val.recall, unit: true },
];
const METRIC_OPTIONS = METRICS.map((m) => ({ value: m.id, label: m.label }));
const Y_OPTIONS: { value: YAxis; label: string }[] = [
  { value: 'fit', label: 'Axis: Fit' },
  { value: 'zero', label: 'Axis: From 0' },
];

export default function RunCompare() {
  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'Compare' });

  // Each run once: a link naming one twice (`?ids=a,a`) would key two rows alike.
  const ids = computed(() => [...new Set((router.route.query.get('ids') ?? '').split(',').filter(Boolean))].slice(0, MAX), {
    equals: (a, b) => a.length === b.length && a.every((v, i) => v === b[i]),
  });
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  // Names for runs fine-tuned from a model ("Start from").
  const models = new Resource((s) => hub.api.models(s));
  const metrics = new Resource(async (s) => {
    const list = ids.get();
    // A run deleted since the link was made has no metrics (and no chip): it leaves the others' alone.
    const gone = (e: unknown) => {
      if (e instanceof ApiError && e.code === 'run_not_found') return null;
      throw e;
    };
    const out = await Promise.all(list.map((id) => hub.api.runMetrics(id, s).then((m) => [id, m] as const, gone)));
    return new Map<string, RunMetrics>(out.filter((x) => x !== null));
  });

  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = () => phoneQuery.current;
  const cursor = signal<number | null>(null);

  const chosen = computed(() => ids.get().map((id) => runs.data?.find((r) => r.id === id)).filter((r): r is Run => !!r));
  // Runs planned for very different lengths (30 vs 150 epochs), or one run that trained over twice as long as every
  // other, would crush the short ones into the start of an epoch axis: then the axis defaults to "By progress" (each run
  // over its planned epochs, ../lib/charts/runAxis), and the header says why.
  const xChoice = signal<Axis | null>(null);
  const reason = computed(() => axisReason(chosen.get()));
  const x = computed<Axis>(() => xChoice.get() ?? (reason.get() ? 'progress' : 'epoch'));
  const xAuto = computed(() => xChoice.get() === null && x.get() === 'progress');
  // Phones: the axis ⓘ opens in place under the control row (ui/infoSlot).
  const axisInfo = provideInfoSlot('cmp-axis-info');
  /** Phones: the progress axis explained in the ⓘ beside its control, with why it was chosen when it was. */
  const progressWhy = computed(() => `Each run over % of its plan${xAuto.get() && reason.get() ? `; chosen because ${AXIS_REASON[reason.get()!]}` : ''}.`);
  // The x axis is named in the chart's header, as on every chart (DESIGN_DIRECTION §2.4), never under the plot.
  const xQual = computed(() =>
    x.get() === 'progress' ? `by progress, % of planned epochs${xAuto.get() && reason.get() ? ` · ${AXIS_REASON[reason.get()!]}` : ''}` : 'by epoch',
  );
  const hint = signal<string | null>(null);
  // Colours follow the run, not its position: removing one run never repaints the others. Each run keeps the slot it
  // had; a new run takes the lowest free one (the Svelte original synced this from an effect; a computed with memory
  // gives the same slots without a second render).
  let slotMemo: Record<string, number> = {};
  const slots = computed(() => {
    const cur = ids.get();
    const next: Record<string, number> = {};
    for (const id of cur) if (slotMemo[id] !== undefined) next[id] = slotMemo[id];
    for (const id of cur) {
      if (next[id] !== undefined) continue;
      let k = 0;
      while (Object.values(next).includes(k)) k++;
      next[id] = k;
    }
    slotMemo = next;
    return next;
  });
  const slotOf = (id: string) => slots.get()[id] ?? Math.max(0, ids.get().indexOf(id));
  const color = (id: string) => `var(--chart-${RUN_SLOTS[slotOf(id)] ?? 1})`;
  // Keys (chips, table header) are fills: the exact palette hue with the swatch edge, never the light theme's darker
  // line ink, so a light amber run reads amber, not brown (design/README.md, chart.fill).
  const fill = (id: string) => `var(--chart-fill-${RUN_SLOTS[slotOf(id)] ?? 1})`;

  function setIds(next: string[]) {
    router.go(`/runs/compare${next.length ? `?ids=${next.join(',')}` : ''}`, { replace: true });
  }

  // One chart, one metric at a time: four small charts repeated the same legend four times.
  const metricId = signal<MetricId>('m95');
  const metricDef = computed(() => METRICS.find((m) => m.id === metricId.get())!);
  const series = computed(() => {
    const f = metricDef.get().f;
    const axis = x.get();
    return chosen.get().map((r): Series => {
      const eps = metrics.data?.get(r.id)?.epochs ?? [];
      return {
        id: r.id,
        label: r.name,
        color: color(r.id),
        pattern: SLOT_PATTERN[slotOf(r.id)],
        // Each line ends in its run's status mark (live, stopped, failed), so a line that stops short says why.
        end: runEnd(r.status) ?? undefined,
        points: eps.map((e) => ({ x: runX(e.epoch, r, axis), y: f(e) })),
      };
    });
  });
  const extent = computed((): [number, number] | null => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series.get())
      for (const p of s.points) {
        if (!Number.isFinite(p.y)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    return Number.isFinite(lo) ? [lo, hi] : null;
  });
  // Runs that all scored low get a zoomed axis whose caption names the full 0–1 scale (./domain UNIT_ZOOM_BELOW).
  const zoomed = computed(() => !!metricDef.get().unit && !!extent.get() && unitTop(extent.get()![1]) < UNIT_ZOOM_BELOW);
  // The plot itself is at least 360 px tall from 768 px up, so four curves that climb together still separate; phones
  // a 210 px plot. LineChart's margins add 8 px on top (24 under a y caption) and 22 px below.
  const plotPx = computed(() => (narrow() ? 210 : 360));
  // The y axis: "Fit" (the default) starts under the values the runs spend most of their length in; "From 0" shows the
  // whole climb. On a phone, by progress, the default is From 0. A choice made by hand holds at every width.
  const yChoice = signal<YAxis | null>(null);
  const yAxis = computed<YAxis>(() => yChoice.get() ?? (narrow() && x.get() === 'progress' ? 'zero' : 'fit'));
  const setY = (v: YAxis) => yChoice.set(v);
  const ys = computed(() => series.get().flatMap((s) => s.points.map((p) => p.y)));
  const lastYs = computed(() => series.get().map((s) => s.points.at(-1)?.y ?? NaN));
  // Where the lines carry direct labels (768 px up) the top leaves a label row over the highest end (`withLabelRow`;
  // `unitFitDomain`'s on the fitted span).
  const fit = computed(() => {
    const e = extent.get();
    if (!e) return undefined;
    const def = metricDef.get();
    if (!def.unit) return niceDomain(yAxis.get() === 'zero' ? Math.min(0, e[0]) : e[0], e[1]);
    if (yAxis.get() === 'fit') return unitFitDomain(ys.get(), lastYs.get(), e[1], plotPx.get(), !narrow());
    return unitDomain(narrow() ? e[1] : withLabelRow(e[1], plotPx.get()), plotPx.get());
  });
  /** A fitted axis's floor, said where the axis is named ("y from 0.40"), as a zoomed axis says its full scale. */
  const floorNote = computed(() => fitFloorNote(fit.get(), metricDef.get().unit ? 2 : 0));
  // Phones have no header qualifier: the floor goes in the y caption, after the zoom note.
  const yCaption = computed(() => [zoomed.get() ? UNIT_ZOOM_NOTE : '', narrow() && floorNote.get() ? floorNote.get() : ''].filter(Boolean).join(' · ') || undefined);
  const height = computed(() => plotPx.get() + (yCaption.get() ? 24 : 8) + 22);
  // The chips are the legend: each shows its run's value under the cursor, or its latest value at rest.
  function valueOf(id: string): number | null {
    const pts = series.get().find((s) => s.id === id)?.points ?? [];
    const first = pts[0];
    const last = pts[pts.length - 1];
    if (!first || !last) return null;
    const c = cursor.get();
    if (c === null) return last.y;
    if (c < first.x - 0.5 || c > last.x + 0.5) return null;
    return pts[nearest(pts, c)]?.y ?? null;
  }
  // Hover, focus or tap a run chip to bring its line forward; unpinning restores every hue.
  const hovered = signal<string | null>(null);
  const pinned = signal<string | null>(null);
  // Only runs still in the comparison count (a removed run must not leave every line faded).
  const highlight = computed(() => [hovered.get(), pinned.get()].find((v) => v !== null && ids.get().includes(v)) ?? null);
  // At rest the chips read each run's latest value: "Latest · ep 37" when the runs share that epoch, else per run.
  const lastEps = computed(() => new Set(chosen.get().map((r) => metrics.data?.get(r.id)?.epochs.at(-1)?.epoch ?? null)));
  const latestNote = computed(() => {
    const [only] = lastEps.get();
    return lastEps.get().size === 1 && only !== null && only !== undefined ? `Latest · ep ${only}` : "Latest · each run's last epoch";
  });
  // Recall is read at each epoch's own best-F1 confidence (docs/API.md, Operating point): the readout says so.
  const readoutAt = computed(() => {
    const c = cursor.get();
    return (c === null ? latestNote.get() : x.get() === 'epoch' ? `Epoch ${Math.round(c)}` : `${Math.round(c)}% through`) + (metricId.get() === 'rec' ? ' · at best-F1 conf' : '');
  });
  const xDomain = computed<[number, number]>(() => (x.get() === 'epoch' ? [0, Math.max(1, ...chosen.get().map((r) => r.epochs))] : [0, 100]), {
    equals: (a, b) => a[0] === b[0] && a[1] === b[1],
  });

  // mAP is only comparable on the same data: no winner across datasets.
  const sameData = computed(() => new Set(chosen.get().map((r) => r.config.dataset_id)).size <= 1);

  const startOf = (r: Run) => startFrom(r.config.weights, models.data);
  const setX = (v: Axis) => xChoice.set(v);
  const xBind = { get: () => x.get(), set: setX };

  const showSame = signal(false);
  const managing = signal(false);
  // Phones compare two runs at a time: a baseline and one other, fully in view and paged with ‹ ›. The baseline is its
  // own choice (the first run until Swap in the table's header), never the legend's emphasis.
  const allRuns = signal(false);
  const otherAt = signal(0);
  const baselineId = signal<string | null>(null);
  // A baseline removed from the comparison is forgotten, not latent: the table falls back to the first run, and adding
  // that run again later must not quietly make it the baseline once more.
  effect(() => {
    const cur = ids.get();
    untrack(() => {
      const b = baselineId.get();
      if (b !== null && !cur.includes(b)) baselineId.set(null);
    });
  });

  const candidates = computed(() => (runs.data ?? []).filter((r) => !ids.get().includes(r.id) && r.status !== 'queued'));

  const lede = () => <span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>;

  const adder = () => <AddRun candidates={candidates.get()} ids={ids.get()} setIds={setIds} />;

  const chips = () => (
    <RunChips
      hint={hint.get()}
      readoutAt={readoutAt.get()}
      bind:managing={managing}
      chosen={chosen.get()}
      valueOf={valueOf}
      startOf={startOf}
      bind:pinned={pinned}
      bind:hovered={hovered}
      fill={fill}
      SLOT_PATTERN={SLOT_PATTERN}
      slotOf={slotOf}
      ids={ids.get()}
      setIds={setIds}
      narrow={narrow()}
      MAX={MAX}
      candidates={candidates.get()}
      adder={adder}
    />
  );

  const plot = () => (
    <>
      {metrics.error ? (
        <ErrorState error={metrics.error} compact retry={() => metrics.refresh()} title="Could not load these runs' metrics" />
      ) : !metrics.data || !runs.data ? (
        <ChartSkeleton height={height.get()} legend={false} />
      ) : (
        // A metric swap fades the new chart in over 160 ms (0 under reduced motion); within a metric the domain eases.
        <Key when={metricId.get()}>
          <div class="swap" in:fade={{ duration: motion(160) }}>
            <LineChart
              series={series.get()}
              height={height.get()}
              xDomain={xDomain.get()}
              xInteger={x.get() === 'epoch'}
              yDomain={fit.get()?.domain}
              yTickValues={fit.get()?.ticks}
              yDecimals={fit.get()?.decimals}
              curve="monotone"
              valueFormat={(v) => metric(v)}
              xFormat={(v) => (x.get() === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
              xTitle={(v) => (x.get() === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
              yLabel={yCaption.get()}
              bind:cursor={cursor}
              bind:hint={hint}
              legend={false}
              highlight={highlight.get()}
              recede
              endDots
              endLabels
              label={`${metricDef.get().label} for the compared runs`}
            />
          </div>
        </Key>
      )}
    </>
  );

  // Phones lead with the plot: the explanation sits in the disclosure under the legend.
  return (
    <Page title="Compare runs" sub={narrow() ? undefined : lede}>
      {runs.error && !runs.data ? (
        // The page's states sit in the chart's frame at the chart's height (DESIGN_DIRECTION 8), the shape the page
        // will have, never a frameless block in the corner of an empty canvas.
        <section class="compare c-state" aria-label="Curves" style:min-height={`${height.get()}px`}>
          <ErrorState error={runs.error} retry={() => runs.refresh()} compact />
        </section>
      ) : ids.get().length === 0 ? (
        <section class="compare c-state" aria-label="Curves" style:min-height={`${height.get()}px`}>
          <EmptyState
            icon={GitCompare}
            title="Pick runs to compare"
            description="Add two to four runs, or select them on the Runs page."
            // One action: pick runs here when there are any, else go and make some.
            actions={() => <>{candidates.get().length ? adder() : <Button href={href('/runs')}>Go to Runs</Button>}</>}
          />
        </section>
      ) : (
        <>
          <section class="compare" aria-label="Curves">
            {narrow() ? (
              // Phones: one control row, the runs, the one caveat that changes how to read it right above the plot,
              // then the plot. Opening, removing and adding runs sit in a sheet (Edit); the longer explanation sits in
              // the disclosure.
              <>
                <header class="c-row">
                  <span class="c-metric">
                    <Select label="Metric" bind:value={metricId} options={METRIC_OPTIONS} />
                  </span>
                  {/* The x axis and its ⓘ move together: on the narrowest phones both wrap under the metric. */}
                  <span class="c-xi">
                    <span class="c-x">
                      <Select label="X axis" bind:value={xBind} options={X_PHONE} />
                    </span>
                    {x.get() === 'progress' ? (
                      <span class="i-slot">
                        <InfoButton chart label="About the progress axis" text={progressWhy.get()} how={PROGRESS_HOW} />
                      </span>
                    ) : null}
                  </span>
                </header>
                {/* The ⓘ opens here, under the controls, never as a popover over the legend and plot (ui/infoSlot). */}
                <InfoRegion slot={axisInfo} />
                {chips()}
                {!sameData.get() && chosen.get().length > 1 ? <p class="caveat">{DATA_CAVEAT}</p> : null}
                {plot()}
                <details class="about">
                  <summary>
                    <ChevronRight size={14} aria-hidden="true" />{' '}
                    <span>How to read this</span>
                  </summary>
                  <div class="about-body">
                    {/* The y axis's choice: the control row has no room for it on a phone. */}
                    <div class="about-axis">
                      <p>{floorNote.get() ? `The y axis starts at ${fitFloor(fit.get())}, under where the runs spend most of their length.` : 'The y axis starts at 0.'}</p>
                      <Segmented
                        size="sm"
                        label="Y axis"
                        value={yAxis.get()}
                        onchange={setY}
                        options={[
                          { value: 'fit', label: 'Fit' },
                          { value: 'zero', label: 'From 0' },
                        ]}
                      />
                    </div>
                    {x.get() === 'progress' ? <p>By progress puts each run over its planned epochs, so a short plan and a long one compare by shape.</p> : null}
                    <p>
                      Each line ends in its run's status: a lime dot while it trains, a square where it was stopped, a cross
                      where it failed.
                    </p>
                    <p>
                      Colours follow the run, not its rank. Tap a run to keep its line in front; Edit shows how each run
                      started (fine-tuned, or from scratch). The table's baseline stays until you make another run the baseline.
                    </p>
                    <p>Tap or drag across the chart to read every run at one point; tap it again or anywhere else to let go.</p>
                  </div>
                </details>
                <ManageSheet
                  bind:managing={managing}
                  chosen={chosen.get()}
                  startOf={startOf}
                  fill={fill}
                  SLOT_PATTERN={SLOT_PATTERN}
                  slotOf={slotOf}
                  ids={ids.get()}
                  setIds={setIds}
                  MAX={MAX}
                  candidates={candidates.get()}
                  adder={adder}
                />
              </>
            ) : (
              <>
                {chips()}
                {/* One line, and only when it changes how to read the plot; the axis names itself (and why). */}
                {!sameData.get() && chosen.get().length > 1 ? <p class="caveat">{DATA_CAVEAT}</p> : null}
                <header class="c-head">
                  <span class="c-what">
                    <Segmented label="Metric" bind:value={metricId} options={METRIC_OPTIONS} />
                    {/* The header qualifier: what the x axis is, and where a fitted y axis starts ("y from 0.40"). */}
                    <span class="qual">{floorNote.get() ? `${xQual.get()} · ${floorNote.get()}` : xQual.get()}</span>
                  </span>
                  <span class="c-ctl">
                    <Select size="sm" label="Y axis" value={yAxis.get()} onchange={setY} options={Y_OPTIONS} />
                    <Segmented size="sm" label="X axis" value={x.get()} onchange={setX} options={X_OPTIONS} />
                  </span>
                </header>
                {plot()}
              </>
            )}
          </section>
          {chosen.get().length ? (
            <CompareTable
              chosen={chosen.get()}
              ids={ids.get()}
              narrow={narrow()}
              datasets={datasets}
              metrics={metrics}
              sameData={sameData.get()}
              startOf={startOf}
              bind:showSame={showSame}
              bind:allRuns={allRuns}
              bind:otherAt={otherAt}
              bind:baselineId={baselineId}
            />
          ) : null}
        </>
      )}
    </Page>
  );
}

css`
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
`;
