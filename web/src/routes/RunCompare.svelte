<script lang="ts" module>
  import { origin } from '$lib/lineage';

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
  import X from '@lucide/svelte/icons/x';
  import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';
  import Plus from '@lucide/svelte/icons/plus';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ArrowLeftRight from '@lucide/svelte/icons/arrow-left-right';
  import { hub } from '$lib/api/index.svelte';
  import { bestMark, bestWord } from '$lib/bestMark';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Epoch, Run, RunMetrics } from '$lib/api/types';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { niceDomain, unitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, withLabelRow } from '$lib/charts/domain';
  import { nearest } from '$lib/charts/downsample';
  import { axisReason, AXIS_REASON, runEnd, runX } from '$lib/charts/runAxis';
  import { conf, deviceName, duration, metric } from '$lib/format';
  import { batchSetting, effectiveBatchSetting, closeMosaicSetting, headSetting, lrSetting, optimizerSetting, precisionSetting, settingText, warmupSetting, type Setting } from './run/resolved';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Note from '$lib/ui/Note.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import InfoRegion from '$lib/ui/InfoRegion.svelte';
  import { provideInfoSlot } from '$lib/ui/infoSlot.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { media } from '$lib/design/tokens';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'Compare' });

  const MAX = 4;
  const ids = $derived((router.route.query.get('ids') ?? '').split(',').filter(Boolean).slice(0, MAX));
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  // Names for runs fine-tuned from a model ("Start from").
  const models = new Resource((s) => hub.api.models(s));
  const metrics = new Resource(async (s) => {
    const list = ids;
    const out = await Promise.all(list.map((id) => hub.api.runMetrics(id, s).then((m) => [id, m] as const)));
    return new Map<string, RunMetrics>(out);
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
  // Phones: the axis ⓘ opens in place under the control row (ui/infoSlot).
  const axisInfo = provideInfoSlot('cmp-axis-info');
  /** Phones: the progress axis explained in the ⓘ beside its control, with why it was chosen when it was. */
  const progressWhy = $derived(
    `By progress puts each run over its planned epochs (% of the plan), so a short plan and a long one compare by shape.${xAuto && reason ? ` Chosen for you: ${AXIS_REASON[reason]}.` : ''}`,
  );
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
  // a 210 px plot. LineChart's margins add 8 px on top (24 under the zoom caption) and 22 px below.
  const plotPx = $derived(narrow ? 210 : 360);
  const height = $derived(plotPx + (zoomed ? 24 : 8) + 22);
  // Fitted to the runs (./domain): four runs between 0.45 and 0.62 spread over 0–0.7, not bunched in a 0–1 axis. Where
  // the lines carry direct labels (768 px up) the top leaves a label row over the highest end (`withLabelRow`).
  const fit = $derived(
    extent ? (metricDef.unit ? unitDomain(narrow ? extent[1] : withLabelRow(extent[1], plotPx), plotPx) : niceDomain(extent[0], extent[1])) : undefined,
  );
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
  const dsName = (id: string) => datasets.data?.find((d) => d.id === id)?.name ?? id;
  /**
   * The best mAP50-95 of each dataset the runs share (lib/bestMark, as the Models list ranks within a group): "best", or
   * "joint best" on every run of an exact tie, and "on <dataset>" when the comparison spans datasets, since scores
   * compare only on the same data. A dataset with one run here has no best. Values that differ only past three
   * decimals are shown with as many as tell them apart.
   */
  const bestOf = $derived.by(() => {
    const marks = new Map<string, { word: string; decimals: number }>();
    for (const id of new Set(chosen.map((r) => r.config.dataset_id))) {
      const group = chosen.filter((r) => r.config.dataset_id === id);
      const m = bestMark(group.map((r) => r.best?.map50_95));
      for (const r of group) if (m.decimals > 3) marks.set(r.id, { word: '', decimals: m.decimals });
      for (const i of m.best) marks.set(group[i]!.id, { word: sameData ? bestWord(m) : `${bestWord(m)} on ${dsName(id)}`, decimals: m.decimals });
    }
    return marks;
  });
  const bestScore = (r: Run) => (r.best && bestOf.get(r.id) ? r.best.map50_95.toFixed(bestOf.get(r.id)!.decimals) : metric(r.best?.map50_95));

  const startOf = (r: Run) => startFrom(r.config.weights, models.data);
  const startLine = (r: Run) => {
    const from = startOf(r);
    return from ? `Fine-tuned from ${from}` : 'From scratch';
  };
  const setX = (v: Axis) => (xChoice = v);

  // A cell is a Setting (./run/resolved): identifiers (an architecture, a checkpoint) in mono like the chips and every
  // other screen (ui/Ident), prose ("From scratch") and named optimizers ("AdamW") in sans. Where the server resolved a
  // run's config (`Run.resolved`), the optimizer, LR and batch are what it trains with, the request a note beside them.
  const plain = (value: string): Setting => ({ value });
  const ident = (value: string): Setting => ({ value, ident: true });
  const rows = $derived<[string, (r: Run) => Setting][]>([
    ['Dataset', (r) => plain(datasets.data?.find((d) => d.id === r.config.dataset_id)?.name ?? r.config.dataset_id)],
    ['Model', (r) => ident(r.config.arch)],
    ['Start from', (r) => (startOf(r) ? ident(startOf(r) ?? '') : plain('From scratch'))],
    ['Epochs', (r) => plain(`${r.epoch}/${r.epochs}`)],
    ['Batch', (r) => batchSetting(r.config, r.resolved)],
    ['Effective batch', (r) => (r.resolved ? effectiveBatchSetting(r.resolved) : plain('—'))],
    ['Image size', (r) => plain(`${r.config.imgsz} px`)],
    ['Optimizer', (r) => optimizerSetting(r.config, r.resolved)],
    ['LR', (r) => lrSetting(r.config, r.resolved)],
    ['Warmup', (r) => (r.resolved ? warmupSetting(r.resolved) : plain('—'))],
    ['Mosaic / mixup', (r) => plain(`${r.config.augment.mosaic} / ${r.config.augment.mixup}`)],
    ['Close mosaic', (r) => closeMosaicSetting(r)],
    ['Precision', (r) => precisionSetting(r.config)],
    ['Head', (r) => headSetting(r.config)],
    ['Device', (r) => plain(deviceName(r.device))],
    ['Duration', (r) => plain(r.started ? duration(((r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000) : '—')],
  ]);
  const differs = (f: (r: Run) => Setting) => new Set(chosen.map((r) => settingText(f(r)))).size > 1;
  /** Precision and recall of the epoch Best names (the run's model), with the threshold they were read at. */
  const bestVal = (r: Run) => (r.best ? metrics.data?.get(r.id)?.epochs.find((e) => e.epoch === r.best?.epoch)?.val : undefined);
  // What differs is the comparison; identical settings fold into one row that can be opened.
  const diffRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => differs(f)) : rows);
  const sameRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => !differs(f)) : []);
  let showSame = $state(false);
  let managing = $state(false);
  let tableEl = $state<HTMLElement>();
  let tableMore = $state(false);
  /** Run columns (1-based) fully in view while the table scrolls sideways; null when every run fits. */
  let tableRange = $state<{ first: number; last: number } | null>(null);
  function measureTable() {
    const el = tableEl;
    tableMore = !!el && el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
    if (!el || el.scrollWidth <= el.clientWidth + 1) {
      tableRange = null;
      return;
    }
    // In view = at least 80 % of the column between the pinned setting column and the right edge.
    const ths = [...el.querySelectorAll<HTMLElement>('thead th')];
    const left = el.scrollLeft + (ths[0]?.offsetWidth ?? 0);
    const right = el.scrollLeft + el.clientWidth;
    let first = 0;
    let last = 0;
    ths.slice(1).forEach((th, i) => {
      if (th.offsetLeft + th.offsetWidth * 0.2 >= left - 1 && th.offsetLeft + th.offsetWidth * 0.8 <= right) {
        first ||= i + 1;
        last = i + 1;
      }
    });
    tableRange = first ? { first, last } : null;
  }
  const scrollTableEnd = () =>
    tableEl?.scrollTo({ left: tableEl.scrollWidth, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  $effect(() => {
    const el = tableEl;
    if (!el) return;
    void chosen.length;
    measureTable();
    const ro = new ResizeObserver(measureTable);
    ro.observe(el);
    return () => ro.disconnect();
  });

  /** The side-by-side table's setting column (104 px two at a time on phones), and the narrowest a run column gets before
   *  the table scrolls. Set on the header row, the one row a fixed table takes its widths from, so the columns hold the
   *  same widths from the first render whatever the pager does (DESIGN_DIRECTION §2.7). */
  const SETTING_COL = 152;
  const PAIR_SETTING_COL = 104;
  const RUN_COL_MIN = 136;
  /** A few runs across a wide panel: bounded run columns and an empty last one (`.rest`, DESIGN_DIRECTION §2.7), so two
   *  runs' values sit side by side, never 500 px apart across 1,200 px. */
  const RUN_COL_BOUND = 224;
  const laptopUp = new MediaQuery(media.laptopUp, true);

  // Phones compare two runs at a time: a baseline and one other, fully in view and paged with ‹ ›; values that differ
  // from the baseline's are marked. "All runs" opens the whole table (it scrolls). The baseline is its own choice (the
  // first run until Swap in the table's header), never the legend's emphasis: keeping a line in front to
  // read it must not change what the table measures against.
  let allRuns = $state(false);
  let otherAt = $state(0);
  let baselineId = $state<string | null>(null);
  // A baseline removed from the comparison is forgotten, not latent: the table falls back to the first run, and adding
  // that run again later must not quietly make it the baseline once more.
  $effect(() => {
    const cur = ids;
    untrack(() => {
      if (baselineId !== null && !cur.includes(baselineId)) baselineId = null;
    });
  });
  const pairMode = $derived(narrow && chosen.length > 2 && !allRuns);
  const bounded = $derived(!pairMode && laptopUp.current && chosen.length <= 3);
  const base = $derived(chosen.find((r) => r.id === baselineId) ?? chosen[0]);
  const others = $derived(chosen.filter((r) => r !== base));
  /** Makes `r` the baseline; the run it replaces takes the other column, so the pair on screen stays the same two. */
  function setBaseline(r: Run) {
    const prev = base;
    baselineId = r.id;
    otherAt = Math.max(0, chosen.filter((x) => x !== r).indexOf(prev!));
  }
  const other = $derived(others[Math.min(otherAt, others.length - 1)]);
  const shown = $derived(pairMode && base && other ? [base, other] : chosen);
  /** The phone pager's height: the pair's sticky head row holds just under it. */
  let pairH = $state(0);
  const stepOther = (d: number) => (otherAt = (Math.min(otherAt, others.length - 1) + d + others.length) % others.length);
  /** Whether a run's value differs from the baseline's: marked (a dot and the weight); an equal one is tertiary. */
  const offBase = (r: Run, f: (r: Run) => Setting) => r !== base && !!base && settingText(f(r)) !== settingText(f(base));

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
        <header class="c-row">
          <Select label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
          <Segmented size="sm" label="X axis" value={x} onchange={setX} options={X_OPTIONS} />
          <!-- The normalised axis, and why it was chosen, one tap away beside its control: the ticks say "%", so the
               line of words under the controls went (it pushed the plot down). -->
          {#if x === 'progress'}<InfoButton chart label="About the progress axis" text={progressWhy} />{/if}
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
            {#if x === 'progress'}<p>By progress puts each run over its planned epochs, so a short plan and a long one compare by shape.</p>{/if}
            <p>Each line ends in its run's status: a lime dot while it trains, a square where it was stopped, a cross
              where it failed.</p>
            <p>Colours follow the run, not its rank. Tap a run to keep its line in front; Edit shows how each run
              started (fine-tuned, or from scratch). The table's baseline stays until you make another run the baseline.</p>
            <p>Tap or drag across the chart to read every run at one point; tap it again or anywhere else to let go.</p>
          </div>
        </details>
        {@render manageSheet()}
      {:else}
        {@render chips()}
        <!-- One line, and only when it changes how to read the plot; the axis names itself (and why it was chosen). -->
        {#if !sameData && chosen.length > 1}<p class="caveat">{DATA_CAVEAT}</p>{/if}
        <header class="c-head">
          <span class="c-what">
            <Segmented label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
            <!-- The header qualifier: what the x axis is ("by epoch", "by progress, % of planned epochs"). -->
            <span class="qual">{xQual}</span>
          </span>
          <Segmented size="sm" label="X axis" value={x} onchange={setX} options={X_OPTIONS} />
        </header>
        {@render plot()}
      {/if}
    </section>

    {#if chosen.length}
      <Panel title="Side by side" meta={sameRows.length ? `${diffRows.length} settings differ` : undefined} pad="none" id="cmp-table">
        {#if pairMode && other}
          <!-- Phones: the baseline and one run beside it; the pager steps through the others. The pager and the pair's
               head row (both names, Baseline, Swap) hold under the top bar while the table is in view, so the names
               never scroll away from the settings being read. -->
          <div class="pair num" bind:offsetHeight={pairH}>
            <button type="button" class="pg" aria-label="Previous run" onclick={() => stepOther(-1)}><ChevronLeft size={16} aria-hidden="true" /></button>
            <span class="pair-what" aria-live="polite">Run {chosen.indexOf(other) + 1} of {chosen.length}<span class="faint">{' '}vs baseline</span></span>
            <button type="button" class="pg" aria-label="Next run" onclick={() => stepOther(1)}><ChevronRight size={16} aria-hidden="true" /></button>
            <button type="button" class="more" onclick={() => (allRuns = true)}>All {chosen.length} runs</button>
          </div>
        {:else if narrow && allRuns && chosen.length > 2}
          <p class="t-range num">
            {#if tableRange}{tableRange.first === tableRange.last ? `Run ${tableRange.first}` : `Runs ${tableRange.first}–${tableRange.last}`} of {chosen.length}<span
                class="sep"
                aria-hidden="true">·</span
              >{/if}<button type="button" class="more" onclick={() => (allRuns = false)}>Two at a time</button>
          </p>
        {:else if tableRange}
          <!-- What the sideways scroll hides: the runs in view, and a way to the others. -->
          <p class="t-range num">
            {tableRange.first === tableRange.last ? `Run ${tableRange.first}` : `Runs ${tableRange.first}–${tableRange.last}`} of {chosen.length}{#if tableRange.last < chosen.length}<span class="sep" aria-hidden="true">·</span><button
                type="button"
                class="more"
                onclick={scrollTableEnd}>Scroll for {chosen.length - tableRange.last} more →</button
              >{/if}
          </p>
        {/if}
        <div class="scroll" class:more={tableMore} class:pairing={pairMode && !!other} style:--pair-h="{pairH}px" bind:this={tableEl} onscroll={measureTable}>
          <!-- Fixed layout: every run column the same width, so each header's key sits over its own values. The widths
               live on the header row (no colgroup, whose keyed columns could outlive a page of runs), so the setting
               column is fixed and the run columns share the rest equally from the first render. -->
          <table class="table fixed" class:two={pairMode} style:min-width={pairMode ? undefined : `${SETTING_COL + chosen.length * RUN_COL_MIN}px`}>
            <thead>
              <tr>
                <th style:width="{pairMode ? PAIR_SETTING_COL : SETTING_COL}px"></th>
                {#each shown as r (r.id)}
                  <!-- The key in a column of its own beside the name, so a name that wraps (at its hyphens) keeps its
                       left edge and never squeezes the key. -->
                  <th style:width={bounded ? `${RUN_COL_BOUND}px` : undefined}>
                    <!-- The run's name over its role: the baseline every other column is read against, or Swap to make
                         it the baseline. No key here: the chips above are the legend, and a swatch at the column's
                         edge could not line up with its chip's. -->
                    <span class="th"
                      ><span class="th-name" class:truncate={!pairMode} title={pairMode ? undefined : r.name}>{r.name}</span
                      >{#if chosen.length > 1}{#if r === base}<span class="base-tag">Baseline</span>{:else}<button
                            type="button"
                            class="set-base"
                            aria-label="Make {r.name} the baseline"
                            title="Make this run the baseline"
                            onclick={() => setBaseline(r)}><ArrowLeftRight size={12} aria-hidden="true" />Swap</button
                          >{/if}{/if}</span
                    >
                  </th>
                {/each}
                {#if bounded}<th class="rest" aria-hidden="true"></th>{/if}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="k">Status</td>
                {#each shown as r (r.id)}<td><StatusBadge status={r.status} plain /></td>{/each}
                {@render rest()}
              </tr>
              <tr>
                <!-- The best epoch's score, which the chips' latest value need not be (a stopped run's 0.595 · stopped
                     against its best 0.606): the row says so. -->
                <td class="k">Best mAP50-95<span class="k-sub">any epoch</span></td>
                {#each shown as r (r.id)}
                  {@const win = bestOf.get(r.id)?.word}
                  <td class="num" class:winner={!!win}>{bestScore(r)}{#if win}{' '}<span class="tag" data-fact>{win}</span>{/if}</td>
                {/each}
                {@render rest()}
              </tr>
              <tr>
                <td class="k">Precision · recall</td>
                {#each shown as r (r.id)}
                  {@const v = bestVal(r)}
                  <td class="num">
                    {#if v}{metric(v.precision)} · {metric(v.recall)}<span class="pr-at"
                        ><Meta><span>{v.best_f1_conf != null ? `conf ${conf(v.best_f1_conf)}` : 'best-F1 conf'}</span><span>ep {r.best?.epoch}</span></Meta></span
                      >{:else}—{/if}
                  </td>
                {/each}
                {@render rest()}
              </tr>
              {#each diffRows as [k, f] (k)}
                <tr>
                  <td class="k">{k}</td>
                  {#each shown as r (r.id)}
                    <!-- Read against the baseline: a value that differs is marked (a dot and the weight), an equal one
                         recedes to tertiary, so "11 settings differ" shows where they differ. -->
                    {@const off = offBase(r, f)}
                    <td class="num cfg" class:diff={off} class:same={r !== base && !off}
                      class:base={r === base}
                      >{#if off}<span class="chg-row"><span class="chg" aria-hidden="true"></span><span
                            >{@render cell(f(r))}<span class="sr-only">, differs from the baseline</span></span
                          ></span
                        >{:else}{@render cell(f(r))}{/if}</td
                    >
                  {/each}
                  {@render rest()}
                </tr>
              {/each}
              {#if sameRows.length}
                <tr class="fold">
                  <td colspan={shown.length + 1 + (bounded ? 1 : 0)}>
                    <button aria-expanded={showSame} onclick={() => (showSame = !showSame)}>
                      {showSame ? 'Hide' : 'Show'} {sameRows.length} identical {sameRows.length === 1 ? 'setting' : 'settings'}
                    </button>
                  </td>
                </tr>
                {#if showSame}
                  {#each sameRows as [k, f] (k)}
                    <tr>
                      <td class="k">{k}</td>
                      {#each shown as r (r.id)}<td class="num cfg">{@render cell(f(r))}</td>{/each}
                      {@render rest()}
                    </tr>
                  {/each}
                {/if}
              {/if}
            </tbody>
          </table>
        </div>
      </Panel>
    {/if}
  {/if}
</Page>

{#snippet rest()}{#if bounded}<td class="rest" aria-hidden="true"></td>{/if}{/snippet}

{#snippet cell(v: Setting)}{#if v.ident}<Ident value={v.value} />{:else}<span class="val" class:whole={v.value.length <= 14} data-fact>{v.value}</span>{/if}{#if v.note}<Note text={v.note} fact />{/if}{/snippet}

{#snippet chips()}
  <!-- The run chips are the legend: colour key, value under the cursor, link and remove. -->
  <div class="chips" role="group" aria-label="Runs being compared">
    <div class="chips-head">
      <!-- The readout row: where the values are read, or how to read them while the plot has keyboard focus or on a
           first touch. -->
      <span class="at num" class:hint={!!hint} aria-hidden="true">{hint ?? readoutAt}</span>
      <!-- Phones: opening, removing and adding runs sit in a sheet, so the legend stays two rows of keys. -->
      <button type="button" class="manage" aria-haspopup="dialog" aria-expanded={managing} onclick={() => (managing = true)}>Edit</button>
    </div>
    {#each chosen as r (r.id)}
      {@const v = valueOf(r.id)}
      {@const from = startOf(r)}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <span
        class="chip"
        class:pinned={pinned === r.id}
        style:--c={fill(r.id)}
        onpointerenter={() => (hovered = r.id)}
        onpointerleave={() => (hovered = null)}
        onfocusin={() => (hovered = r.id)}
        onfocusout={() => (hovered = null)}
      >
        <!-- The legend entry (key and name) is the toggle: it keeps the run's line in front. -->
        <button
          type="button"
          class="key"
          aria-pressed={pinned === r.id}
          title={pinned === r.id ? 'Let the line fall back' : 'Keep this line in front'}
          onclick={() => (pinned = pinned === r.id ? null : r.id)}
        ><i aria-hidden="true" data-pattern={SLOT_PATTERN[slotOf(r.id)]}></i><span class="nm">{r.name}</span></button>
        <!-- How the line ended (live, stopped, failed) in words beside its value: the end label's fallback, and on phones,
             which have no end labels, the only word for the end marker's shape (beside the value, or under the name on phones). -->
        <span class="v num">{v === null ? '—' : metric(v)}{#if runEnd(r.status)}<span class="st"><span class="sep">{' · '}</span>{runEnd(r.status)}</span>{/if}</span>
        <!-- How the run started: fine-tuned vs from scratch often explains a gap between two curves. It lies over the
             toggle's lower half (clicks pass through to it) and runs on under the value. Phones show it in the Edit
             sheet, on a full-width line of its own. "From <checkpoint>" (the tooltip says "Fine-tuned from"): the checkpoint
             is an identifier, a fact, so it is never cut; a narrow chip breaks it after a hyphen onto a second line. -->
        <span class="from" class:tuned={!!from} title={startLine(r)} data-fact>{#if from}From <Ident value={from} class="src" />{:else}From scratch{/if}</span>
        <span class="acts">
          <a class="open" href={href(`/runs/${r.id}`)} aria-label="Open {r.name}" title="Open run"><ArrowUpRight size={12} /></a>
          <button class="rm" aria-label="Remove {r.name}" onclick={() => setIds(ids.filter((id) => id !== r.id))}><X size={12} /></button>
        </span>
      </span>
    {/each}
    {#if !narrow && ids.length < MAX && candidates.length}{@render adder()}{/if}
  </div>
{/snippet}

{#snippet manageSheet()}
  <!-- Phones: the legend's management in a bottom sheet: each run with how it started, open and remove, then Add. -->
  <Dialog bind:open={managing} title="Runs in this comparison" description="Open a run, take it out, or add another (up to {MAX}).">
    <ul class="m-list">
      {#each chosen as r (r.id)}
        {@const from = startOf(r)}
        <li class="m-row" style:--c={fill(r.id)}>
          <i aria-hidden="true" data-pattern={SLOT_PATTERN[slotOf(r.id)]}></i>
          <span class="m-what">
            <span class="m-name">{r.name}</span>
            <span class="m-from" class:tuned={!!from}>{#if from}Fine-tuned from <Ident value={from} />{:else}From scratch{/if}</span>
          </span>
          <a class="m-act" href={href(`/runs/${r.id}`)} aria-label="Open {r.name}" title="Open run"><ArrowUpRight size={16} aria-hidden="true" /></a>
          <button type="button" class="m-act" aria-label="Remove {r.name}" title="Remove from the comparison" onclick={() => setIds(ids.filter((id) => id !== r.id))}
            ><X size={16} aria-hidden="true" /></button
          >
        </li>
      {/each}
    </ul>
    {#if ids.length < MAX && candidates.length}<div class="m-add">{@render adder()}</div>{/if}
  </Dialog>
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
      yLabel={zoomed ? UNIT_ZOOM_NOTE : undefined}
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
  <!-- The face is a secondary Button's; the native picker lies over all of it, so any point of it opens the list. -->
  <label class="add">
    <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
    <span aria-hidden="true">Add run…</span>
    <select
      aria-label="Add a run to the comparison"
      onchange={(e) => {
        const v = (e.currentTarget as HTMLSelectElement).value;
        if (v) setIds([...ids, v]);
      }}
    >
      <option value="">Add run…</option>
      {#each candidates as c (c.id)}<option value={c.id}>{c.name} · {c.status}</option>{/each}
    </select>
  </label>
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
  /* The legend: one card per run (key, name, value under the cursor; how it started below), side by side. */
  .chips {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(232px, 1fr));
    gap: var(--space-2);
    align-items: stretch;
  }
  /* Laptops and tablets: a 2 × 2 grid, so four runs never leave one card orphaned on a row of its own. */
  @media (min-width: 768px) and (max-width: 1279px) {
    .chips {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  .chips-head {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .at {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .chip {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    grid-template-rows: auto auto;
    column-gap: var(--space-1);
    min-width: 0;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .chip:hover {
    border-color: var(--line-strong);
  }
  /* The run kept in front: the one selection rule (bg.active and the 2 px ink bar). */
  .chip.pinned {
    background: var(--bg-active);
  }
  .chip.pinned::before {
    content: '';
    position: absolute;
    left: -1px;
    top: -1px;
    bottom: -1px;
    width: 2px;
    border-radius: var(--radius-md) 0 0 var(--radius-md);
    background: var(--text-primary);
  }
  /* The legend entry is the toggle: key and name, over both lines of the card. */
  .key {
    grid-column: 1;
    grid-row: 1 / 3;
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-1);
    margin-left: calc(-1 * var(--space-1));
    border-radius: var(--radius-sm);
    min-width: 0;
    text-align: left;
  }
  .key:hover {
    background: var(--bg-hover);
  }
  /* On the name's centre line (20 px line). */
  .key i {
    margin-top: 8px;
  }
  .nm {
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chip .v {
    grid-column: 2;
    grid-row: 1;
    align-self: start;
    padding: var(--space-1) var(--space-1) 0 0;
    min-width: 3.2em;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
  }
  .chip .st {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  /* Indented under the name, past the key (16 px swatch + 8 px gap); clicks fall through to the toggle. */
  .from {
    grid-column: 1 / 3;
    grid-row: 2;
    align-self: end;
    position: relative;
    min-width: 0;
    padding: 0 var(--space-1) var(--space-1) var(--space-6);
    pointer-events: none;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .from.tuned {
    color: var(--text-secondary);
  }
  /* Breaks only after a hyphen (fasteners-n-/best), never mid-word and never with an ellipsis. */
  .from :global(.ident) {
    font-size: var(--type-sm-size);
    overflow-wrap: normal;
    word-break: normal;
  }
  .acts {
    grid-column: 3;
    grid-row: 1 / 3;
    display: grid;
    align-content: center;
  }
  .open,
  .chip .rm {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .open:hover,
  .chip .rm:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .chip .rm {
      position: relative;
    }
    .chip .rm::after {
      content: '';
      position: absolute;
      inset: -8px;
    }
  }
  .manage {
    display: none;
  }
  .t-range {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    min-height: 24px;
    padding: var(--space-2) var(--space-4) 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .t-range .sep {
    margin-inline: var(--space-2);
  }
  .t-range .more {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .t-range .more:hover {
    color: var(--text-primary);
  }
  .at.hint {
    color: var(--text-secondary);
  }
  /* Phones, two at a time: what is shown, the pager (44 px targets), and the way to the whole table. */
  .pair {
    display: flex;
    align-items: center;
    /* The ‹ chevron's glyph on the content edge; its 44 px target reaches into the padding. */
    padding: 0 var(--space-4) 0 calc(var(--space-4) - 14px);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .pair-what {
    text-align: center;
  }
  /* Two at a time, the pair fits the width: no sideways scroller (which would also trap the sticky head), and the
     pager and the head row stick under the top bar while the table is in view. */
  .pair {
    position: sticky;
    top: var(--size-topbar);
    z-index: var(--z-sticky);
    background: var(--bg-panel);
  }
  .scroll.pairing {
    overflow: visible;
  }
  .scroll.pairing :global(thead th) {
    position: sticky;
    top: calc(var(--size-topbar) + var(--pair-h, 0px));
    z-index: var(--z-sticky);
    background: var(--bg-panel);
    box-shadow: inset 0 -1px 0 var(--line-subtle);
  }
  .pair .more {
    margin-left: auto;
  }
  .pair .pg {
    flex-shrink: 0;
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
  }
  .pair .pg:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .pair .more,
  .t-range .more {
    min-height: 44px;
  }
  .pair .more {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .pair .more:hover {
    color: var(--text-primary);
  }
  /* Nothing truncates: labels and run names wrap (a name at its hyphens, a word only when it is longer than the
     column), and the two headers start on one line. */
  table.two th,
  table.two td {
    white-space: normal;
    overflow-wrap: break-word;
    vertical-align: top;
  }
  table.two th {
    height: auto;
    padding-block: var(--space-2);
  }
  .base-tag {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  /* A changed value: the dot hangs in the cell's padding, on the value's first line, so changed and unchanged values
     share one left edge and the dot never wraps away from its value. */
  .chg-row {
    display: block;
    position: relative;
  }
  .chg {
    position: absolute;
    left: -10px; /* grid-ok: a 6 px dot 4 px before the text */
    top: 8px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-primary);
  }
  td.same {
    color: var(--text-tertiary);
  }
  /* The baseline is the reference the other column is read against: plain body text. */
  td.cfg.base {
    color: var(--text-primary);
  }
  /* The one caveat that changes how to read the plot, always in view, on one line above it. */
  .caveat {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    /* Phones: the legend is two columns of compact keys above the plot, each one line of text that wraps only where it
       must: the swatch, the run's whole name (wrapping at its hyphens, never cut: "shapes-hard-s" beside
       "shapes-hard-m"), its value right after it, and how the line ended ("failed") under the name, so name and value
       read together and four runs take two short rows. The whole key is the toggle that keeps a line in front; its hit
       area is the card. Opening, removing and adding runs live in the Edit sheet. */
    .chips {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-1) var(--space-2);
    }
    .chips-head {
      min-height: 24px;
    }
    .manage {
      position: relative;
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding-left: var(--space-3);
      font-size: var(--type-sm-size);
      color: var(--text-secondary);
    }
    /* A 48 px target around the 24 px label, over the panel's padding above it. */
    .manage::after {
      content: '';
      position: absolute;
      inset: -12px -8px;
    }
    /* A hanging indent: a wrapped line starts under the name, past the swatch (12 px and its 6 px gap). */
    .chips .chip {
      display: block;
      min-height: 32px;
      padding: var(--space-1-5) var(--space-2) var(--space-1-5) calc(var(--space-2) + 18px);
      text-indent: -18px;
      border: 0;
      background: none;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
    }
    .chips .chip.pinned {
      background: var(--bg-active);
    }
    .chips .chip.pinned::before {
      top: 0;
      bottom: 0;
      left: 0;
    }
    .chips .chip .key {
      display: inline;
      position: static;
      margin: 0;
      padding: 0;
      font: inherit;
    }
    .chips .chip .key:hover {
      background: none;
    }
    .chips .chip .key::after {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: var(--radius-md);
    }
    .chips .chip .key:focus-visible {
      outline: none;
    }
    .chips .chip .key:focus-visible::after {
      outline: 2px solid var(--focus);
      outline-offset: -2px;
    }
    /* On the text's centre line (16 px line). */
    .chips .chip .key i {
      display: inline-block;
      width: 12px;
      margin: 0 6px 0 0;
      vertical-align: middle;
    }
    .chips .nm {
      white-space: normal;
      overflow: visible;
      overflow-wrap: break-word;
    }
    .chips .chip .v {
      display: inline;
      min-width: 0;
      margin-left: var(--space-1-5);
      padding: 0;
      text-align: left;
      white-space: normal;
      pointer-events: none;
    }
    /* How the line ended ("stopped") on a line of its own under the name: beside the value it would not fit a column. */
    .chips .chip .st {
      display: block;
      text-indent: 0;
    }
    .chips .chip .st .sep {
      display: none;
    }
    .chips .chip .from,
    .chips .chip .acts {
      display: none;
    }
    /* The side-by-side table scrolls sideways on phones: setting names stay pinned, and the right edge fades only
       while more runs lie beyond it. */
    .scroll.more {
      mask-image: linear-gradient(90deg, #000 calc(100% - 32px), transparent);
    }
    .scroll :global(th:first-child),
    .scroll :global(td.k) {
      position: sticky;
      left: 0;
      z-index: 1;
      background: var(--bg-panel);
    }
    .compare {
      padding: var(--space-3);
    }
  }
  /* The axis ⓘ's region sits on the frame's own edges; the section's gap spaces it. */
  .compare {
    --info-inset: 0;
    --info-after: 0;
  }
  /* Phones: metric picker and x axis share one row above the legend. */
  .c-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .c-row > :global(.select) {
    flex: 1;
  }
  .about {
    margin-top: calc(-1 * var(--space-2));
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
  .fold td {
    padding-top: 0;
    padding-bottom: 0;
  }
  .fold button {
    height: 32px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .fold button:hover {
    color: var(--text-primary);
  }
  /* Adding a run: a native picker (the platform list is the most accessible) dressed as the standard secondary Button
     (ui/Button): raised fill, default edge, 28 px, the icon and a medium label. */
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    align-self: center;
    justify-self: start;
    gap: var(--space-1-5);
    height: var(--control-md);
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    box-shadow: var(--shadow-sm);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard);
  }
  .add:hover {
    border-color: var(--line-strong);
    background: color-mix(in srgb, var(--bg-raised), var(--text-primary) 4%);
  }
  .add select {
    position: absolute;
    inset: 0;
    width: 100%;
    opacity: 0;
    cursor: pointer;
  }
  @media (pointer: coarse) {
    .add {
      min-height: 44px;
    }
  }
  .add:has(select:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .add select:focus-visible {
    outline: none;
  }
  .add option {
    background: var(--bg-overlay);
  }
  .scroll {
    overflow-x: auto;
  }
  table.fixed {
    table-layout: fixed;
    width: 100%;
  }
  .fixed th,
  .fixed td {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* A setting is a fact: the request beside its value moves, whole, to a line of its own rather than lose its end;
     the value itself ("360 steps") stays whole. */
  .fixed td.cfg {
    white-space: normal;
    overflow-wrap: break-word;
  }
  /* A short value ("360 steps", "0.000833") stays whole; a value in words ("Mixed precision (f16)", "End-to-end (no
     NMS)") wraps at its spaces in a narrow column, never cut. */
  .val.whole {
    white-space: nowrap;
  }
  /* A run's head: its name over its role (Baseline, or Swap), each on a line of its own. */
  .th {
    display: grid;
    justify-items: start;
    color: var(--text-primary);
    font-size: var(--type-sm-size);
  }
  .th-name {
    max-width: 100%;
  }
  .fixed thead th {
    height: auto;
    padding-block: var(--space-2);
    vertical-align: top;
  }
  /* The row's label over what it measures ("any epoch"): 12 px tertiary. */
  .k-sub {
    display: block;
    font-size: var(--type-sm-size);
  }
  /* Same patterns as the lines, so a run's key matches its curve without relying on colour. A key is a fill: the
     palette hue (--c is chart.fill) with the swatch hairline, which the pattern's mask cuts along with the dashes. */
  .chip i {
    flex-shrink: 0;
    width: 16px;
    height: 4px;
    border-radius: 2px;
    background: var(--c);
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  i[data-pattern='dash'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 5px, transparent 5px 7px);
    border-radius: 0;
  }
  i[data-pattern='dot'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 3px, transparent 3px 5px);
    border-radius: 0;
  }
  .k {
    color: var(--text-tertiary);
  }
  td.diff {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* The best run's value: ink at 600 and a tertiary "best", never a lime value or chip (DESIGN_DIRECTION §2.6: status
     hue on the mark, not the metric; the Models list marks its best rows the same way). */
  .winner {
    color: var(--text-primary);
    font-weight: var(--weight-semibold);
  }
  /* A fact: in a narrow column the word moves, whole, under its value, never cut ("best on fast…" at 820). */
  .fixed td.winner {
    white-space: normal;
  }
  /* A phone's pair columns are narrower than "best on shapes-hard": there it wraps at its spaces. */
  .two .tag {
    white-space: normal;
  }
  .tag {
    display: inline-block;
    white-space: nowrap;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
  }
  /* Where P · R were read. A setting's request beside its value is ui/Note. */
  .pr-at {
    display: block;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* The pair's other column: "Swap" makes it the baseline, the old baseline taking its place (the baseline never
     follows the legend's emphasis). One short word, so it stays on one line in the narrow column. */
  .set-base {
    justify-self: start;
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    white-space: nowrap;
    position: relative;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .set-base:hover {
    color: var(--text-primary);
  }
  /* A 44 px target around the 16 px line. */
  .set-base::after {
    content: '';
    position: absolute;
    inset: -14px -8px; /* grid-ok: a 44 px target around the 16 px line */
  }
  /* The Edit sheet: one 56 px row per run (key, name over how it started, open, remove), then Add run. */
  .m-list {
    display: grid;
    margin: 0 calc(-1 * var(--space-2));
  }
  .m-row {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) 44px 44px;
    align-items: center;
    column-gap: var(--space-2);
    min-height: 56px;
    padding: var(--space-1) var(--space-2);
  }
  .m-row + .m-row {
    border-top: 1px solid var(--line-subtle);
  }
  .m-row i {
    align-self: start;
    margin-top: 14px; /* grid-ok: on the name line, centred in the 56 px row */
  }
  .m-what {
    display: grid;
    min-width: 0;
  }
  .m-name {
    font-weight: var(--weight-medium);
    overflow-wrap: break-word;
  }
  .m-from {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    overflow-wrap: anywhere;
  }
  .m-from.tuned {
    color: var(--text-secondary);
  }
  .m-from :global(.ident) {
    font-size: var(--type-sm-size);
  }
  .m-act {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
  }
  .m-act:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .m-add {
    padding-top: var(--space-3);
  }
  .m-row i {
    flex-shrink: 0;
    width: 16px;
    height: 4px;
    border-radius: 2px;
    background: var(--c);
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  /* The fading wrapper of a swapped chart: it takes the chart's place in the layout. */
  .swap {
    min-width: 0;
  }
</style>
