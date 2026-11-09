<script lang="ts" module>
  import { lr as fmtLr } from '$lib/format';

  const PLAIN: Record<string, string> = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };

  /**
   * "3.42" and "× 10⁻⁴" from 3.42e-4: the value set large, its power of ten as the unit, so a tab never cuts it off.
   * `exp` ("−4") is the exponent in plain digits, for a real superscript at a legible size (Unicode superscripts render
   * at about 8 px in a 12 px unit).
   */
  export function lrParts(v: number | null | undefined): { value: string; unit?: string; exp?: string } {
    const [m, e] = fmtLr(v).split(' × ');
    return e ? { value: m!, unit: `× ${e}`, exp: [...e.slice(2)].map((c) => PLAIN[c] ?? c).join('') } : { value: m! };
  }
</script>

<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import { fade } from 'svelte/transition';
  import { motion } from '$lib/ui/motion';
  import { pageKeys } from '$lib/shell/pageCommands.svelte';
  /**
   * The run's metrics as one instrument: the epoch progress, a row of KPI tabs (each with its current value) and the one
   * chart the selected tab names. A single run is drawn in ink (primary metric in text.primary, secondary ones
   * tertiary with dash patterns); hue is reserved for entities (runs in Compare, classes).
   */
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import { BestPulse } from '$lib/charts/bestPulse';
  import { scoresTitle, type EvalSource } from '$lib/evaluated';
  import LossOptions, { type Scale, type Smoothing } from '$lib/charts/LossOptions.svelte';
  import { holdDomain, logDomain, niceDomain, unitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, type Fit, type Held } from '$lib/charts/domain';
  import { smooth, type Pt } from '$lib/charts/downsample';
  import LineChart, { type Marker, type Series } from '$lib/charts/LineChart.svelte';
  import { runHorizon } from '$lib/charts/horizon';
  import type { Epoch, Run, Step } from '$lib/api/types';
  import { progressText, runProgress, stepsPerEpoch } from '$lib/api/progress';
  import { effectiveCloseMosaic } from './resolved';
  import { conf, duration, eta, metric, num } from '$lib/format';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Select from '$lib/ui/Select.svelte';
  import { media } from '$lib/design/tokens';

  let {
    run,
    steps,
    epochs,
    version,
    loading,
    queued = false,
    folded = false,
    evalOn = null,
  }: {
    run: Run;
    steps: Step[];
    epochs: Epoch[];
    version: number;
    loading: boolean;
    /**
     * A queued run with nothing yet: the same frame as a running run (KPI tabs reading "—", the empty plot); on phones a
     * one-line "Metrics appear when training starts" that opens it.
     */
    queued?: boolean;
    /** What the epochs' scores are on (lib/evaluated `runEvalSource`): the titles name the split when it is not validation. */
    evalOn?: EvalSource | null;
    /** Phones: another section is picked, so the chart folds to its one-line readout (it opens on a tap). */
    folded?: boolean;
  } = $props();

  type Tab = 'map' | 'map50' | 'pr' | 'loss' | 'lr' | 'ips';
  let tab = $state<Tab>('map');

  // Phones (< 768 px) turn the six tabs into one metric picker in the chart header, where it replaces the title, and
  // get a shorter plot, so the samples start within the first screen. Tablets keep the tabs (scrolling sideways when
  // a value needs it), as on laptops and desktops.
  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = $derived(phoneQuery.current);
  // Phones: a queued run's plot is the axes and one line, folded until asked for. An ended run that learned almost
  // nothing (its best mAP50-95 gets a zoomed 0–1 axis, ../../lib/charts/domain UNIT_ZOOM_BELOW) gets a
  // shorter plot on every tab, so its few low samples are not a thin line along the floor of a 320 px frame.
  const flat = $derived(!['queued', 'running', 'stopping'].includes(run.status) && unitTop(run.best?.map50_95 ?? 0) < UNIT_ZOOM_BELOW);
  const height = $derived(queued && narrow ? 120 : narrow ? 168 : flat ? 200 : 320);
  let queuedOpen = $state(false);
  const queuedFold = $derived(queued && narrow && !queuedOpen);
  // Folded on phones while another section is picked; a tap on the readout opens it until the section changes.
  let peek = $state(false);
  $effect(() => {
    void folded;
    peek = false;
  });
  const fold = $derived(folded && narrow && !peek && !queued);

  const live = $derived(['queued', 'running', 'stopping'].includes(run.status));
  const progress = $derived(runProgress(run));
  /** Epochs without mosaic as the trainer runs them (capped at half the run: 7 of a 15-epoch run asked for 10). */
  const closeN = $derived(effectiveCloseMosaic(run));
  const closeFrom = $derived(1 - closeN / run.epochs);
  /** The run's last epochs train without mosaic: the bar hatches them and every chart marks where they start. */
  const mosaicOff = $derived(closeFrom > 0 && closeFrom < 1);
  const elapsed = $derived(run.started ? ((run.finished ? Date.parse(run.finished) : Date.now()) - Date.parse(run.started)) / 1000 : null);

  const lastStep = $derived.by(() => {
    void version;
    return steps[steps.length - 1];
  });
  const lastEpoch = $derived.by(() => {
    void version;
    return epochs[epochs.length - 1];
  });

  // ---- chart data ----
  let smoothing = $state<Smoothing>('0.9');
  // Log by default: the total starts near 8 while box / cls / dfl sit between 0.4 and 3, which a linear axis squashes
  // into its bottom quarter.
  let scale = $state<Scale>('log');
  let horizon = $state<'observed' | 'full'>('observed');
  const observed = $derived.by(() => {
    void version;
    const spe = Math.max(1, stepsPerEpoch(run, steps));
    return Math.max(lastStep ? lastStep.step / spe : 0, lastEpoch ? lastEpoch.epoch : 0);
  });
  const xDomain = $derived(runHorizon(observed, run.epochs, horizon));
  const epochTitle = (x: number) => `Epoch ${x.toFixed(x % 1 === 0 ? 0 : 1)}`;

  const INK = 'var(--text-primary)';
  const INK2 = 'var(--text-tertiary)';
  const ep = (f: (e: Epoch) => number): Pt[] => epochs.map((e) => ({ x: e.epoch, y: f(e) }));
  const perStep = (f: (s: Step) => number): Pt[] => {
    const spe = Math.max(1, stepsPerEpoch(run, steps));
    return steps.map((s) => ({ x: s.step / spe, y: f(s) }));
  };

  /**
   * How the focal line ends, in StatusBadge's marks (LineChart `end`): a ringed dot while the run trains, a square once
   * stopped, a cross when it failed, a plain dot when it completed. With the lime ring on the best epoch this is the
   * focal chart's vocabulary, shared with Overview's live card.
   */
  const endMark = $derived(run.status === 'running' ? ('live' as const) : run.status === 'stopped' ? ('stopped' as const) : run.status === 'failed' ? ('failed' as const) : undefined);
  const series = $derived.by((): Series[] => {
    void version;
    switch (tab) {
      case 'map':
        return [
          { id: 'm95', label: 'mAP50-95', color: INK, area: true, end: endMark, points: ep((e) => e.val.map50_95) },
          { id: 'm50', label: 'mAP50', color: INK2, pattern: 'dash', points: ep((e) => e.val.map50) },
        ];
      case 'map50':
        return [
          { id: 'm50', label: 'mAP50', color: INK, area: true, end: endMark, points: ep((e) => e.val.map50) },
          { id: 'm95', label: 'mAP50-95', color: INK2, pattern: 'dash', points: ep((e) => e.val.map50_95) },
        ];
      case 'pr':
        return [
          { id: 'p', label: 'Precision', color: INK, end: endMark, points: ep((e) => e.val.precision) },
          { id: 'r', label: 'Recall', color: INK2, pattern: 'dash', points: ep((e) => e.val.recall) },
        ];
      case 'loss': {
        const a = Number(smoothing);
        const total = perStep((s) => s.loss.total);
        return [
          { id: 'raw', label: 'total (raw)', color: INK, ghost: true, points: a > 0 ? total : [] },
          // Said in the legend, so the smoothed total never seems to contradict the KPI's raw last step.
          { id: 'total', label: a > 0 ? `Total, smoothed (EMA ${smoothing})` : 'Total, raw', endLabel: 'total', color: INK, emphasis: true, points: smooth(total, a) },
          // The components as design/README.md maps them in both apps: box amber, cls violet dashed, dfl teal dotted.
          // Box and dfl run together late in a run (0.85 vs 0.83): box is painted over dfl's dots, and each line is
          // labelled at its end, so both stay readable where they meet.
          // Full names where the legend has room; the short ones at the lines' ends and on phones (defined under ⓘ).
          { id: 'box', label: narrow ? 'box' : 'Box', endLabel: 'box', color: 'var(--chart-2)', front: true, points: smooth(perStep((s) => s.loss.box), a) },
          { id: 'cls', label: narrow ? 'cls' : 'Classification', endLabel: 'cls', color: 'var(--chart-5)', pattern: 'dash', points: smooth(perStep((s) => s.loss.cls), a) },
          { id: 'dfl', label: narrow ? 'dfl' : 'Distribution focal', endLabel: 'dfl', color: 'var(--chart-4)', pattern: 'dot', points: smooth(perStep((s) => s.loss.dfl), a) },
        ];
      }
      case 'lr': {
        const k = Math.max(1, Math.floor(steps.length / 600));
        return [{ id: 'lr', label: 'Learning rate', color: INK, points: perStep((s) => s.lr).filter((_, i) => i % k === 0) }];
      }
      case 'ips':
        return [{ id: 'ips', label: 'Images/s', color: INK, points: smooth(perStep((s) => s.imgs_per_s), 0.9) }];
    }
  });

  // LR ticks share one power of ten (stated in the header): "2" "4" with "× 10⁻³" instead of "2.00e-3".
  const lrExp = $derived.by(() => {
    void version;
    let m = 0;
    for (const s of steps) if (s.lr > m) m = s.lr;
    return m > 0 ? Math.floor(Math.log10(m)) : 0;
  });

  // ---- y domain ----
  // Fitted to the data: a failed run's mAP of 0.5 fills the plot instead of half of a 0–1 axis, and a run that learned
  // almost nothing gets a zoomed axis that says so (its caption names the full 0–1 scale). A live chart re-fits at
  // most once per epoch (./domain holdDomain), and values never tween.
  const unit = $derived(tab === 'map' || tab === 'map50' || tab === 'pr');
  const extent = $derived.by((): [number, number] | null => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series)
      for (const p of s.points) {
        if (!Number.isFinite(p.y) || (tab === 'loss' && scale === 'log' && p.y <= 0)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    return Number.isFinite(lo) ? [lo, hi] : null;
  });
  const zoomed = $derived(unit && !!extent && unitTop(extent[1]) < UNIT_ZOOM_BELOW);
  // The plot inside LineChart's margins (8 px top, 24 with the mosaic mark's label or the zoom caption; 22 px bottom).
  const plotPx = $derived(height - (mosaicOff || zoomed ? 46 : 30));
  let held: Held<Fit & { log?: boolean }> | null = null;
  const fit = $derived.by((): (Fit & { log?: boolean }) | null => {
    if (!extent) return null;
    const [lo, hi] = extent;
    let next: Fit & { log?: boolean };
    if (unit) next = unitDomain(hi, plotPx);
    else if (tab === 'loss' && scale === 'log') next = { domain: logDomain(lo, hi), ticks: [], decimals: 0, log: true };
    // Throughput starts at 0 (a rate), with a little headroom over its peak.
    else if (tab === 'ips') next = niceDomain(0, hi * 1.05);
    else next = niceDomain(lo, hi);
    const epochNow = live ? Math.floor(observed) : -1;
    held = holdDomain(held, `${tab}|${scale}|${smoothing}|${height}|${epochNow}`, next);
    return held.value;
  });

  // A new best while the run trains pulses its ring once (LineChart, over the chart duration; nothing moves under reduced motion).
  const pulser = new BestPulse();
  let pulse = $state(0);
  $effect.pre(() => {
    pulse = pulser.see(run.best?.epoch ?? null, run.status === 'running');
  });
  const best = $derived<Marker[]>(
    run.best && (tab === 'map' || tab === 'map50')
      ? [
          {
            x: run.best.epoch,
            y: tab === 'map' ? run.best.map50_95 : (epochs.find((e) => e.epoch === run.best!.epoch)?.val.map50 ?? 0),
            label: 'best',
            color: 'var(--accent-line)',
            quiet: true,
            pulse,
          },
        ]
      : [],
  );

  interface TabDef {
    id: Tab;
    label: string;
    value: string;
    /** Set one step smaller after the value; plain text (the phone picker's option reads it as is). */
    unit?: string;
    /** A power of ten's exponent ("−4"): the tab draws the unit "× 10" with a real superscript instead. */
    exp?: string;
    caption: string;
    /** Tablets (768–1023 px): a shorter label and caption, so all six tabs fit without clipping. */
    short?: string;
    capShort?: string;
    title: string;
    qualifier: string;
    /** The qualifier's power-of-ten exponent, drawn as a superscript after it. */
    qExp?: string;
  }
  // The one live-value label, as in the chart's legend: "Latest · ep 37". Per-epoch values come from the last
  // validation; per-step values from the epoch the last step is in.
  const latestEp = $derived(lastEpoch ? `Latest · ep ${lastEpoch.epoch}` : 'after the first epoch');
  const stepEp = $derived(lastStep ? `Latest · ep ${Math.max(1, Math.ceil(observed - 1e-9))}` : 'after the first step');
  // Tablets: the same readings in fewer words ("ep 37", "not yet").
  const latestShort = $derived(lastEpoch ? `ep ${lastEpoch.epoch}` : 'not yet');
  const stepShort = $derived(lastStep ? `ep ${Math.max(1, Math.ceil(observed - 1e-9))}` : 'not yet');
  /** Precision and recall are read at each epoch's best-F1 confidence (docs/API.md, Operating point): the latest one. */
  const prConf = $derived(lastEpoch?.val.best_f1_conf);
  /** The LR schedule the trainer resolved (`Run.resolved`), else what the recipe defaults to. */
  const schedule = $derived(
    run.resolved ? `warmup ${run.resolved.warmup_iters} steps · ${run.resolved.cos_lr ? 'cosine' : 'linear'}` : 'warmup 3 epochs · linear',
  );
  const tabs = $derived.by((): TabDef[] => [
    {
      id: 'map',
      label: 'mAP50-95',
      value: metric(lastEpoch?.val.map50_95),
      caption: run.best ? `best ${metric(run.best.map50_95)} · ep ${run.best.epoch}` : 'after the first epoch',
      capShort: run.best ? `best · ep ${run.best.epoch}` : 'not yet',
      title: `${scoresTitle(evalOn)} mAP50-95`,
      qualifier: run.best ? `by epoch · ○ best ${metric(run.best.map50_95)} at epoch ${run.best.epoch}` : 'by epoch',
    },
    { id: 'map50', label: 'mAP50', value: metric(lastEpoch?.val.map50), caption: latestEp, capShort: latestShort, title: `${scoresTitle(evalOn)} mAP50`, qualifier: 'by epoch' },
    {
      id: 'pr',
      label: 'Precision · Recall',
      short: 'P · R',
      value: lastEpoch ? `${metric(lastEpoch.val.precision)} · ${metric(lastEpoch.val.recall)}` : '—',
      // The latest epoch's threshold: "conf 0.31 · ep 37" (each epoch has its own, so the chart says "each at").
      caption: lastEpoch ? (prConf != null ? `conf ${conf(prConf)} · ep ${lastEpoch.epoch}` : `best-F1 conf · ep ${lastEpoch.epoch}`) : 'after the first epoch',
      capShort: lastEpoch && prConf != null ? `conf ${conf(prConf)}` : latestShort,
      title: 'Precision and recall',
      qualifier: prConf != null ? `by epoch · at each epoch's best-F1 conf (latest ${conf(prConf)})` : "by epoch · at each epoch's best-F1 confidence",
    },
    {
      id: 'loss',
      label: 'Loss',
      value: num(lastStep?.loss.total, 3),
      // The last step's raw reading, which the chart may draw smoothed: "raw" is its qualifier, set as a unit.
      unit: 'raw',
      caption: stepEp,
      capShort: stepShort,
      title: 'Training loss',
      qualifier: 'by epoch',
    },
    {
      id: 'lr',
      label: 'Learning rate',
      short: 'LR',
      // The power of ten is the value's unit, set smaller, so the tab never cuts the value off.
      ...lrParts(lastStep?.lr),
      caption: schedule,
      capShort: run.resolved ? (run.resolved.cos_lr ? 'cosine' : 'linear') : 'linear',
      title: 'Learning rate',
      qualifier: 'by epoch · × 10',
      qExp: String(lrExp).replace('-', '−'),
    },
    {
      id: 'ips',
      label: 'Throughput',
      value: num(lastStep?.imgs_per_s, 0),
      unit: 'img/s',
      caption: stepEp,
      capShort: stepShort,
      title: 'Throughput',
      qualifier: 'images/s by epoch',
    },
  ]);
  /** The legend's note: the latest value's epoch, per epoch for validation metrics, per step for the rest. */
  /** The loss terms in plain words (DESIGN_DIRECTION §2.3: no jargon without its definition). */
  const LOSS_TERMS: [term: string, means: string][] = [
    ['Box', 'how far predicted boxes are from the true ones.'],
    ['Classification (cls)', 'how wrong the class scores are.'],
    ['Distribution focal (dfl)', 'how sharply each box edge is placed.'],
    ['EMA 0.9', 'smoothing; a higher value reacts more slowly.'],
  ];
  const legendNote = $derived(unit ? (lastEpoch ? latestEp : undefined) : lastStep ? stepEp : undefined);

  // The last epochs train without mosaic (the hatched end of the progress bar): marked where they start, on every chart.
  const xMark = $derived(mosaicOff ? { x: run.epochs - closeN, label: 'mosaic off' } : undefined);
  const current = $derived(tabs.find((t) => t.id === tab)!);


  // 1–6 pick a metric tab from anywhere on the page (a single-key shortcut: off with the others in Settings).
  pageKeys((k) => {
    const t = /^[1-9]$/.test(k) ? tabs[Number(k) - 1] : undefined;
    if (!t) return false;
    tab = t.id;
    return true;
  });

  function onkeydown(e: KeyboardEvent) {
    const i = tabs.findIndex((t) => t.id === tab);
    let j = i;
    if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = tabs.length - 1;
    else return;
    e.preventDefault();
    tab = tabs[j]!.id;
    (e.currentTarget as HTMLElement).querySelector<HTMLElement>(`[data-tab="${tab}"]`)?.focus();
  }
</script>

<!-- A power of ten as "× 10" with a real 12 px superscript; other units as they are. -->
{#snippet unitOf(t: TabDef)}
  {#if t.exp}<span class="unit" aria-label={t.unit}>× 10<sup class="exp">{t.exp}</sup></span>{:else}<span class="unit">{t.unit}</span>{/if}
{/snippet}

{#snippet picker()}
  <Select
    label="Metric to chart"
    bind:value={tab}
    options={tabs.map((t) => ({ value: t.id, label: `${t.label} · ${t.value}${t.unit && t.value !== '—' ? ` ${t.unit}` : ''}` }))}
  />
{/snippet}

<section class="metrics" aria-label="Run metrics" data-tip-bounds>
  {#if queued && narrow}
    <!-- Phones, queued: one line until there is something to chart; the banner above says when it starts. -->
    <button type="button" class="pending" aria-expanded={queuedOpen} aria-controls="run-metrics-body" onclick={() => (queuedOpen = !queuedOpen)}>
      <span class="p-t">Metrics appear when training starts</span>
      <span class="r-a">{queuedOpen ? 'Hide chart' : 'Show chart'}<ChevronDown size={14} aria-hidden="true" /></span>
    </button>
  {/if}
  {#if !queuedFold}
  <div class="metrics-body" id="run-metrics-body">
  <!-- Phones: a finished run's strip only repeats the completed banner above it (status, duration), so it goes. -->
  <!-- One structure at every width: the epoch as a full-width row (count, bar, caption) over the KPI tabs. -->
  <div class="strip" class:finished={run.status === 'completed'}>
    <div class="epoch">
      <span class="sr-only">Epoch</span>
      <span class="big num">{run.epoch}<span class="of">/ {run.epochs}</span></span>
      <div class="bar">
        <div class="track" aria-hidden="true">
          <div class="fill" class:done={run.status === 'completed'} class:bad={run.status === 'failed'} style:transform="scaleX({run.status === 'completed' ? 1 : progress})"></div>
          <!-- Over the filled part of the bar (a finished run) the hatch is cut out of the fill in the panel colour. -->
          {#if mosaicOff}<div class="close" class:over={run.status === 'completed' || progress >= closeFrom} style:left="{closeFrom * 100}%"></div>{/if}
        </div>
        <!-- The hatched end explains itself on hover: a hit area over it, taller than the 4 px bar. -->
        {#if mosaicOff}<span class="close-hit" aria-hidden="true" style:left="{closeFrom * 100}%" use:tooltip={`Last ${closeN} ${closeN === 1 ? 'epoch trains' : 'epochs train'} without mosaic`}></span>{/if}
      </div>
      {#if mosaicOff}<span class="sr-only">The hatched end of the bar marks the last {closeN} {closeN === 1 ? 'epoch' : 'epochs'}, trained without mosaic.</span>{/if}
      <span class="cap num">
        {#if run.status === 'queued'}Not started{:else if live}<Meta line><span>{progressText(run)}</span><span>{run.eta_s !== null ? `${eta(run.eta_s)} left` : 'estimating…'}</span></Meta>{:else}{elapsed !== null ? `${duration(elapsed)} total` : '—'}{/if}
      </span>
    </div>
    <!-- Queued: the tabs read "—" until the first step, so the frame is the one a running run fills. -->
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div class="tabs" role="tablist" aria-label="Metric to chart" data-scrolls {onkeydown}>
        {#each tabs as t (t.id)}
          <button
            role="tab"
            class="tab"
            class:on={tab === t.id}
            aria-selected={tab === t.id}
            aria-controls="run-chart"
            tabindex={tab === t.id ? 0 : -1}
            data-tab={t.id}
            onclick={() => (tab = t.id)}
          >
            <span class="label"><span class="full">{t.label}</span><span class="short" aria-hidden="true">{t.short ?? t.label}</span></span>
            <span class="val num">{t.value}{#if t.unit && t.value !== '—'}{@render unitOf(t)}{/if}</span>
            <span class="cap num"><span class="full">{t.caption}</span><span class="short" aria-hidden="true">{t.capShort ?? t.caption}</span></span>
          </button>
        {/each}
      </div>
  </div>

  <div class="chart" class:folded={fold} id="run-chart" role="tabpanel" aria-label={current.title}>
    {#if folded && narrow && !queued}
      <!-- Phones, another section picked: the chart as its one-line readout, which opens it in place. -->
      <button type="button" class="readout" aria-expanded={!fold} aria-controls="run-chart-body" onclick={() => (peek = !peek)}>
        <span class="r-l">{current.label}</span>
        <span class="r-v num">{current.value}{#if current.unit && current.value !== '—'}{@render unitOf(current)}{/if}</span>
        <span class="r-c">{current.caption}</span>
        <span class="r-a">{fold ? 'Show chart' : 'Hide chart'}<ChevronDown size={14} aria-hidden="true" /></span>
      </button>
    {/if}
    {#if !fold}
      <div class="chart-body" id="run-chart-body">
        <header>
          <!-- Phones: the picker names the metric in place of the title (the tabs above are hidden). -->
          {#if narrow}<div class="pick">{@render picker()}</div>{/if}
          <h2><span class="t">{current.title}</span><span class="q">{current.qualifier}{#if current.qExp}<sup class="exp">{current.qExp}</sup>{/if}</span></h2>
          <div class="ctl">
            <!-- One control for the loss view (smoothing and y scale behind one button), so the header keeps at most one
                 segmented group. -->
            {#if tab === 'loss'}
              <!-- What the terms mean, for anyone who has not trained a detector before: a tap, a click or Enter opens
                   it (ui/InfoButton). -->
              <InfoButton label="What the loss terms mean">
                <span class="terms-list">
                  {#each LOSS_TERMS as [term, means] (term)}<span><b>{term}</b>: {means}</span>{/each}
                </span>
              </InfoButton>
              <LossOptions bind:smoothing bind:scale />
            {/if}
            {#if live && run.status !== 'queued'}
              <Segmented
                size="sm"
                label="Chart range"
                bind:value={horizon}
                options={[
                  { value: 'observed', label: 'So far' },
                  { value: 'full', label: `All ${run.epochs}` },
                ]}
              />
            {/if}
          </div>
        </header>
        {#if loading}
          <ChartSkeleton {height} />
        {:else}
          <!-- A tab swap fades the new chart in over 160 ms (0 under reduced motion, ui/motion). -->
          {#key tab}
            <div class="swap" in:fade={{ duration: motion(160) }}>
            <LineChart
              {series}
              {height}
              {xDomain}
              xInteger
              yDomain={fit?.domain}
              yTickValues={fit && !fit.log ? fit.ticks : undefined}
              yDecimals={fit && !fit.log ? fit.decimals : undefined}
              yLog={tab === 'loss' && scale === 'log'}
              yFormat={tab === 'lr' ? (v) => String(+(v / 10 ** lrExp).toFixed(2)) : undefined}
              valueFormat={tab === 'lr' ? (v) => (v / 10 ** lrExp).toFixed(2) : tab === 'ips' ? (v) => `${num(v, 0)} img/s` : (v) => metric(v)}
              xTitle={epochTitle}
              curve={unit ? 'monotone' : 'linear'}
              markers={best}
              endDots={unit}
              {xMark}
              yLabel={zoomed ? UNIT_ZOOM_NOTE : undefined}
              {legendNote}
              label="{current.title} by epoch"
              endLabels={tab === 'loss'}
              empty={queued ? 'Curves appear here once training starts' : unit ? 'Waiting for the first validation…' : 'Waiting for the first step…'}
            />
            </div>
          {/key}
        {/if}
      </div>
    {/if}
  </div>
  </div>
  {/if}
</section>

<style>
  .metrics {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  .strip {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    border-bottom: 1px solid var(--line-subtle);
  }
  /* The epoch: one full-width row over the tabs at every width, "38 / 100", the bar, and its caption. */
  .epoch {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    column-gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  /* Each tab is at least as wide as its value and caption, so no value is ever cut off; spare width is shared. Every
     tier is sized to fit all six; a sideways scroll is only the fallback for values far wider than real ones. */
  .tabs {
    display: grid;
    grid-template-columns: repeat(6, minmax(max-content, 1fr));
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tab {
    position: relative;
    display: grid;
    grid-template-rows: 16px 40px 16px;
    gap: 6px;
    align-content: start;
    padding: var(--space-4);
    text-align: left;
    min-width: 0;
    border-left: 1px solid var(--line-subtle);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .tab:first-child {
    border-left: 0;
  }
  .tab:hover {
    background: var(--bg-hover);
  }
  /* The selected metric: an ink bar on the edge it shares with its chart (inside the tab, which the row may clip). */
  .tab::after {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 2px;
    background: var(--text-primary);
    transform: scaleX(0);
    transition: transform var(--dur-base) var(--ease-out);
  }
  .tab.on::after {
    transform: scaleX(1);
  }
  .tab:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  .label {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tab.on .label {
    color: var(--text-primary);
  }
  .big {
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .of {
    margin-left: 6px;
    font-size: var(--type-lg-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  /* A fixed line box: the value's baseline sits at the same height in every tab, with or without a unit, an exponent or
     a fallback glyph (the unit's own line-height is 1, so a 12 px span can never deepen the line). */
  .val {
    align-self: end;
    height: 32px;
    overflow: visible;
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .tab.on .val {
    color: var(--text-primary);
  }
  .unit {
    line-height: 1;
    margin-left: 4px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  /* A power of ten's exponent at the unit's own 12 px (the text floor), raised without opening up the line. */
  .exp {
    position: relative;
    top: -0.45em;
    margin-left: 1px;
    font-size: var(--type-sm-size);
    line-height: 0;
    vertical-align: baseline;
  }
  .cap {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track {
    position: relative;
    height: 4px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--accent-line);
    border-radius: 2px;
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  .fill.done {
    background: var(--text-secondary);
  }
  .fill.bad {
    background: var(--danger-solid);
  }
  .bar {
    position: relative;
    align-self: center;
    min-width: 0;
  }
  .close-hit {
    position: absolute;
    top: -12px;
    bottom: -12px;
    right: 0;
  }
  .close {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--line-strong) 3px 4px);
    border-left: 1px solid var(--line-strong);
  }
  .close.over {
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--bg-panel) 3px 4.5px);
    border-left-color: var(--bg-panel);
  }
  .chart {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
    min-width: 0;
  }
  .chart-body {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  /* Phones, queued: the frame as one 48 px line that opens it. */
  .pending {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    width: 100%;
    min-height: 48px;
    padding: var(--space-1) var(--space-3);
    text-align: left;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    border-radius: var(--radius-lg);
  }
  .pending[aria-expanded='true'] {
    border-bottom: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  }
  .pending:focus-visible {
    outline-offset: -2px;
  }
  .p-t {
    color: var(--text-secondary);
  }
  .pending:hover .r-a {
    color: var(--text-primary);
  }
  .pending[aria-expanded='true'] .r-a :global(svg) {
    transform: rotate(180deg);
  }
  /* Phones, another section picked: the chart folded to one 44 px readout row that opens it. */
  .chart.folded {
    padding-bottom: var(--space-1);
  }
  .readout {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: 44px;
    margin-top: var(--space-1);
    text-align: left;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    border-radius: var(--radius-md);
  }
  .readout:focus-visible {
    outline-offset: -2px;
  }
  .r-l {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .r-v {
    color: var(--text-primary);
    font-size: var(--type-base-size);
    font-weight: var(--weight-semibold);
    white-space: nowrap;
  }
  .r-c {
    flex: 1;
    min-width: 0;
    color: var(--text-tertiary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .r-a {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .readout:hover .r-a {
    color: var(--text-primary);
  }
  .r-a :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .readout[aria-expanded='true'] .r-a :global(svg) {
    transform: rotate(180deg);
  }
  .chart header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    min-height: 48px;
    flex-wrap: wrap;
    padding-top: var(--space-2);
  }
  h2 {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    min-width: 0;
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .q {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .terms-list {
    display: grid;
    gap: var(--space-1);
  }
  .ctl {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  /* Laptops: the same six tabs in one row, a step more compact (12 px sides, 20 px values), so all six fit from
     1024 px up beside the collapsed sidebar. */
  @media (max-width: 1279px) {
    .tab {
      grid-template-rows: 16px 32px 16px;
      gap: var(--space-1);
      padding: var(--space-3);
    }
    .val {
      font-size: var(--type-xl-size);
      line-height: 32px;
      letter-spacing: var(--type-xl-tracking);
    }
  }
  /* Tablets: six tabs share about 700 px, so each says it in fewer words ("P · R", "ep 37"), the full words their
     accessible name; nothing clips. */
  .short {
    display: none;
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .tab .full {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    .tab .short {
      display: inline;
    }
  }
  @media (max-width: 767px) {
    /* Phones: six tiles would push the chart down: one compact metric picker in the chart header instead, in place of
       the title (the qualifier stays beside it). */
    .tabs {
      display: none;
    }
    .epoch {
      border-bottom: 0;
    }
    .chart header {
      padding-top: var(--space-3);
      row-gap: var(--space-2);
    }
    h2 .t {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    h2 {
      font-size: var(--type-sm-size);
      font-weight: var(--weight-regular);
    }
    /* "23 / 100" and its meta on one baseline, the bar under the whole row 8 px below; the chart follows. */
    .epoch {
      grid-template-columns: auto minmax(0, 1fr);
      align-items: baseline;
      row-gap: var(--space-2);
    }
    .big {
      grid-row: 1;
      grid-column: 1;
      align-self: baseline;
    }
    .epoch .cap {
      grid-row: 1;
      grid-column: 2;
      align-self: baseline;
      text-align: right;
    }
    .epoch .bar {
      grid-row: 2;
      grid-column: 1 / -1;
      align-self: center;
    }
    .val {
      height: 28px;
      font-size: var(--type-lg-size);
      line-height: 28px;
    }
    .strip.finished {
      display: none;
    }
    .chart {
      padding: 0 var(--space-3) var(--space-3);
    }
    /* One row: the metric picker and the range control; the qualifier (best value and epoch) on a line below. */
    .chart header {
      min-height: 0;
    }
    .pick {
      flex: 1 1 160px;
      min-width: 0;
      max-width: 240px;
    }
    .ctl {
      margin-left: auto;
    }
    h2 {
      order: 1;
      flex-basis: 100%;
    }
    .q {
      white-space: normal;
    }
  }
  /* The fading wrapper of a swapped chart: it takes the chart's place in the layout. */
  .swap {
    min-width: 0;
  }
</style>
