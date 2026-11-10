<script lang="ts" module>
  import { lr as fmtLr } from '$lib/format';

  const CHART_KEY = 'folo.run.chartOpen';
  /**
   * Phones: whether the run's chart is open, one state for every section (Samples, Classes, Logs, Config) and every run
   * this session: the chart never opens on one section and folds on the next (round 26). Open until the person folds it;
   * a tap on its readout row folds or opens it, wherever they are.
   */
  export const chartFold = $state({ open: readChartOpen() });
  function readChartOpen(): boolean {
    try {
      return sessionStorage.getItem(CHART_KEY) !== '0';
    } catch {
      return true;
    }
  }
  export function setChartOpen(open: boolean) {
    chartFold.open = open;
    try {
      sessionStorage.setItem(CHART_KEY, open ? '1' : '0');
    } catch {
      // Storage off (a private window): the state still holds for this page.
    }
  }

  const PLAIN: Record<string, string> = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };

  /**
   * "3.42" and "× 10⁻⁴" from 3.42e-4: the value set large, its power of ten as the unit, so a tab never cuts it off.
   * `exp` ("−4") is the exponent in plain digits, for a real superscript at a legible size (Unicode superscripts render
   * at about 8 px in a 12 px unit).
   */
  /**
   * What a progress bar's hatched end means (the last `n` of `epochs` train without mosaic, the trainer's capped count),
   * said for where the run is (`epoch` done, `live`), so a phase still ahead never reads as the current state (round
   * 27's "mosaic off" at epoch 38 of 100): before it, "Mosaic off after epoch 90"; while a live run is in it, "Mosaic off
   * · final 10 epochs"; once past, "Last 10 epochs: mosaic off". `cap` is the caption under the hatch, `full` its tooltip
   * and the bar's accessible description.
   */
  export function mosaicOffText(n: number, epochs: number, epoch: number, live: boolean): { cap: string; full: string } {
    const from = epochs - n;
    const last = `${n} ${n === 1 ? 'epoch' : 'epochs'}`;
    if (epoch < from) return { cap: `Mosaic off after epoch ${from}`, full: `Mosaic off after epoch ${from}: the last ${last} train without mosaic` };
    if (live) {
      const final = n === 1 ? 'final epoch' : `final ${n} epochs`;
      return { cap: `Mosaic off · ${final}`, full: `Mosaic off now: the ${final} train without mosaic` };
    }
    return { cap: `Last ${last}: mosaic off`, full: `Last ${last}: mosaic off` };
  }

  export function lrParts(v: number | null | undefined): { value: string; unit?: string; exp?: string } {
    const [m, e] = fmtLr(v).split(' × ');
    return e ? { value: m!, unit: `× ${e}`, exp: [...e.slice(2)].map((c) => PLAIN[c] ?? c).join('') } : { value: m! };
  }

  /**
   * The epoch a per-step reading (loss, LR, throughput) comes from: "ep 24". A run that ended inside an epoch (failed or
   * stopped part-way) has steps past its last finished epoch, so their epoch says so, "ep 24 (partial)", and never seems
   * to contradict the run's 23 finished epochs. `observed` is the progress in epochs (23.4).
   */
  export function stepEpoch(status: string, observed: number, finished: number): string {
    const n = Math.max(1, Math.ceil(observed - 1e-9));
    return `ep ${n}${status !== 'running' && status !== 'queued' && n > finished ? ' (partial)' : ''}`;
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
  import InfoRegion from '$lib/ui/InfoRegion.svelte';
  import { provideInfoSlot } from '$lib/ui/infoSlot.svelte';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import { tick } from 'svelte';
  import Button from '$lib/ui/Button.svelte';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import { BestPulse, CompleteFocus } from '$lib/charts/bestPulse';
  import { scoresTitle, type EvalSource } from '$lib/evaluated';
  import LossOptions, { DEFAULT_SMOOTHING, type Scale, type Smoothing } from '$lib/charts/LossOptions.svelte';
  import { holdDomain, logDomain, niceDomain, unitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, type Fit, type Held } from '$lib/charts/domain';
  import { smooth, type Pt } from '$lib/charts/downsample';
  import LineChart, { type Marker, type Series } from '$lib/charts/LineChart.svelte';
  import { runHorizon } from '$lib/charts/horizon';
  import type { Epoch, Run, Step } from '$lib/api/types';
  import { runProgress, stepsPerEpoch } from '$lib/api/progress';
  import { effectiveCloseMosaic, warmupEpochs } from './resolved';
  import { conf, metric, num } from '$lib/format';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import { media } from '$lib/design/tokens';
  import MetricStrip, { type Tab, type TabDef } from './MetricStrip.svelte';
  // The loss terms' ⓘ opens in place on phones (ui/infoSlot), under the chart's heading.
  const lossInfo = provideInfoSlot('run-chart-info');

  let {
    run,
    steps,
    epochs,
    version,
    loading,
    queued = false,
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
  } = $props();

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
  // Phones: folded to its readout row while the person has folded it, the same in every section (chartFold).
  const fold = $derived(narrow && !queued && !chartFold.open);

  const live = $derived(['queued', 'running', 'stopping'].includes(run.status));
  const progress = $derived(runProgress(run));
  /** Epochs without mosaic as the trainer runs them (capped at half the run: 7 of a 15-epoch run asked for 10). */
  const closeN = $derived(effectiveCloseMosaic(run));
  const closeFrom = $derived(1 - closeN / run.epochs);
  /** The run's last epochs train without mosaic: the bar hatches them and every chart marks where they start. */
  const mosaicOff = $derived(closeFrom > 0 && closeFrom < 1);
  /** What the hatch means, for where the run is: its caption, and in full its tooltip and the bar's description. */
  const closeText = $derived(mosaicOffText(closeN, run.epochs, run.epoch, live));
  const uid = $props.id();
  const closeId = `${uid}-close`;
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
  let smoothing = $state<Smoothing>(DEFAULT_SMOOTHING);
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
   * How the focal line ends, in StatusBadge's marks (LineChart `end`): a filled lime dot with the heartbeat while the run
   * trains, a square once stopped, a cross when it failed, a plain dot when it completed. With the hollow ink ring on
   * the best epoch (the header's "○ best") this is the focal chart's vocabulary, shared with Overview's live card.
   */
  const endMark = $derived(run.status === 'running' ? ('live' as const) : run.status === 'stopped' ? ('stopped' as const) : run.status === 'failed' ? ('failed' as const) : undefined);
  /** The total loss as the chart draws it (smoothed at the chosen EMA), whose last value the Loss tab's caption states. */
  const smoothedTotal = $derived.by(() => {
    void version;
    return smooth(perStep((s) => s.loss.total), Number(smoothing));
  });
  const smoothedLast = $derived(smoothing !== '0' ? smoothedTotal.at(-1)?.y : undefined);
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
          // One name per series (DESIGN_DIRECTION §2.23): the legend, the end labels and the phone legend all say
          // total, box, cls and dfl, with the full names under ⓘ (LOSS_TERMS). Raw vs smoothed is said once, in the
          // Loss tab's caption ("raw · smoothed 2.075"); the smoothing itself in the loss view's control.
          { id: 'total', label: 'total', color: INK, emphasis: true, points: smoothedTotal },
          // The components as design/README.md maps them in both apps: box amber, cls violet dashed, dfl teal dotted.
          // Box and dfl run together late in a run (0.85 vs 0.83): box is painted over dfl's dots, and each line is
          // labelled at its end, so both stay readable where they meet.
          { id: 'box', label: 'box', color: 'var(--chart-2)', front: true, points: smooth(perStep((s) => s.loss.box), a) },
          { id: 'cls', label: 'cls', color: 'var(--chart-5)', pattern: 'dash', points: smooth(perStep((s) => s.loss.cls), a) },
          { id: 'dfl', label: 'dfl', color: 'var(--chart-4)', pattern: 'dot', points: smooth(perStep((s) => s.loss.dfl), a) },
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
  // The run completing while the page is open focuses its best ring once (lime viewfinder corners; none under reduced motion).
  const focuser = new CompleteFocus();
  let focus = $state(0);
  $effect.pre(() => {
    focus = focuser.see(run.status);
  });
  const best = $derived<Marker[]>(
    run.best && (tab === 'map' || tab === 'map50')
      ? [
          {
            x: run.best.epoch,
            y: tab === 'map' ? run.best.map50_95 : (epochs.find((e) => e.epoch === run.best!.epoch)?.val.map50 ?? 0),
            label: 'best',
            quiet: true,
            pulse,
            focus,
          },
        ]
      : [],
  );

  // The one live-value label, as in the chart's legend: "Latest · ep 37". Per-epoch values come from the last
  // validation; per-step values from the epoch the last step is in.
  const latestEp = $derived(lastEpoch ? `Latest · ep ${lastEpoch.epoch}` : 'after the first epoch');
  // One label for every per-step reading (the Loss, LR and Throughput tabs and the legend), "(partial)" past the end.
  const stepEpWord = $derived(stepEpoch(run.status, observed, lastEpoch?.epoch ?? 0));
  const stepEp = $derived(lastStep ? `Latest · ${stepEpWord}` : 'after the first step');
  // Tablets: the same readings in fewer words ("ep 37", "not yet").
  const latestShort = $derived(lastEpoch ? `ep ${lastEpoch.epoch}` : 'not yet');
  const stepShort = $derived(lastStep ? stepEpWord : 'not yet');
  /** Precision and recall are read at each epoch's best-F1 confidence (docs/API.md, Operating point): the latest one. */
  const prConf = $derived(lastEpoch?.val.best_f1_conf);
  const warmupWords = (r: NonNullable<Run['resolved']>) => {
    const e = warmupEpochs(r);
    return r.warmup_iters <= 0 ? 'no warmup' : `warmup ${e} ${e === 1 ? 'epoch' : 'epochs'}`;
  };
  /** The LR schedule the trainer resolved (`Run.resolved`), else what the recipe defaults to. */
  const schedule = $derived(
    // In epochs, as the chart's axis is (Config has the optimizer steps beside them): short, and never a bare "steps".
    run.resolved ? `${warmupWords(run.resolved)} · ${run.resolved.cos_lr ? 'cosine' : 'linear'}` : 'warmup 3 epochs · linear',
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
      // The last step's raw reading; the chart draws the total smoothed. The caption is the one place that says which
      // is which ("raw · smoothed 2.075", the legend's total), so the two numbers never seem to disagree.
      caption: lastStep ? (smoothedLast !== undefined ? `raw · smoothed ${num(smoothedLast, 3)}` : `raw · ${stepEpWord}`) : 'after the first step',
      capShort: lastStep ? (smoothedLast !== undefined ? `raw · smoothed ${num(smoothedLast, 3)}` : `raw · ${stepShort}`) : stepShort,
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
    ['total', 'box, cls and dfl weighted and added: what training lowers.'],
    ['box', 'box regression: how far predicted boxes are from the true ones.'],
    ['cls', 'classification: how wrong the class scores are.'],
    ['dfl', 'distribution focal: how sharply each box edge is placed.'],
    ['EMA', 'smoothing over steps (0.98: about the last 50); a higher value reacts more slowly. The Loss tab shows the last raw step and the smoothed total.'],
  ];
  const legendNote = $derived(unit ? (lastEpoch ? latestEp : undefined) : lastStep ? stepEp : undefined);

  // The last epochs train without mosaic (the hatched end of the progress bar): marked where they start, on every chart.
  const xMark = $derived(mosaicOff ? { x: run.epochs - closeN, label: 'mosaic off' } : undefined);
  const current = $derived(tabs.find((t) => t.id === tab)!);
  /**
   * Phones, the chart open: the one line between the metric picker and the legend. A value appears once above its chart
   * (DESIGN_DIRECTION §12): the legend reads the latest value (or the cursor's), so the best is stated here only where
   * it differs from it ("○ best 0.599 · ep 36"); a run whose best is its last epoch says where ("○ best · ep 150").
   */
  const phoneQualifier = $derived(
    tab === 'map' && run.best
      ? `by epoch · ○ best${run.best.epoch === lastEpoch?.epoch ? '' : ` ${metric(run.best.map50_95)}`} · ep ${run.best.epoch}`
      : current.qualifier,
  );

  // Phones: folding hands focus to the control that reopens it, and opening to the one that folds it again, so a
  // keyboard never lands on <body> when the button it pressed goes away.
  let readoutEl = $state<HTMLButtonElement>();
  let hideEl = $state<HTMLElement>();
  async function setFold(open: boolean) {
    setChartOpen(open);
    await tick();
    (open ? hideEl?.querySelector<HTMLElement>('button') : readoutEl)?.focus();
  }


  // 1–6 pick a metric tab from anywhere on the page (a single-key shortcut: off with the others in Settings).
  pageKeys((k) => {
    const t = /^[1-9]$/.test(k) ? tabs[Number(k) - 1] : undefined;
    if (!t) return false;
    tab = t.id;
    return true;
  });

</script>

<!-- A power of ten as "× 10" with a real 12 px superscript; other units as they are. -->
<!-- The loss terms in full, behind the ⓘ's short line ("How it's computed"). -->
{#snippet lossHow()}
  <span class="terms-list">
    {#each LOSS_TERMS as [term, means] (term)}<span><b>{term}</b>: {means}</span>{/each}
  </span>
{/snippet}

{#snippet unitOf(t: TabDef)}
  {#if t.exp}<span class="unit" aria-label={t.unit}>× 10<sup class="exp">{t.exp}</sup></span>{:else}<span class="unit">{t.unit}</span>{/if}
{/snippet}

{#snippet picker()}
  <!-- The metric's name only: its value is the legend's, under the picker (a value appears once above its chart). -->
  <Select label="Metric to chart" bind:value={tab} options={tabs.map((t) => ({ value: t.id, label: t.label }))} />
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
  <MetricStrip {run} {live} {progress} {mosaicOff} {closeFrom} {closeId} {closeText} {elapsed} {tabs} bind:tab {unitOf} />

  <div class="chart" class:folded={fold} id="run-chart" role="tabpanel" aria-label={current.title}>
    {#if fold}
      <!-- Phones, folded: the chart's one-line readout, which opens it, one state in every section. Open, the picker,
           the qualifier and the legend say all of this, so the row goes and "Hide chart" sits in the chart's header. -->
      <button type="button" class="readout" aria-expanded="false" aria-controls="run-chart-body" bind:this={readoutEl} onclick={() => setFold(true)}>
        <span class="r-l">{current.label}</span>
        <span class="r-v num">{current.value}{#if current.unit && current.value !== '—'}{@render unitOf(current)}{/if}</span>
        <span class="r-c">{current.caption}</span>
        <span class="r-a">Show chart<ChevronDown size={14} aria-hidden="true" /></span>
      </button>
    {/if}
    {#if !fold}
      <div class="chart-body" id="run-chart-body">
        <header>
          <!-- Phones: the picker names the metric in place of the title (the tabs above are hidden), and "Hide chart"
               folds the chart from its own header row. -->
          {#if narrow}
            <div class="pick">{@render picker()}</div>
            {#if !queued}
              <span class="hide" bind:this={hideEl}>
                <Button size="sm" variant="ghost" iconRight={ChevronUp} aria-expanded="true" aria-controls="run-chart-body" onclick={() => setFold(false)}>Hide chart</Button>
              </span>
            {/if}
          {/if}
          <h2><span class="t">{current.title}</span><span class="q">{narrow ? phoneQualifier : current.qualifier}{#if current.qExp}<sup class="exp">{current.qExp}</sup>{/if}</span></h2>
          <div class="ctl">
            <!-- One control for the loss view (smoothing and y scale behind one button), so the header keeps at most one
                 segmented group. -->
            {#if tab === 'loss'}
              <!-- What the terms mean, for anyone who has not trained a detector before: a tap, a click or Enter opens
                   it (ui/InfoButton). -->
              <InfoButton chart label="What the loss terms mean" text="What training lowers: total, the weighted sum of box (placement), cls (class) and dfl (edges)." how={lossHow} />
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
        <!-- Phones: the loss terms' ⓘ opens here, between the heading and the legend's readout, not over the plot. -->
        <InfoRegion slot={lossInfo} />
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
    /* The in-place ⓘ region (phones) on the chart's own edges; the grid's gap spaces it. */
    --info-inset: 0;
    --info-after: 0;
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
  /* Phones, the chart folded: one 44 px readout row that opens it. */
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
    color: var(--text-secondary);
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
  @media (max-width: 767px) {
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
    .chart {
      padding: 0 var(--space-3) var(--space-3);
    }
    /* Two rows, 12 px under the epoch row (or the panel's top): the metric picker, with "Hide chart" at the row's end;
       then the one summary line (what the x axis is, the best where the legend does not already show it) with the
       chart's range control on its right. The legend follows: a value appears once above its chart. */
    .chart header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: var(--space-2) var(--space-3);
      min-height: 0;
    }
    .pick {
      min-width: 0;
    }
    .hide {
      display: flex;
    }
    h2 {
      grid-column: 1;
      grid-row: 2;
    }
    .ctl {
      grid-column: 2;
      grid-row: 2;
      justify-content: flex-end;
    }
    .chart header:not(:has(.ctl > *)) h2 {
      grid-column: 1 / -1;
    }
    /* Figures people read (the best and its epoch): secondary, as a KPI caption (DESIGN_DIRECTION §2.3). */
    .q {
      white-space: normal;
      color: var(--text-secondary);
    }
  }
  /* The fading wrapper of a swapped chart: it takes the chart's place in the layout. */
  .swap {
    min-width: 0;
  }
</style>
