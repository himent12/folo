/**
 * The run's metrics as one instrument: the epoch progress, a row of KPI tabs (each with its current value) and the one
 * chart the selected tab names. A single run is drawn in ink (primary metric in text.primary, secondary ones
 * tertiary with dash patterns); hue is reserved for entities (runs in Compare, classes).
 *
 * Streaming cost: the points come from the run's append logs through ./metricSeries, so a new step maps and smooths
 * that step alone (the Svelte original re-mapped and re-smoothed every step of the run on each update), the per-epoch
 * series change only when an epoch finishes (not on every step), and the loss and throughput y extents are running
 * bounds instead of a loop over every point. Same numbers, bit for bit.
 */
import ChevronDown from '$icons/chevron-down';
import ChevronUp from '$icons/chevron-up';
import { computed, css, deep, effect, fade, For, Key, MediaQuery, signal, tick, untrack, createUniqueId } from 'fw';
import { lr as fmtLr } from '$lib/format';
import { motion } from '$lib/ui/motion';
import { pageKeys } from '$lib/shell/pageCommands';
import InfoButton from '$lib/ui/InfoButton';
import InfoRegion from '$lib/ui/InfoRegion';
import { provideInfoSlot } from '$lib/ui/infoSlot';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import { BestPulse, CompleteFocus } from '$lib/charts/bestPulse';
import { scoresTitle, type EvalSource } from '$lib/evaluated';
import LossOptions, { DEFAULT_SMOOTHING, type Scale, type Smoothing } from '$lib/charts/LossOptions';
import { holdDomain, logDomain, niceDomain, unitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, type Fit, type Held } from '$lib/charts/domain';
import type { Pt } from '$lib/charts/downsample';
import LineChart, { type Marker, type Series } from '$lib/charts/LineChart';
import { runHorizon } from '$lib/charts/horizon';
import type { AppendLog } from '$lib/api/appendLog';
import type { Epoch, Run, Step } from '$lib/api/types';
import { runProgress } from '$lib/api/progress';
import { effectiveCloseMosaic, warmupEpochs } from './resolved';
import { conf, metric, num } from '$lib/format';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select.tsx';
import { media } from '$lib/design/tokens';
import MetricStrip, { type Tab, type TabDef } from './MetricStrip';
import { EpochSeries, joinExtents, liveHorizon, StepSeries, type Extent } from './metricSeries';

const CHART_KEY = 'folo.run.chartOpen';
/**
 * Phones: whether the run's chart is open, one state for every section (Samples, Classes, Logs, Config) and every run
 * this session: the chart never opens on one section and folds on the next (round 26). Open until the person folds it;
 * a tap on its readout row folds or opens it, wherever they are.
 */
export const chartFold = deep({ open: readChartOpen() });
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

/**
 * "3.42" and "× 10⁻⁴" from 3.42e-4: the value set large, its power of ten as the unit, so a tab never cuts it off.
 * `exp` ("−4") is the exponent in plain digits, for a real superscript at a legible size (Unicode superscripts render
 * at about 8 px in a 12 px unit).
 */
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

const INK = 'var(--text-primary)';
const INK2 = 'var(--text-tertiary)';
const ACTIVE = ['queued', 'running', 'stopping'];

/** The loss terms in plain words (DESIGN_DIRECTION §2.3: no jargon without its definition). */
const LOSS_TERMS: [term: string, means: string][] = [
  ['total', 'box, cls and dfl weighted and added: what training lowers.'],
  ['box', 'box regression: how far predicted boxes are from the true ones.'],
  ['cls', 'classification: how wrong the class scores are.'],
  ['dfl', 'distribution focal: how sharply each box edge is placed.'],
  ['EMA', 'smoothing over steps (0.98: about the last 50); a higher value reacts more slowly. The Loss tab shows the last raw step and the smoothed total.'],
];

export interface RunMetricsProps {
  run: Run;
  /** The run's steps (`live.stepsLog`), read after `version` bumps: only what was added since. */
  steps: AppendLog<Step>;
  /** The run's finished epochs (`live.epochsLog`). */
  epochs: AppendLog<Epoch>;
  version: number;
  loading: boolean;
  /**
   * A queued run with nothing yet: the same frame as a running run (KPI tabs reading "—", the empty plot); on phones a
   * one-line "Metrics appear when training starts" that opens it.
   */
  queued?: boolean;
  /** What the epochs' scores are on (lib/evaluated `runEvalSource`): the titles name the split when it is not validation. */
  evalOn?: EvalSource | null;
  /** The run's history failed to load: the panel says so with Retry, never "Waiting for the first validation…". */
  error?: unknown;
  retry?: () => void;
}

export default function RunMetrics(props: RunMetricsProps) {
  // The loss terms' ⓘ opens in place on phones (ui/infoSlot), under the chart's heading.
  const lossInfo = provideInfoSlot('run-chart-info');
  const queued = () => props.queued ?? false;
  const evalOn = () => props.evalOn ?? null;

  const tab = signal<Tab>('map');

  // Phones (< 768 px) turn the six tabs into one metric picker in the chart header, where it replaces the title, and
  // get a shorter plot, so the samples start within the first screen. Tablets keep the tabs (scrolling sideways when
  // a value needs it), as on laptops and desktops.
  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = () => phoneQuery.current;
  // Phones: a queued run's plot is the axes and one line, folded until asked for. An ended run that learned almost
  // nothing (its best mAP50-95 gets a zoomed 0–1 axis, ../../lib/charts/domain UNIT_ZOOM_BELOW) gets a
  // shorter plot on every tab, so its few low samples are not a thin line along the floor of a 320 px frame.
  const flat = computed(() => !ACTIVE.includes(props.run.status) && unitTop(props.run.best?.map50_95 ?? 0) < UNIT_ZOOM_BELOW);
  const height = computed(() => (queued() && narrow() ? 120 : narrow() ? 168 : flat.get() ? 200 : 320));
  const queuedOpen = signal(false);
  const queuedFold = computed(() => queued() && narrow() && !queuedOpen.get());
  // Phones: folded to its readout row while the person has folded it, the same in every section (chartFold).
  const fold = computed(() => narrow() && !queued() && !chartFold.open);

  const live = computed(() => ACTIVE.includes(props.run.status));
  const progress = computed(() => runProgress(props.run));
  /** Epochs without mosaic as the trainer runs them (capped at half the run: 7 of a 15-epoch run asked for 10). */
  const closeN = computed(() => effectiveCloseMosaic(props.run));
  const closeFrom = computed(() => 1 - closeN.get() / props.run.epochs);
  /** The run's last epochs train without mosaic: the bar hatches them and every chart marks where they start. */
  const mosaicOff = computed(() => closeFrom.get() > 0 && closeFrom.get() < 1);
  /** What the hatch means, for where the run is: its caption, and in full its tooltip and the bar's description. */
  const closeText = computed(() => mosaicOffText(closeN.get(), props.run.epochs, props.run.epoch, live.get()), {
    equals: (a, b) => a.cap === b.cap && a.full === b.full,
  });
  const uid = createUniqueId();
  const closeId = `${uid}-close`;
  const elapsed = computed(() => {
    const run = props.run;
    return run.started ? ((run.finished ? Date.parse(run.finished) : Date.now()) - Date.parse(run.started)) / 1000 : null;
  });

  // ---- the series, from the append logs ----
  const stepData = new StepSeries(untrack(() => props.steps));
  const epochData = new EpochSeries(untrack(() => props.epochs));
  /** Moves when the steps (or their x, steps per epoch) changed: what a per-step reading depends on. */
  const stepsRev = computed(() => {
    void props.version;
    stepData.sync(props.run.steps_per_epoch);
    return stepData.rev;
  });
  /** Moves when the epochs changed (once an epoch, not on every step). */
  const epochsRev = computed(() => {
    void props.version;
    epochData.sync();
    return epochData.rev;
  });

  const lastStep = computed(() => {
    stepsRev.get();
    return stepData.steps[stepData.steps.length - 1];
  });
  const lastEpoch = computed(() => {
    epochsRev.get();
    return epochData.epochs[epochData.epochs.length - 1];
  });

  // ---- chart data ----
  const smoothing = signal<Smoothing>(DEFAULT_SMOOTHING);
  // Log by default: the total starts near 8 while box / cls / dfl sit between 0.4 and 3, which a linear axis squashes
  // into its bottom quarter.
  const scale = signal<Scale>('log');
  const horizon = signal<'observed' | 'full'>('observed');
  const observed = computed(() => {
    stepsRev.get();
    const spe = stepData.spe;
    const ls = lastStep.get();
    const le = lastEpoch.get();
    return Math.max(ls ? ls.step / spe : 0, le ? le.epoch : 0);
  });
  // While the run trains, "So far" grows in steps (./metricSeries liveHorizon), so the axis holds still.
  // The first epoch carries the start (loading, compiling the GPU pipelines): its time left swings by minutes. The ETA
  // shows once an epoch has finished, or halfway through a one-epoch run.
  const etaReady = computed(() => !!lastEpoch.get() || (props.run.epochs === 1 && observed.get() >= 0.5));
  const xDomain = computed(() => (live.get() ? liveHorizon : runHorizon)(observed.get(), props.run.epochs, horizon.get()), {
    equals: (a, b) => a[0] === b[0] && a[1] === b[1],
  });
  const epochTitle = (x: number) => `Epoch ${x.toFixed(x % 1 === 0 ? 0 : 1)}`;

  /**
   * How the focal line ends, in StatusBadge's marks (LineChart `end`): a filled lime dot with the heartbeat while the run
   * trains, a square once stopped, a cross when it failed, a plain dot when it completed. With the hollow ink ring on
   * the best epoch (the header's "○ best") this is the focal chart's vocabulary, shared with Overview's live card.
   */
  const endMark = computed(() => {
    const s = props.run.status;
    return s === 'running' ? ('live' as const) : s === 'stopped' ? ('stopped' as const) : s === 'failed' ? ('failed' as const) : undefined;
  });
  const alpha = () => Number(smoothing.get());
  /** The total loss as the chart draws it (smoothed at the chosen EMA): its last value is the Loss tab's caption. */
  const smoothedLast = computed(() => {
    stepsRev.get();
    return smoothing.get() !== '0' ? stepData.smoothed('total', alpha()).pts.at(-1)?.y : undefined;
  });
  const ep = (f: 'map50_95' | 'map50' | 'precision' | 'recall'): Pt[] => epochData.points(f).slice();
  /** The step-based lists the current tab draws, with their bounds (for the y extent without a loop over the points). */
  const stepTracks = computed((): Extent[] => {
    const t = tab.get();
    stepsRev.get();
    if (t === 'loss') {
      const a = alpha();
      return [...(a > 0 ? [stepData.raw('total').ext] : []), ...(['total', 'box', 'cls', 'dfl'] as const).map((f) => stepData.smoothed(f, a).ext)];
    }
    if (t === 'ips') return [stepData.smoothed('ips', 0.9).ext];
    return [];
  });
  const series = computed((): Series[] => {
    switch (tab.get()) {
      case 'map':
        epochsRev.get();
        return [
          { id: 'm95', label: 'mAP50-95', color: INK, area: true, end: endMark.get(), points: ep('map50_95') },
          { id: 'm50', label: 'mAP50', color: INK2, pattern: 'dash', points: ep('map50') },
        ];
      case 'map50':
        epochsRev.get();
        return [
          { id: 'm50', label: 'mAP50', color: INK, area: true, end: endMark.get(), points: ep('map50') },
          { id: 'm95', label: 'mAP50-95', color: INK2, pattern: 'dash', points: ep('map50_95') },
        ];
      case 'pr':
        epochsRev.get();
        return [
          { id: 'p', label: 'Precision', color: INK, end: endMark.get(), points: ep('precision') },
          { id: 'r', label: 'Recall', color: INK2, pattern: 'dash', points: ep('recall') },
        ];
      case 'loss': {
        stepsRev.get();
        const a = alpha();
        const total = stepData.raw('total').pts;
        return [
          { id: 'raw', label: 'total (raw)', color: INK, ghost: true, points: a > 0 ? total.slice() : [] },
          // One name per series (DESIGN_DIRECTION §2.23): the legend, the end labels and the phone legend all say
          // total, box, cls and dfl, with the full names under ⓘ (LOSS_TERMS). Raw vs smoothed is said once, in the
          // Loss tab's caption ("raw · smoothed 2.075"); the smoothing itself in the loss view's control.
          { id: 'total', label: 'total', color: INK, emphasis: true, points: stepData.smoothed('total', a).pts.slice() },
          // The components as design/README.md maps them in both apps: box amber, cls violet dashed, dfl teal dotted.
          // Box and dfl run together late in a run (0.85 vs 0.83): box is painted over dfl's dots, and each line is
          // labelled at its end, so both stay readable where they meet.
          { id: 'box', label: 'box', color: 'var(--chart-2)', front: true, points: stepData.smoothed('box', a).pts.slice() },
          { id: 'cls', label: 'cls', color: 'var(--chart-5)', pattern: 'dash', points: stepData.smoothed('cls', a).pts.slice() },
          { id: 'dfl', label: 'dfl', color: 'var(--chart-4)', pattern: 'dot', points: stepData.smoothed('dfl', a).pts.slice() },
        ];
      }
      case 'lr': {
        stepsRev.get();
        // Every k-th step: at most about 1,200 points, whatever the run's length.
        const all = stepData.raw('lr').pts;
        const k = Math.max(1, Math.floor(stepData.steps.length / 600));
        const points: Pt[] = [];
        for (let i = 0; i < all.length; i += k) points.push(all[i]!);
        return [{ id: 'lr', label: 'Learning rate', color: INK, points }];
      }
      case 'ips':
        stepsRev.get();
        return [{ id: 'ips', label: 'Images/s', color: INK, points: stepData.smoothed('ips', 0.9).pts.slice() }];
    }
  });

  // LR ticks share one power of ten (stated in the header): "2" "4" with "× 10⁻³" instead of "2.00e-3".
  const lrExp = computed(() => {
    stepsRev.get();
    const m = stepData.lrMax();
    return m > 0 ? Math.floor(Math.log10(m)) : 0;
  });

  // ---- y domain ----
  // Fitted to the data: a failed run's mAP of 0.5 fills the plot instead of half of a 0–1 axis, and a run that learned
  // almost nothing gets a zoomed axis that says so (its caption names the full 0–1 scale). A live chart re-fits at
  // most once per epoch (./domain holdDomain), and values never tween.
  const unit = computed(() => tab.get() === 'map' || tab.get() === 'map50' || tab.get() === 'pr');
  const extent = computed(
    (): [number, number] | null => {
      const t = tab.get();
      const logY = t === 'loss' && scale.get() === 'log';
      // The loss and throughput lists keep running bounds (./metricSeries): the same numbers as the loop below.
      if (t === 'loss' || t === 'ips') return joinExtents(stepTracks.get(), logY);
      let lo = Infinity;
      let hi = -Infinity;
      for (const s of series.get())
        for (const p of s.points) {
          if (!Number.isFinite(p.y) || (logY && p.y <= 0)) continue;
          if (p.y < lo) lo = p.y;
          if (p.y > hi) hi = p.y;
        }
      return Number.isFinite(lo) ? [lo, hi] : null;
    },
    { equals: (a, b) => a === b || (!!a && !!b && a[0] === b[0] && a[1] === b[1]) },
  );
  const zoomed = computed(() => {
    const e = extent.get();
    return unit.get() && !!e && unitTop(e[1]) < UNIT_ZOOM_BELOW;
  });
  // The plot inside LineChart's margins (8 px top, 24 with the mosaic mark's label or the zoom caption; 22 px bottom).
  const plotPx = computed(() => height.get() - (mosaicOff.get() || zoomed.get() ? 46 : 30));
  let held: Held<Fit & { log?: boolean }> | null = null;
  const fit = computed((): (Fit & { log?: boolean }) | null => {
    const e = extent.get();
    if (!e) return null;
    const [lo, hi] = e;
    const t = tab.get();
    let next: Fit & { log?: boolean };
    if (unit.get()) next = unitDomain(hi, plotPx.get());
    else if (t === 'loss' && scale.get() === 'log') next = { domain: logDomain(lo, hi), ticks: [], decimals: 0, log: true };
    // Throughput starts at 0 (a rate), with a little headroom over its peak.
    else if (t === 'ips') next = niceDomain(0, hi * 1.05);
    else next = niceDomain(lo, hi);
    const epochNow = live.get() ? Math.floor(observed.get()) : -1;
    held = holdDomain(held, `${t}|${scale.get()}|${smoothing.get()}|${height.get()}|${epochNow}`, next);
    return held.value;
  });

  // A new best while the run trains pulses its ring once (LineChart, over the chart duration; nothing moves under reduced motion).
  const bestEpochNo = computed(() => props.run.best?.epoch ?? null);
  const running = computed(() => props.run.status === 'running');
  const status = computed(() => props.run.status);
  const pulser = new BestPulse();
  const pulse = signal(0);
  effect.pre(() => {
    pulse.set(pulser.see(bestEpochNo.get(), running.get()));
  });
  // The run completing while the page is open focuses its best ring once (lime viewfinder corners; none under reduced motion).
  const focuser = new CompleteFocus();
  const focus = signal(0);
  effect.pre(() => {
    focus.set(focuser.see(status.get()));
  });
  const runBest = computed(() => props.run.best);
  const best = computed<Marker[]>(() => {
    const b = runBest.get();
    const t = tab.get();
    if (!b || (t !== 'map' && t !== 'map50')) return [];
    let y = b.map50_95;
    if (t !== 'map') {
      epochsRev.get();
      y = epochData.epochs.find((e) => e.epoch === b.epoch)?.val.map50 ?? 0;
    }
    return [{ x: b.epoch, y, label: 'best', quiet: true, pulse: pulse.get(), focus: focus.get() }];
  });

  // The one live-value label, as in the chart's legend: "Latest · ep 37". Per-epoch values come from the last
  // validation; per-step values from the epoch the last step is in.
  const latestEp = computed(() => (lastEpoch.get() ? `Latest · ep ${lastEpoch.get()!.epoch}` : 'after the first epoch'));
  // One label for every per-step reading (the Loss, LR and Throughput tabs and the legend), "(partial)" past the end.
  const stepEpWord = computed(() => stepEpoch(props.run.status, observed.get(), lastEpoch.get()?.epoch ?? 0));
  const stepEp = computed(() => (lastStep.get() ? `Latest · ${stepEpWord.get()}` : 'after the first step'));
  // Tablets: the same readings in fewer words ("ep 37", "not yet").
  const latestShort = computed(() => (lastEpoch.get() ? `ep ${lastEpoch.get()!.epoch}` : 'not yet'));
  const stepShort = computed(() => (lastStep.get() ? stepEpWord.get() : 'not yet'));
  /** Precision and recall are read at each epoch's best-F1 confidence (docs/API.md, Operating point): the latest one. */
  const prConf = computed(() => lastEpoch.get()?.val.best_f1_conf);
  const warmupWords = (r: NonNullable<Run['resolved']>) => {
    const e = warmupEpochs(r);
    return r.warmup_iters <= 0 ? 'no warmup' : `warmup ${e} ${e === 1 ? 'epoch' : 'epochs'}`;
  };
  /** The LR schedule the trainer resolved (`Run.resolved`), else what the recipe defaults to. */
  const schedule = computed(() => {
    // In epochs, as the chart's axis is (Config has the optimizer steps beside them): short, and never a bare "steps".
    const r = props.run.resolved;
    return r ? `${warmupWords(r)} · ${r.cos_lr ? 'cosine' : 'linear'}` : 'warmup 3 epochs · linear';
  });
  const tabs = computed((): TabDef[] => {
    const run = props.run;
    const le = lastEpoch.get();
    const ls = lastStep.get();
    const pc = prConf.get();
    const sl = smoothedLast.get();
    return [
      {
        id: 'map',
        label: 'mAP50-95',
        value: metric(le?.val.map50_95),
        caption: run.best ? `best ${metric(run.best.map50_95)} · ep ${run.best.epoch}` : 'after the first epoch',
        capShort: run.best ? `best · ep ${run.best.epoch}` : 'not yet',
        title: `${scoresTitle(evalOn())} mAP50-95`,
        qualifier: run.best ? `by epoch · ○ best ${metric(run.best.map50_95)} at epoch ${run.best.epoch}` : 'by epoch',
      },
      { id: 'map50', label: 'mAP50', value: metric(le?.val.map50), caption: latestEp.get(), capShort: latestShort.get(), title: `${scoresTitle(evalOn())} mAP50`, qualifier: 'by epoch' },
      {
        id: 'pr',
        label: 'Precision · Recall',
        short: 'P · R',
        value: le ? `${metric(le.val.precision)} · ${metric(le.val.recall)}` : '—',
        // The latest epoch's threshold: "conf 0.31 · ep 37" (each epoch has its own, so the chart says "each at").
        caption: le ? (pc != null ? `conf ${conf(pc)} · ep ${le.epoch}` : `best-F1 conf · ep ${le.epoch}`) : 'after the first epoch',
        capShort: le && pc != null ? `conf ${conf(pc)}` : latestShort.get(),
        title: 'Precision and recall',
        qualifier: pc != null ? `by epoch · at each epoch's best-F1 conf (latest ${conf(pc)})` : "by epoch · at each epoch's best-F1 confidence",
      },
      {
        id: 'loss',
        label: 'Loss',
        value: num(ls?.loss.total, 3),
        // The last step's raw reading; the chart draws the total smoothed. The caption is the one place that says which
        // is which ("raw · smoothed 2.075", the legend's total), so the two numbers never seem to disagree.
        caption: ls ? (sl !== undefined ? `raw · smoothed ${num(sl, 3)}` : `raw · ${stepEpWord.get()}`) : 'after the first step',
        capShort: ls ? (sl !== undefined ? `raw · smoothed ${num(sl, 3)}` : `raw · ${stepShort.get()}`) : stepShort.get(),
        title: 'Training loss',
        qualifier: 'by epoch',
      },
      {
        id: 'lr',
        label: 'Learning rate',
        short: 'LR',
        // The power of ten is the value's unit, set smaller, so the tab never cuts the value off.
        ...lrParts(ls?.lr),
        caption: schedule.get(),
        capShort: run.resolved ? (run.resolved.cos_lr ? 'cosine' : 'linear') : 'linear',
        title: 'Learning rate',
        qualifier: 'by epoch · × 10',
        qExp: String(lrExp.get()).replace('-', '−'),
      },
      {
        id: 'ips',
        label: 'Throughput',
        value: num(ls?.imgs_per_s, 0),
        unit: 'img/s',
        caption: stepEp.get(),
        capShort: stepShort.get(),
        title: 'Throughput',
        qualifier: 'images/s by epoch',
      },
    ];
  });
  /** The legend's note: the latest value's epoch, per epoch for validation metrics, per step for the rest. */
  const legendNote = computed(() => (unit.get() ? (lastEpoch.get() ? latestEp.get() : undefined) : lastStep.get() ? stepEp.get() : undefined));

  // The last epochs train without mosaic (the hatched end of the progress bar): marked where they start, on every chart.
  const xMark = computed(() => (mosaicOff.get() ? { x: props.run.epochs - closeN.get(), label: 'mosaic off' } : undefined), {
    equals: (a, b) => a === b || (!!a && !!b && a.x === b.x),
  });
  const current = computed(() => tabs.get().find((t) => t.id === tab.get())!);
  /**
   * Phones, the chart open: the one line between the metric picker and the legend. A value appears once above its chart
   * (DESIGN_DIRECTION §12): the legend reads the latest value (or the cursor's), so the best is stated here only where
   * it differs from it ("○ best 0.599 · ep 36"); a run whose best is its last epoch says where ("○ best · ep 150").
   */
  const phoneQualifier = computed(() => {
    const b = props.run.best;
    return tab.get() === 'map' && b ? `by epoch · ○ best${b.epoch === lastEpoch.get()?.epoch ? '' : ` ${metric(b.map50_95)}`} · ep ${b.epoch}` : current.get().qualifier;
  });

  // Phones: folding hands focus to the control that reopens it, and opening to the one that folds it again, so a
  // keyboard never lands on <body> when the button it pressed goes away.
  const readoutEl = signal<HTMLButtonElement>();
  const hideEl = signal<HTMLElement>();
  async function setFold(open: boolean) {
    setChartOpen(open);
    await tick();
    (open ? hideEl.peek()?.querySelector<HTMLElement>('button') : readoutEl.peek())?.focus();
  }

  // 1–6 pick a metric tab from anywhere on the page (a single-key shortcut: off with the others in Settings).
  pageKeys((k) => {
    const t = /^[1-9]$/.test(k) ? tabs.peek()[Number(k) - 1] : undefined;
    if (!t) return false;
    tab.set(t.id);
    return true;
  });

  // The loss terms in full, behind the ⓘ's short line ("How it's computed").
  const lossHow = () => (
    <span class="terms-list">
      <For each={LOSS_TERMS} key={(t) => t[0]}>
        {(t) => (
          <span>
            <b>{t()[0]}</b>: {t()[1]}
          </span>
        )}
      </For>
    </span>
  );

  // A power of ten as "× 10" with a real 12 px superscript; other units as they are.
  const unitOf = (t: () => TabDef) => (
    <>
      {t().exp ? (
        <span class="unit" aria-label={t().unit}>
          × 10<sup class="exp">{t().exp}</sup>
        </span>
      ) : (
        <span class="unit">{t().unit}</span>
      )}
    </>
  );

  // The metric's name only: its value is the legend's, under the picker (a value appears once above its chart).
  const picker = () => <Select label="Metric to chart" bind:value={tab} options={tabs.get().map((t) => ({ value: t.id, label: t.label }))} />;

  return (
    <section class="metrics" aria-label="Run metrics" data-tip-bounds>
      {props.error !== undefined ? (
        <div class="failed">
          <ErrorState error={props.error} title="Could not load this run's metrics" retry={props.retry} compact />
        </div>
      ) : queued() && narrow() ? (
        // Phones, queued: one line until there is something to chart; the banner above says when it starts.
        <button type="button" class="pending" aria-expanded={queuedOpen.get()} aria-controls="run-metrics-body" onclick={() => queuedOpen.set(!queuedOpen.peek())}>
          <span class="p-t">Metrics appear when training starts</span>
          <span class="r-a">
            {queuedOpen.get() ? 'Hide chart' : 'Show chart'}
            <ChevronDown size={14} aria-hidden="true" />
          </span>
        </button>
      ) : null}
      {props.error === undefined && !queuedFold.get() ? (
        <div class="metrics-body" id="run-metrics-body">
          <MetricStrip
            run={props.run}
            live={live.get()}
            progress={progress.get()}
            mosaicOff={mosaicOff.get()}
            closeFrom={closeFrom.get()}
            closeId={closeId}
            closeText={closeText.get()}
            elapsed={elapsed.get()}
            etaReady={etaReady.get()}
            tabs={tabs.get()}
            bind:tab={tab}
            unitOf={unitOf}
          />

          <div class="chart" class:folded={fold.get()} id="run-chart" role="tabpanel" aria-label={current.get().title}>
            {fold.get() ? (
              // Phones, folded: the chart's one-line readout, which opens it, one state in every section. Open, the
              // picker, the qualifier and the legend say all of this, so the row goes and "Hide chart" sits in the
              // chart's header.
              <button type="button" class="readout" aria-expanded="false" aria-controls="run-chart-body" ref={readoutEl} onclick={() => setFold(true)}>
                <span class="r-l">{current.get().label}</span>
                <span class="r-v num">
                  {current.get().value}
                  {current.get().unit && current.get().value !== '—' ? unitOf(() => current.get()) : null}
                </span>
                <span class="r-c">{current.get().caption}</span>
                <span class="r-a">
                  Show chart
                  <ChevronDown size={14} aria-hidden="true" />
                </span>
              </button>
            ) : null}
            {!fold.get() ? (
              <div class="chart-body" id="run-chart-body">
                <header>
                  {/* Phones: the picker names the metric in place of the title (the tabs above are hidden), and "Hide
                      chart" folds the chart from its own header row. */}
                  {narrow() ? (
                    <>
                      <div class="pick">{picker()}</div>
                      {!queued() ? (
                        <span class="hide" ref={hideEl}>
                          <Button size="sm" variant="ghost" iconRight={ChevronUp} aria-expanded="true" aria-controls="run-chart-body" onclick={() => setFold(false)}>
                            Hide chart
                          </Button>
                        </span>
                      ) : null}
                    </>
                  ) : null}
                  <h2>
                    <span class="t">{current.get().title}</span>
                    <span class="q">
                      {narrow() ? phoneQualifier.get() : current.get().qualifier}
                      {current.get().qExp ? <sup class="exp">{current.get().qExp}</sup> : null}
                    </span>
                  </h2>
                  <div class="ctl">
                    {/* One control for the loss view (smoothing and y scale behind one button), so the header keeps at
                        most one segmented group. */}
                    {tab.get() === 'loss' ? (
                      <>
                        {/* What the terms mean, for anyone who has not trained a detector before: a tap, a click or
                            Enter opens it (ui/InfoButton). */}
                        <InfoButton
                          chart
                          label="What the loss terms mean"
                          text="What training lowers: total, the weighted sum of box (placement), cls (class) and dfl (edges)."
                          how={lossHow}
                        />
                        <LossOptions bind:smoothing={smoothing} bind:scale={scale} />
                      </>
                    ) : null}
                    {live.get() && props.run.status !== 'queued' ? (
                      <Segmented
                        size="sm"
                        label="Chart range"
                        bind:value={horizon}
                        options={[
                          { value: 'observed', label: 'So far' },
                          { value: 'full', label: `All ${props.run.epochs}` },
                        ]}
                      />
                    ) : null}
                  </div>
                </header>
                {/* Phones: the loss terms' ⓘ opens here, between the heading and the legend's readout, not over the plot. */}
                <InfoRegion slot={lossInfo} />
                {props.loading ? (
                  <ChartSkeleton height={height.get()} />
                ) : (
                  // A tab swap fades the new chart in over 160 ms (0 under reduced motion, ui/motion).
                  <Key when={tab.get()}>
                    <div class="swap" in:fade={{ duration: motion(160) }}>
                      {/* No point yet: the legend's row is held empty, so the plot does not drop when it appears. */}
                      {!queued() && !series.get().some((x) => x.points.length > 0 && !x.ghost && !x.context && !x.lower) ? (
                        <div class="legend-slot" aria-hidden="true"></div>
                      ) : null}
                      <LineChart
                        series={series.get()}
                        height={height.get()}
                        xDomain={xDomain.get()}
                        xInteger
                        yDomain={fit.get()?.domain}
                        yTickValues={fit.get() && !fit.get()!.log ? fit.get()!.ticks : undefined}
                        yDecimals={fit.get() && !fit.get()!.log ? fit.get()!.decimals : undefined}
                        yLog={tab.get() === 'loss' && scale.get() === 'log'}
                        yFormat={tab.get() === 'lr' ? (v) => String(+(v / 10 ** lrExp.get()).toFixed(2)) : undefined}
                        valueFormat={
                          tab.get() === 'lr' ? (v) => (v / 10 ** lrExp.get()).toFixed(2) : tab.get() === 'ips' ? (v) => `${num(v, 0)} img/s` : (v) => metric(v)
                        }
                        xTitle={epochTitle}
                        curve={unit.get() ? 'monotone' : 'linear'}
                        markers={best.get()}
                        endDots={unit.get()}
                        xMark={xMark.get()}
                        yLabel={zoomed.get() ? UNIT_ZOOM_NOTE : undefined}
                        legendNote={legendNote.get()}
                        label={`${current.get().title} by epoch`}
                        endLabels={tab.get() === 'loss'}
                        empty={queued() ? 'Curves appear here once training starts' : unit.get() ? 'Waiting for the first validation…' : 'Waiting for the first step…'}
                      />
                    </div>
                  </Key>
                )}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}

css`
  .metrics {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  .failed {
    padding: var(--space-4);
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
  /* The legend's line and the chart's gap under it (lib/charts LineChart, ChartLegend). */
  .legend-slot {
    height: calc(var(--type-sm-line) + var(--space-2));
  }
`;
