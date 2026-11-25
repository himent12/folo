/**
 * What a run's page reads from its history, once, for the KPI strip and the four charts (./RunKpis, ./RunChart): the
 * series in the series hues (mAP50-95 sky with the page's one area fill, mAP50 amber dashed, precision magenta, recall
 * teal dashed, the learning rate neutral, the loss terms in their README hues), their fitted y domains, the one x axis
 * the charts share (the planned run while it trains, its future shaded with the ETA over the boundary; "So far" fits
 * every chart to the epochs trained) and a synchronized crosshair cursor (run/RunChart).
 *
 * Streaming cost: the points come from the run's append logs through ./metricSeries, so a new step maps and smooths
 * that step alone, the per-epoch series change only when an epoch finishes (not on every step), and the loss y extent
 * is a running bound instead of a loop over every point.
 */
import { computed, effect, signal, untrack, type Signal } from 'fw';
import type { AppendLog } from '$lib/api/appendLog';
import type { Epoch, Run, Step } from '$lib/api/types';
import { etaSettled, heldResults, runProgress, stoppedEarly, trainingSeconds } from '$lib/api/progress';
import { BestPulse, CompleteFocus } from '$lib/charts/bestPulse';
import { holdDomain, logDomain, niceDomain, unitDomain, type Fit, type Held } from '$lib/charts/domain';
import type { Pt } from '$lib/charts/downsample';
import { runHorizon } from '$lib/charts/horizon';
import type { Future, Marker, Series, XMark } from '$lib/charts/LineChart';
import type { Scale, Smoothing } from '$lib/charts/LossOptions';
import { axisMetric, eta } from '$lib/format';
import { prefs } from '$lib/prefs';
import { countNonPositive, EpochSeries, joinExtents, liveHorizon, StepSeries } from './metricSeries';
import { failedEpoch } from './failure';
import { mosaicOffText, stepEpoch, epochsDone } from './readings';
import { effectiveCloseMosaic, warmupEpochs } from './resolved';

export const ACTIVE = ['queued', 'running', 'stopping'];
/** The steps the Throughput tile's sparkline and average read: the last 120, as Desktop's tile. */
export const IPS_WINDOW = 120;

/** What the model reads; getters, so it follows the page's live run (read only once it has loaded). */
export interface RunSource {
  readonly run: Run | undefined;
  /** The run's steps (`live.stepsLog`), read after `version` bumps: only what was added since. */
  readonly steps: AppendLog<Step>;
  /** The run's finished epochs (`live.epochsLog`). */
  readonly epochs: AppendLog<Epoch>;
  readonly version: number;
  /** The run as the server's latest snapshot has it (`live.server`): how many epochs an ended run validated. */
  readonly server?: Run;
  /** History answers in a row short of that snapshot (`live.shortAnswers`): at `SHORT_ANSWERS`, all there is. */
  readonly shortAnswers?: number;
}

type EpochField = 'map50_95' | 'map50' | 'precision' | 'recall';
export type ChartFit = Fit & { log?: boolean };

/** A y domain fitted to one chart's data and held between epochs (lib/charts/domain holdDomain). */
function held(key: () => string, next: () => ChartFit | null) {
  let h: Held<ChartFit> | null = null;
  return computed((): ChartFit | null => {
    const n = next();
    if (!n) return null;
    h = holdDomain(h, key(), n);
    return h.value;
  });
}

/** The [lo, hi] of some series' finite values (a log axis skips values at or under 0); null without one. */
function extentOf(series: Series[], log = false): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series)
    for (const p of s.points) {
      if (!Number.isFinite(p.y) || (log && p.y <= 0)) continue;
      if (p.y < lo) lo = p.y;
      if (p.y > hi) hi = p.y;
    }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

export class RunModel {
  readonly run: () => Run;
  /** The run as the server's latest snapshot has it, else as the view has it: what an ended run validated. */
  private readonly server: () => Run;
  private readonly shortAnswers: () => number;
  readonly live = computed(() => ACTIVE.includes(this.run().status));
  readonly progress = computed(() => runProgress(this.run()));
  readonly elapsed = computed(() => trainingSeconds(this.run()));

  // ---- the series, from the append logs ----
  private readonly stepData: StepSeries;
  private readonly epochData: EpochSeries;
  /** Moves when the steps (or their x, steps per epoch) changed: what a per-step reading depends on. */
  readonly stepsRev: () => number;
  /** Moves when the epochs changed (once an epoch, not on every step). */
  readonly epochsRev: () => number;
  readonly lastStep = computed(() => {
    this.stepsRev();
    return this.stepData.steps[this.stepData.steps.length - 1];
  });
  readonly lastEpoch = computed(() => {
    this.epochsRev();
    return this.epochData.epochs[this.epochData.epochs.length - 1];
  });
  /** The epoch before the last one: what the mAP tile's change is against. */
  readonly prevEpoch = computed(() => {
    this.epochsRev();
    return this.epochData.epochs[this.epochData.epochs.length - 2];
  });
  /** The furthest x with data, in epochs (fractional while a live run is mid-epoch). */
  readonly observed = computed(() => {
    this.stepsRev();
    const ls = this.lastStep.get();
    const le = this.lastEpoch.get();
    return Math.max(ls ? ls.step / this.stepData.spe : 0, le ? le.epoch : 0);
  });

  // ---- what the charts' controls set ----
  /** The loss lines' EMA and y scale (LossOptions), persisted in prefs. Log by default: total starts near 8, terms 0.4–3. */
  readonly smoothing: Signal<Smoothing> = {
    get: () => prefs.lossSmoothing,
    peek: () => untrack(() => prefs.lossSmoothing),
    set: (v) => {
      prefs.lossSmoothing = typeof v === 'function' ? v(untrack(() => prefs.lossSmoothing)) : v;
    },
  };
  readonly scale: Signal<Scale> = {
    get: () => prefs.lossScale,
    peek: () => untrack(() => prefs.lossScale),
    set: (v) => {
      prefs.lossScale = typeof v === 'function' ? v(untrack(() => prefs.lossScale)) : v;
    },
  };
  /** Shared crosshair position (data x, epochs) synchronized across the run's stacked metric charts. */
  readonly cursor = signal<number | null>(null);
  /** "So far / All N" in the first chart's header, for every chart of the page. While a run trains: the plan. */
  readonly horizon = signal<'observed' | 'full'>('full');

  constructor(src: RunSource) {
    this.run = () => src.run!;
    this.server = () => src.server ?? src.run!;
    this.shortAnswers = () => src.shortAnswers ?? 0;
    this.stepData = new StepSeries(untrack(() => src.steps));
    this.epochData = new EpochSeries(untrack(() => src.epochs));
    const stepsRev = computed(() => {
      void src.version;
      this.stepData.sync(src.run?.steps_per_epoch ?? 0);
      return this.stepData.rev;
    });
    const epochsRev = computed(() => {
      void src.version;
      this.epochData.sync();
      return this.epochData.rev;
    });
    this.stepsRev = () => stepsRev.get();
    this.epochsRev = () => epochsRev.get();
    // A new best while the run trains pulses its ring once; the run completing while the page is open focuses it.
    const pulser = new BestPulse();
    const focuser = new CompleteFocus();
    effect.pre(() => {
      const r = src.run;
      if (r) this.pulse.set(pulser.see(r.best?.epoch ?? null, r.status === 'running'));
    });
    effect.pre(() => {
      const r = src.run;
      if (r) this.focus.set(focuser.see(r.status));
    });
  }

  private readonly pulse = signal(0);
  private readonly focus = signal(0);

  // ---- the x axis, shared ----
  /**
   * The first epoch carries the start (loading, compiling the GPU pipelines): its time left swings by minutes. The ETA
   * shows once an epoch has finished, or halfway through a one-epoch run (lib/api/progress etaSettled).
   */
  readonly etaReady = computed(() => !!this.lastEpoch.get() || etaSettled(this.run()) || (this.run().epochs === 1 && this.observed.get() >= 0.5));
  /** While it trains, the planned run (or "So far", in steps, ./metricSeries liveHorizon); once ended, what it ran. */
  readonly xDomain = computed(
    (): [number, number] => (this.live.get() ? liveHorizon(this.observed.get(), this.run().epochs, this.horizon.get()) : runHorizon(this.observed.get(), this.run().epochs, 'observed')),
    { equals: (a, b) => a[0] === b[0] && a[1] === b[1] },
  );
  /** A training run's epochs still to come, shaded from the latest point; the first chart says when they end. */
  readonly future = computed((): Future | undefined => {
    const r = this.run();
    if (r.status !== 'running' && r.status !== 'stopping') return undefined;
    return { x: this.observed.get(), label: this.etaReady.get() && r.eta_s !== null ? `ETA ${eta(r.eta_s)}` : 'remaining' };
  });

  // ---- mosaic off: the rail's hatched end, a mark on every chart ----
  /** Epochs without mosaic as the trainer runs them (capped at half the run: 7 of a 15-epoch run asked for 10). */
  readonly closeN = computed(() => effectiveCloseMosaic(this.run()));
  readonly closeFrom = computed(() => 1 - this.closeN.get() / this.run().epochs);
  readonly mosaicOff = computed(() => this.closeFrom.get() > 0 && this.closeFrom.get() < 1);
  readonly closeText = computed(() => {
    const r = this.run();
    const ended = r.status === 'stopped' || r.status === 'failed' ? r.status : null;
    // How far it trained, in epochs, steps of an epoch it never finished included (2.25: a quarter into epoch 3).
    const trained = Math.max(this.observed.get(), r.steps_per_epoch > 0 ? r.step / r.steps_per_epoch : 0);
    // A failed run's `epoch` may also be its last finished one (it failed before the next epoch's first step): the
    // history's last epoch says so.
    const done = Math.max(epochsDone(r.status, r.epoch), this.lastEpoch.get()?.epoch ?? 0);
    return mosaicOffText(this.closeN.get(), r.epochs, done, this.live.get(), stoppedEarly(r), ended, trained);
  }, {
    equals: (a, b) => a.cap === b.cap && a.full === b.full,
  });
  readonly xMark = computed((): XMark | undefined => (this.mosaicOff.get() ? { x: this.run().epochs - this.closeN.get(), label: 'mosaic off' } : undefined), {
    equals: (a, b) => a === b || (!!a && !!b && a.x === b.x),
  });

  /**
   * Where a failed run stopped, on Validation mAP and Training loss (design/UNIFIED.md §3 Frame, Desktop's words): its
   * cross says that it failed, this label in which epoch ("failed during epoch 24": the one the Epoch tile and the
   * callout name, ./failure). At the last step, the end of what it ran: the loss line's cross.
   */
  readonly failEvent = computed(
    (): XMark | undefined => {
      const r = this.run();
      const e = r.status === 'failed' ? failedEpoch(r) : null;
      const x = this.observed.get();
      return e !== null && x > 0 ? { x, label: `failed during epoch ${e}` } : undefined;
    },
    { equals: (a, b) => a === b || (!!a && !!b && a.x === b.x && a.label === b.label) },
  );

  // ---- the series ----
  /**
   * How the focal line ends, in StatusBadge's marks (LineChart `end`): a filled lime dot with the heartbeat while the run
   * trains, a square once stopped, a cross when it failed, a plain dot when it completed. Every chart's first series
   * takes it (mAP50-95, precision, total loss, the learning rate or the throughput); see `trailMark` for the others
   * (design/UNIFIED.md §3, Marks).
   */
  readonly endMark = computed(() => {
    const s = this.run().status;
    return s === 'running' ? ('live' as const) : s === 'stopped' ? ('stopped' as const) : s === 'failed' ? ('failed' as const) : undefined;
  });
  /**
   * How a chart's other lines end: a dot of their own hue while the run trains (one lime dot per chart) and once it
   * completed, the run's square or cross once it stopped or failed, as the first line.
   */
  readonly trailMark = computed(() => {
    const end = this.endMark.get();
    return end === 'live' ? undefined : end;
  });
  private ep = (f: EpochField): Pt[] => this.epochData.points(f);
  readonly mapSeries = computed((): Series[] => {
    const rev = this.epochsRev();
    return [
      { id: 'm95', label: 'mAP50-95', color: 'var(--chart-1)', area: true, end: this.endMark.get(), points: this.ep('map50_95'), rev },
      { id: 'm50', label: 'mAP50', color: 'var(--chart-2)', pattern: 'dash', end: this.trailMark.get(), points: this.ep('map50'), rev },
    ];
  });
  readonly prSeries = computed((): Series[] => {
    const rev = this.epochsRev();
    return [
      { id: 'p', label: 'precision', color: 'var(--chart-3)', end: this.endMark.get(), points: this.ep('precision'), rev },
      { id: 'r', label: 'recall', color: 'var(--chart-4)', pattern: 'dash', end: this.trailMark.get(), points: this.ep('recall'), rev },
    ];
  });
  private alpha = () => Number(this.smoothing.get());
  readonly lossSeries = computed((): Series[] => {
    const rev = this.stepsRev();
    const a = this.alpha();
    const d = this.stepData;
    const total = d.smoothed('total', a);
    const box = d.smoothed('box', a);
    const cls = d.smoothed('cls', a);
    const dfl = d.smoothed('dfl', a);
    return [
      { id: 'raw', label: 'total (raw)', color: 'var(--text-primary)', ghost: true, points: a > 0 ? d.raw('total').pts : [], rev },
      // One name per series (DESIGN_DIRECTION §2.23): the legend, the end labels and the phone legend all say total,
      // box, cls and dfl, with the full names under ⓘ (./readings LOSS_TERMS).
      { id: 'total', label: 'total', color: 'var(--text-primary)', emphasis: true, end: this.endMark.get(), points: total.pts, bounds: total.ext, rev },
      // The components as design/README.md maps them in both apps: box amber, cls violet dashed, dfl teal dotted. Box
      // is painted over dfl's dots where they run together late in a run, and each line is labelled at its end.
      { id: 'box', label: 'box', color: 'var(--chart-2)', front: true, end: this.trailMark.get(), points: box.pts, bounds: box.ext, rev },
      { id: 'cls', label: 'cls', color: 'var(--chart-5)', pattern: 'dash', end: this.trailMark.get(), points: cls.pts, bounds: cls.ext, rev },
      { id: 'dfl', label: 'dfl', color: 'var(--chart-4)', pattern: 'dot', end: this.trailMark.get(), points: dfl.pts, bounds: dfl.ext, rev },
    ];
  });
  readonly lrSeries = computed((): Series[] => {
    this.stepsRev();
    // Every k-th step (a schedule is smooth): at most about 1,200 points, whatever the run's length, and always the last
    // step, so the line and its resting legend end on the reading the run is at.
    const all = this.stepData.raw('lr').pts;
    const k = Math.max(1, Math.floor(all.length / 600));
    const points: Pt[] = [];
    for (let i = 0; i < all.length; i += k) points.push(all[i]!);
    if (all.length && points.at(-1) !== all.at(-1)) points.push(all.at(-1)!);
    return [{ id: 'lr', label: 'lr', color: 'var(--text-secondary)', end: this.endMark.get(), points }];
  });
  /** The fourth chart's view: the learning rate, or the throughput over the whole run (the tile keeps the last steps). */
  readonly lrView = signal<'lr' | 'ips'>('lr');
  /**
   * Throughput at every step, smoothed as the tile's sparkline is: the run's whole history, read by the cursor. All of
   * it, as the loss lines: the chart's downsampler keeps each bucket's extremes (a slowdown survives) and the last step.
   */
  readonly ipsSeries = computed((): Series[] => {
    const rev = this.stepsRev();
    const ips = this.stepData.smoothed('ips', 0.9);
    return [{ id: 'ips', label: 'img/s', color: 'var(--text-secondary)', end: this.endMark.get(), points: ips.pts, bounds: ips.ext, rev }];
  });
  /** The throughput of the last steps (the tile's sparkline, smoothed as the old chart drew it) and their average. */
  readonly ips = computed(() => {
    this.stepsRev();
    const pts = this.stepData.smoothed('ips', 0.9).pts;
    const raw = this.stepData.raw('ips').pts;
    const last = raw.slice(-IPS_WINDOW);
    const avg = last.length ? last.reduce((s, p) => s + p.y, 0) / last.length : null;
    return { trend: pts.slice(-IPS_WINDOW).map((p) => p.y), avg, n: last.length };
  });
  /** The total loss as the chart draws it (smoothed at the chosen EMA): the loss chart's header says it beside the raw step. */
  readonly smoothedLast = computed(() => {
    this.stepsRev();
    return this.smoothing.get() !== '0' ? this.stepData.smoothed('total', this.alpha()).pts.at(-1)?.y : undefined;
  });
  /** The learning rate's power of ten, stated once in its header: ticks "2" "4" with "× 10⁻³". */
  readonly lrExp = computed(() => {
    this.stepsRev();
    const m = this.stepData.lrMax();
    return m > 0 ? Math.floor(Math.log10(m)) : 0;
  });

  // ---- y domains ----
  /** The plot's height inside LineChart's margins, for the tick count: `height` less the band and the x ticks. */
  readonly plotPx = (height: number) => height - 46;
  /** Re-fit at most once per epoch while it trains (values never tween). */
  private epochKey = () => (this.live.get() ? Math.floor(this.observed.get()) : -1);
  /**
   * A 0–1 metric's domain: 0 to its top plus headroom; zoomed under 0.25 (the y caption says so). A fit to values that
   * are all zero is 0–1 for want of a top (`unitTop`), not a scale to hold: the first non-zero value re-fits at once,
   * also when it arrives within the epoch (issue #382).
   */
  unitFit(series: () => Series[], height: () => number) {
    const ext = computed(() => extentOf(series()));
    return held(
      () => `${height()}|${this.epochKey()}|${(ext.get()?.[1] ?? 0) > 0}`,
      () => {
        const e = ext.get();
        return e ? unitDomain(e[1], this.plotPx(height())) : null;
      },
    );
  }
  /** Safety status for plotting the active loss series on a log Y scale (handles <= 0 values cleanly). */
  readonly lossLogStatus = computed(() => {
    this.stepsRev();
    const a = this.alpha();
    const shown = (['total', 'box', 'cls', 'dfl'] as const).map((f) => this.stepData.smoothed(f, a).ext);
    const plotted = a > 0 ? [this.stepData.raw('total').ext, ...shown] : shown;
    const hasAny = joinExtents(shown, false) !== null;
    const hasPositive = joinExtents(shown, true) !== null;
    const nonPositive = countNonPositive(plotted);
    const wantLog = this.scale.get() === 'log';
    return {
      hasAny,
      hasPositive,
      nonPositive,
      fallbackLinear: wantLog && hasAny && !hasPositive,
      skippedNonPositive: wantLog && hasPositive && nonPositive > 0,
    };
  });

  lossFit(height: () => number) {
    const tracks = computed(() => {
      this.stepsRev();
      const a = this.alpha();
      return [...(a > 0 ? [this.stepData.raw('total').ext] : []), ...(['total', 'box', 'cls', 'dfl'] as const).map((f) => this.stepData.smoothed(f, a).ext)];
    });
    return held(
      () => `${this.scale.get()}|${this.lossLogStatus.get().fallbackLinear}|${this.smoothing.get()}|${height()}|${this.epochKey()}`,
      () => {
        // Gate log scale on the displayed series having positive values so a ghost raw point alone never forces a log
        // axis while all displayed smoothed curves are <= 0.
        const log = this.scale.get() === 'log' && !this.lossLogStatus.get().fallbackLinear;
        if (log) {
          const pos = joinExtents(tracks.get(), true);
          if (pos) return { domain: logDomain(pos[0], pos[1]), ticks: [], decimals: 0, log: true };
        }
        const e = joinExtents(tracks.get(), false);
        return e ? { ...niceDomain(e[0], e[1]), log: false } : null;
      },
    );
  }
  lrFit(height: () => number) {
    return held(
      () => `${height()}|${this.epochKey()}`,
      () => {
        const e = extentOf(this.lrSeries.get());
        return e ? niceDomain(e[0], e[1]) : null;
      },
    );
  }
  /** Throughput from 0, so a slowdown reads at its true size. */
  ipsFit(height: () => number) {
    return held(
      () => `ips|${height()}|${this.epochKey()}`,
      () => {
        const e = extentOf(this.ipsSeries.get());
        return e ? niceDomain(0, e[1]) : null;
      },
    );
  }
  // ---- the best epoch: a hollow ink ring on mAP50-95, the legend's "○ best 0.656 @ 41" ----
  readonly best = computed((): Marker[] => {
    const b = this.run().best;
    return b ? [{ x: b.epoch, y: b.map50_95, label: 'best', quiet: true, pulse: this.pulse.get(), focus: this.focus.get() }] : [];
  });
  /** At the decimals of the mAP chart's fitted top (lib/format `axisMetric`), as its series' values. */
  bestLegend(top?: number): string | undefined {
    const b = this.run().best;
    return b && b.map50_95 !== 0 ? `${axisMetric(b.map50_95, top)} @ ${b.epoch}` : undefined;
  }

  // ---- the words of the readings ----
  /** What the view holds of an ended run's results (issue #327, progress `heldResults`). */
  readonly results = computed(() => heldResults(this.server(), this.lastEpoch.get()?.epoch ?? 0, this.shortAnswers()));
  /**
   * It ended, and the server says it validated a later epoch than the last the view holds: the rest are on their way.
   * The per-epoch readings hold their place as loading, call nothing final, and say neither "Waiting for the first
   * validation…" nor that it never validated.
   */
  readonly pending = computed(() => !this.live.get() && this.results.get().pending);
  /** An ended snapshot confirms the last epoch held is the run's last validated one: its reading may say "final". */
  readonly final = computed(() => !this.live.get() && this.results.get().final);
  /** Its history came back short of what the run validated, for good (its metrics file missed epochs): the page's note. */
  readonly shortNote = computed(() => {
    const r = this.results.get();
    if (this.live.get() || !r.short) return null;
    const held = this.lastEpoch.get()?.epoch ?? 0;
    const v = r.validated!;
    return {
      title: held + 1 === v ? `Epoch ${v} is missing from its history` : `Epochs ${held + 1}–${v} are missing from its history`,
      why: `The run's record names ${v} validated epochs; its history ${held ? `ends at epoch ${held}` : 'has none'}. The readings show what was recorded.`,
    };
  });
  /**
   * It ended (stopped, failed) after training steps but before its first validation, as the server's snapshot says:
   * the per-epoch readings will never come, so they say so instead of "after the first epoch" and "Waiting for the
   * first validation…".
   */
  readonly unvalidated = computed(() => !this.live.get() && !this.lastEpoch.get() && this.results.get().validated === 0);
  readonly firstEpoch = computed(() => (this.unvalidated.get() ? 'not validated' : 'after the first epoch'));
  /** "Latest · ep 37", the per-epoch legends' note; per-step readings name the epoch the last step is in. */
  readonly latestEp = computed(() => (this.lastEpoch.get() ? `Latest · ep ${this.lastEpoch.get()!.epoch}` : undefined));
  readonly stepEpWord = computed(() => stepEpoch(this.run().status, this.observed.get(), this.lastEpoch.get()?.epoch ?? 0));
  readonly stepEp = computed(() => (this.lastStep.get() ? `Latest · ${this.stepEpWord.get()}` : undefined));
  /** Precision and recall are read at each epoch's best-F1 confidence (docs/API.md, Operating point): the latest one. */
  readonly prConf = computed(() => this.lastEpoch.get()?.val.best_f1_conf);
  /** The LR schedule the trainer resolved (`Run.resolved`), else what the recipe defaults to, in epochs. */
  readonly schedule = computed(() => {
    const r = this.run().resolved;
    if (!r) return 'warmup 3 epochs · linear';
    const e = warmupEpochs(r);
    return `${r.warmup_iters <= 0 ? 'no warmup' : `warmup ${e} ${e === 1 ? 'epoch' : 'epochs'}`} · ${r.cos_lr ? 'cosine' : 'linear'}`;
  });
  /** Nothing detected yet (a best mAP of exactly 0): said in words, never "best 0.000 at epoch 1" or "conf 1.00". */
  readonly noneYet = computed(() => (this.live.get() ? 'no detections yet' : 'no detections'));
}
