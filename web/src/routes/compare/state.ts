// Reactive state, polling, per-run metrics recovery, and chart derivation for Compare runs (`../RunCompare.tsx`).
// Call `compareState()` inside the component body so its resources, effects, and cleanup follow the component's life.
import { computed, effect, MediaQuery, onCleanup, signal, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { pollTogether, reuse } from '$lib/api/share';
import type { Epoch, Run, RunMetrics } from '$lib/api/types';
import { fitFloorNote, niceDomain, unitDomain, unitFitDomain, unitTop, UNIT_ZOOM_BELOW, UNIT_ZOOM_NOTE, withLabelRow } from '$lib/charts/domain';
import { nearest, type Pt } from '$lib/charts/downsample';
import type { Series } from '$lib/charts/LineChart';
import { axisReason, AXIS_REASON, runEnd, runLength, runX } from '$lib/charts/runAxis';
import { media } from '$lib/design/tokens';
import { router } from '$lib/router';
import { watched } from '$lib/shell/runList';
import { epochsDone } from '../run/readings';
import {
  COMPARE_POLL_MS,
  isActiveStatus,
  MAX,
  METRICS,
  RUN_SLOTS,
  SLOT_PATTERN,
  startFrom,
  type Axis,
  type MetricId,
  type YAxis,
} from './config';

export function compareState() {
  // Each run once: a link naming one twice (`?ids=a,a`) would key two rows alike.
  const ids = computed(() => [...new Set((router.route.query.get('ids') ?? '').split(',').filter(Boolean))].slice(0, MAX), {
    equals: (a, b) => a.length === b.length && a.every((v, i) => v === b[i]),
  });

  function setIds(next: string[]) {
    router.go(`/runs/compare${next.length ? `?ids=${next.join(',')}` : ''}`, { replace: true });
  }

  let borrowed = false;
  let confirming = false;
  let confirmSeq = 0;
  const confirmed = signal(0);
  const runs = new Resource((s) => hub.api.runs(s), {
    startFrom: () => {
      const src = watched.source;
      if (src) borrowed = true;
      return src;
    },
  });
  const datasets = new Resource((s) => hub.api.datasets(s));
  // Names for runs fine-tuned from a model ("Start from").
  const models = new Resource((s) => hub.api.models(s));
  const runById = computed(() => new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  const chosen = computed(() => {
    const map = runById.get();
    const out: Run[] = [];
    for (const id of ids.get()) {
      const r = map.get(id);
      if (r) out.push(r);
    }
    return out;
  });
  /** Where the compared runs are: each one's status and epoch ("run_a:running:3,run_b:completed:30"). */
  const moved = computed(() => chosen.get().map((r) => `${r.id}:${r.status}:${r.epoch}`).join(','));
  const runStamp = (r: Run) => `${r.status}:${r.epoch}`;

  /** Missing run IDs removed from `?ids=` after the run list loaded, announced to the user. */
  const dropped = signal<string[]>([]);
  /** Per-run metrics fetch errors so one failed read never hides healthy runs' curves. */
  const metricErrors = signal<Record<string, unknown>>({});
  const retryingMetric = signal<Record<string, boolean>>({});
  /** Set when a confirmed 404 run_not_found arrives while a runs list refresh is already in flight. */
  const needRunsRefresh = signal(false);
  const retryCtrls = new Map<string, AbortController>();
  const seriesCache = new Map<string, { epochs: readonly Epoch[]; axis: Axis; metric: MetricId; denom: number; points: Pt[]; rev: number }>();
  let seriesRev = 0;
  let metricsGen = 0;
  let alive = true;
  onCleanup(() => {
    alive = false;
    for (const c of retryCtrls.values()) c.abort();
    retryCtrls.clear();
  });

  // Reset per-comparison transient notices/errors and abort in-flight per-run retries when switching servers.
  let prevEpoch = untrack(() => hub.epoch);
  effect(() => {
    const epoch = hub.epoch;
    if (epoch === prevEpoch) return;
    prevEpoch = epoch;
    borrowed = false;
    confirming = false;
    confirmSeq++;
    for (const c of retryCtrls.values()) c.abort();
    retryCtrls.clear();
    seriesCache.clear();
    untrack(() => {
      dropped.set([]);
      metricErrors.set({});
      retryingMetric.set({});
      needRunsRefresh.set(false);
    });
  });

  // Reconcile `?ids=` against the loaded run list so deleted run IDs never hold slots or trap the page in an empty dead-end.
  // When the initial list came from `watched.source`, confirm missing IDs with a fresh read before rewriting `?ids=`.
  effect(() => {
    const list = runs.data;
    void confirmed.get();
    if (!list) return;
    const cur = ids.get();
    if (!cur.length) return;
    const known = new Set(list.map((r) => r.id));
    const missing = cur.filter((id) => !known.has(id));
    if (!missing.length) return;
    if (borrowed) {
      if (runs.refreshing || runs.error || runs.stale || confirming) return;
      confirming = true;
      const seq = ++confirmSeq;
      const epoch = untrack(() => hub.epoch);
      void runs.refresh().then(() => {
        if (seq !== confirmSeq) return;
        confirming = false;
        if (!alive || untrack(() => hub.epoch) !== epoch || runs.refreshing || runs.error || runs.stale) return;
        borrowed = false;
        confirmed.set((n) => n + 1);
      });
      return;
    }
    untrack(() => {
      if (router.route.path !== '/runs/compare') return;
      const keep = cur.filter((id) => known.has(id));
      dropped.set([...new Set([...dropped.peek(), ...missing])]);
      setIds(keep);
    });
  });

  // Drop stale per-run error/retry state and abort retries for runs removed from the comparison.
  effect(() => {
    const cur = new Set(ids.get());
    for (const [id, ctrl] of retryCtrls) {
      if (!cur.has(id)) {
        ctrl.abort();
        retryCtrls.delete(id);
      }
    }
    untrack(() => {
      const errs = metricErrors.peek();
      const errKeys = Object.keys(errs);
      if (errKeys.some((id) => !cur.has(id))) {
        const next: Record<string, unknown> = {};
        for (const id of errKeys) if (cur.has(id)) next[id] = errs[id];
        metricErrors.set(next);
      }
      const retrying = retryingMetric.peek();
      const retryKeys = Object.keys(retrying);
      if (retryKeys.some((id) => !cur.has(id))) {
        const next: Record<string, boolean> = {};
        for (const id of retryKeys) if (cur.has(id)) next[id] = retrying[id]!;
        retryingMetric.set(next);
      }
    });
  });

  /** Schedules a fresh runs list read after a confirmed 404 run_not_found, even if one is already in flight. */
  function refreshRunsAfter404() {
    if (!alive) return;
    if (runs.loading || runs.refreshing) needRunsRefresh.set(true);
    else void runs.refresh();
  }
  effect(() => {
    if (!needRunsRefresh.get() || hub.mode === 'connecting' || runs.loading || runs.refreshing) return;
    untrack(() => {
      needRunsRefresh.set(false);
      void runs.refresh();
    });
  });

  /**
   * Each metrics map's `ids` (joined) and per-run `status:epoch` stamp captured when that run's metrics were asked for
   * (`null` when asked before the runs list came).
   */
  const readAt = new WeakMap<object, { ids: string; byRun: Map<string, string | null> }>();
  const metrics = new Resource(async (s) => {
    const list = ids.get();
    const gen = ++metricsGen;
    for (const c of retryCtrls.values()) c.abort();
    retryCtrls.clear();
    untrack(() => retryingMetric.set({}));
    const snapByRun = untrack(() => (runs.data ? new Map(chosen.get().map((r) => [r.id, runStamp(r)])) : null));
    const prev = untrack(() => metrics.data);
    const prevByRun = prev ? readAt.get(prev)?.byRun : undefined;
    if (!list.length) {
      if (!s.aborted) untrack(() => metricErrors.set({}));
      const empty = new Map<string, RunMetrics>();
      readAt.set(empty, { ids: '', byRun: new Map() });
      return empty;
    }
    const settled = await Promise.allSettled(list.map((id) => hub.api.runMetrics(id, s)));
    if (s.aborted) throw new DOMException('Aborted', 'AbortError');
    const read = new Map<string, RunMetrics>();
    const byRun = new Map<string, string | null>();
    const errs: Record<string, unknown> = {};
    let firstErr: unknown = undefined;
    let failedCount = 0;
    let saw404 = false;
    for (let i = 0; i < list.length; i++) {
      const id = list[i]!;
      const res = settled[i]!;
      if (res.status === 'fulfilled') {
        const cached = prev?.get(id);
        read.set(id, cached ? reuse(cached, res.value) : res.value);
        byRun.set(id, snapByRun ? (snapByRun.get(id) ?? null) : null);
      } else {
        const err = res.reason;
        // A run deleted since the link was made has no metrics (and no chip): it leaves the others' alone.
        if (err instanceof ApiError && err.code === 'run_not_found') {
          if (snapByRun?.has(id)) {
            byRun.set(id, snapByRun.get(id)!);
            saw404 = true;
          }
          continue;
        }
        failedCount++;
        if (firstErr === undefined) firstErr = err;
        errs[id] = err;
        const cached = prev?.get(id);
        if (cached) {
          read.set(id, cached);
          byRun.set(id, prevByRun?.get(id) ?? null);
        }
      }
    }
    if (gen !== metricsGen) throw new DOMException('Aborted', 'AbortError');
    if (saw404) untrack(() => refreshRunsAfter404());
    // If every non-deleted run failed and no cached curve can be shown, hand off cleanly to `metrics.error` / ErrorState.
    if (failedCount > 0 && read.size === 0) {
      untrack(() => metricErrors.set({}));
      throw firstErr;
    }
    untrack(() => metricErrors.set(errs));
    readAt.set(read, { ids: list.join(','), byRun });
    return read;
  });

  const failedRuns = computed(() => chosen.get().filter((r) => metricErrors.get()[r.id] !== undefined));
  const hasMetricErrors = computed(() => failedRuns.get().length > 0);
  const failedWithCache = computed(() => failedRuns.get().some((r) => !!metrics.data?.has(r.id)));

  /** Fetches a single run's metrics without disturbing other runs' curves or read stamps. */
  async function fetchRunMetrics(id: string) {
    if (!alive || retryingMetric.peek()[id] || metrics.loading || metrics.refreshing) return;
    const epoch = hub.epoch;
    const gen = metricsGen;
    const stamp = untrack(() => {
      const r = runs.data?.find((x) => x.id === id);
      return r ? runStamp(r) : null;
    });
    retryCtrls.get(id)?.abort();
    const ctrl = new AbortController();
    retryCtrls.set(id, ctrl);
    retryingMetric.set({ ...retryingMetric.peek(), [id]: true });
    try {
      const m = await hub.api.runMetrics(id, ctrl.signal);
      if (!alive || ctrl.signal.aborted || hub.epoch !== epoch || gen !== metricsGen || !ids.peek().includes(id)) return;
      const prev = metrics.data;
      const prevByRun = prev ? readAt.get(prev)?.byRun : undefined;
      const next = new Map(prev ?? []);
      const cached = prev?.get(id);
      next.set(id, cached ? reuse(cached, m) : m);
      const nextByRun = new Map(prevByRun ?? []);
      nextByRun.set(id, stamp);
      const curErrs = metricErrors.peek();
      const hadErr = id in curErrs;
      const nextErrs = hadErr ? { ...curErrs } : curErrs;
      if (hadErr) delete nextErrs[id];
      const curIds = ids.peek().join(',');
      readAt.set(next, { ids: curIds, byRun: nextByRun });
      if (hadErr) metricErrors.set(nextErrs);
      metrics.error = undefined;
      if (Object.keys(nextErrs).length === 0) metrics.stale = undefined;
      metrics.set(next);
    } catch (err) {
      if (!alive || ctrl.signal.aborted || hub.epoch !== epoch || gen !== metricsGen || !ids.peek().includes(id)) return;
      if (err instanceof ApiError && err.code === 'run_not_found') {
        if (stamp !== null && metrics.data) readAt.get(metrics.data)?.byRun.set(id, stamp);
        const curErrs = metricErrors.peek();
        if (id in curErrs) {
          const nextErrs = { ...curErrs };
          delete nextErrs[id];
          metricErrors.set(nextErrs);
        }
        refreshRunsAfter404();
        return;
      }
      metricErrors.set({ ...metricErrors.peek(), [id]: err });
    } finally {
      if (retryCtrls.get(id) === ctrl) {
        retryCtrls.delete(id);
        if (alive) {
          const nextRetrying = { ...retryingMetric.peek() };
          delete nextRetrying[id];
          retryingMetric.set(nextRetrying);
        }
      }
    }
  }

  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = () => phoneQuery.current;
  const cursor = signal<number | null>(null);

  /**
   * Whether `r`'s metrics hold every epoch its list entry may have finished. A failed run's `epoch` is the one it
   * failed in, or its last finished one when it failed before the next epoch's first step (docs/API.md `Run.epoch`):
   * the list cannot tell them apart, so a pre-list read (`stamp === null`) counts as caught up only through `epoch`,
   * while a read started after observing `failed:<epoch>` counts as synced once it finishes.
   */
  const runCaughtUp = (r: Run, read: Map<string, RunMetrics>) =>
    (read.get(r.id)?.epochs.at(-1)?.epoch ?? 0) >= (r.status === 'failed' ? r.epoch : epochsDone(r.status, r.epoch));
  const runSynced = (r: Run, read: Map<string, RunMetrics>, byRun: Map<string, string | null> | undefined) => {
    if (metricErrors.get()[r.id] !== undefined || isActiveStatus(r.status)) return false;
    const stamp = byRun?.get(r.id);
    if (stamp === runStamp(r)) return true;
    return stamp === null && runCaughtUp(r, read);
  };

  // Poll runs every COMPARE_POLL_MS only while at least one compared run is active or not yet synced;
  // once every compared run is terminal and synced, stop the timer to avoid idle background polling.
  const needsPoll = computed(() => {
    if (!runs.data) return false;
    if (runs.stale !== undefined) return true;
    const cur = chosen.get();
    if (!cur.length) return false;
    if (cur.some((r) => isActiveStatus(r.status))) return true;
    if (metrics.loading || metrics.error !== undefined) return false;
    if (metrics.stale !== undefined || hasMetricErrors.get()) return true;
    const read = metrics.data;
    const asked = read ? readAt.get(read) : undefined;
    if (!read || !asked || asked.ids !== ids.get().join(',')) return false;
    return !cur.every((r) => runSynced(r, read, asked.byRun));
  });

  // A compared run changes under the page: it trains, ends, is resumed or deleted in another tab. While any compared run
  // is active (or metrics are catching up), runs are read again every COMPARE_POLL_MS; once all runs are terminal and
  // synced, polling stops and a single refresh runs if the user returns to this tab.
  effect(() => {
    if (!needsPoll.get()) return;
    return pollTogether(COMPARE_POLL_MS, () => {
      if (document.visibilityState === 'visible' && hub.mode !== 'connecting' && !untrack(() => runs.loading || runs.refreshing)) void runs.refresh();
    });
  });
  effect(() => {
    const onVisible = () => {
      if (document.visibilityState !== 'visible') return;
      if (hub.mode === 'connecting' || !ids.peek().length || untrack(() => !runs.data || runs.loading || runs.refreshing || needsPoll.peek())) return;
      void runs.refresh();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  });

  // A failed refresh or per-run metrics read is tried again COMPARE_POLL_MS after it failed (its own effect: the runs
  // polled meanwhile never put it off).
  effect(() => {
    const stale = metrics.stale !== undefined;
    const failed = Object.keys(metricErrors.get());
    if ((!stale && !failed.length) || hub.mode === 'connecting' || metrics.loading || metrics.refreshing || metrics.error !== undefined) return;
    const t = setTimeout(() => {
      if (stale) void metrics.refresh();
      else for (const id of Object.keys(metricErrors.peek())) void fetchRunMetrics(id);
    }, COMPARE_POLL_MS);
    return () => clearTimeout(t);
  });
  effect(() => {
    const read = metrics.data;
    if (!read || !runs.data || metrics.loading || metrics.refreshing || metrics.stale !== undefined || metrics.error !== undefined) return;
    const asked = readAt.get(read);
    // The link names other runs: the metrics follow `ids` by themselves.
    if (!asked || asked.ids !== ids.get().join(',')) return;
    // Track per-run movement independently of any other run's failure, so healthy runs keep advancing while a broken one retries.
    const errs = metricErrors.get();
    const cur = chosen.get();
    void moved.get();
    const behind: string[] = [];
    for (const r of cur) {
      if (errs[r.id] !== undefined) continue;
      const stamp = asked.byRun.get(r.id);
      const nowStamp = runStamp(r);
      if (stamp === nowStamp) continue;
      if (stamp === null && runCaughtUp(r, read)) continue;
      behind.push(r.id);
    }
    if (!behind.length) return;
    untrack(() => {
      if (!Object.keys(errs).length && behind.length === cur.length) void metrics.refresh();
      else for (const id of behind) void fetchRunMetrics(id);
    });
  });

  // Runs planned for very different lengths (30 vs 150 epochs), or one run that trained over twice as long as every
  // other, would crush the short ones into the start of an epoch axis: then the axis defaults to "By progress" (each run
  // over its planned epochs, ../lib/charts/runAxis), and the header says why.
  const xChoice = signal<Axis | null>(null);
  const reason = computed(() => axisReason(chosen.get()));
  const x = computed<Axis>(() => xChoice.get() ?? (reason.get() ? 'progress' : 'epoch'));
  const xAuto = computed(() => xChoice.get() === null && x.get() === 'progress');
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

  // One chart, one metric at a time: four small charts repeated the same legend four times.
  const metricId = signal<MetricId>('m95');
  const metricDef = computed(() => METRICS.find((m) => m.id === metricId.get())!);
  const emptyEpochs: readonly Epoch[] = [];
  const samePts = (a: readonly Pt[], b: readonly Pt[]) =>
    a.length === b.length && a.every((p, i) => Object.is(p.x, b[i]!.x) && Object.is(p.y, b[i]!.y));
  const series = computed(() => {
    const def = metricDef.get();
    const f = def.f;
    const mId = def.id;
    const axis = x.get();
    const list = chosen.get();
    const data = metrics.data;
    const activeIds = new Set<string>();
    const out = list.map((r): Series => {
      activeIds.add(r.id);
      const eps = data?.get(r.id)?.epochs ?? emptyEpochs;
      const denom = axis === 'progress' ? runLength(r) : 0;
      let cached = seriesCache.get(r.id);
      if (!cached || cached.epochs !== eps || cached.axis !== axis || cached.metric !== mId || cached.denom !== denom) {
        const points = eps.map((e) => ({ x: runX(e.epoch, r, axis), y: f(e) }));
        if (cached && samePts(cached.points, points)) {
          cached = { epochs: eps, axis, metric: mId, denom, points: cached.points, rev: cached.rev };
        } else {
          cached = { epochs: eps, axis, metric: mId, denom, points, rev: ++seriesRev };
        }
        seriesCache.set(r.id, cached);
      }
      return {
        id: r.id,
        label: r.name,
        color: color(r.id),
        pattern: SLOT_PATTERN[slotOf(r.id)],
        // Each line ends in its run's status mark (live, stopped, failed), so a line that stops short says why.
        end: runEnd(r.status) ?? undefined,
        points: cached.points,
        rev: cached.rev,
      };
    });
    for (const id of seriesCache.keys()) if (!activeIds.has(id)) seriesCache.delete(id);
    return out;
  });
  const seriesById = computed(() => new Map<string, Series>(series.get().map((s) => [s.id, s])));
  const extent = computed(
    (): [number, number] | null => {
      let lo = Infinity;
      let hi = -Infinity;
      for (const s of series.get())
        for (const p of s.points) {
          if (!Number.isFinite(p.y)) continue;
          if (p.y < lo) lo = p.y;
          if (p.y > hi) hi = p.y;
        }
      return Number.isFinite(lo) ? [lo, hi] : null;
    },
    { equals: (a, b) => a === b || (!!a && !!b && a[0] === b[0] && a[1] === b[1]) },
  );
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
  const ys = computed(() => {
    const out: number[] = [];
    for (const s of series.get()) for (const p of s.points) out.push(p.y);
    return out;
  });
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
    const pts = seriesById.get().get(id)?.points ?? [];
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
  // Precision and recall are read at each epoch's own best-F1 confidence (docs/API.md, Operating point): the readout says so.
  const readoutAt = computed(() => {
    const c = cursor.get();
    const m = metricId.get();
    return (c === null ? latestNote.get() : x.get() === 'epoch' ? `Epoch ${Math.round(c)}` : `${Math.round(c)}% through`) + (m === 'prec' || m === 'rec' ? ' · at best-F1 conf' : '');
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
  // A baseline, pinned, or hovered run removed from the comparison is forgotten, not latent: adding that run again
  // later must not quietly restore its old emphasis or baseline state.
  effect(() => {
    const cur = ids.get();
    untrack(() => {
      const b = baselineId.get();
      if (b !== null && !cur.includes(b)) baselineId.set(null);
      const p = pinned.get();
      if (p !== null && !cur.includes(p)) pinned.set(null);
      const h = hovered.get();
      if (h !== null && !cur.includes(h)) hovered.set(null);
    });
  });

  const candidates = computed(() => (runs.data ?? []).filter((r) => !ids.get().includes(r.id) && r.status !== 'queued'));

  return {
    ids,
    setIds,
    runs,
    datasets,
    metrics,
    chosen,
    dropped,
    metricErrors,
    retryingMetric,
    failedRuns,
    failedWithCache,
    fetchRunMetrics,
    narrow,
    cursor,
    hint,
    x,
    progressWhy,
    xQual,
    slotOf,
    fill,
    metricId,
    metricDef,
    series,
    fit,
    floorNote,
    yCaption,
    yAxis,
    setY,
    height,
    valueOf,
    hovered,
    pinned,
    highlight,
    readoutAt,
    xDomain,
    sameData,
    startOf,
    setX,
    xBind,
    showSame,
    managing,
    allRuns,
    otherAt,
    baselineId,
    candidates,
  };
}
