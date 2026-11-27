// The state of BenchCompare (../../BenchCompare.tsx), apart from its markup so the screen stays under 1000 lines: the
// entries compared (from the URL), their results, their slots (hue and shape), the legend's highlight and every
// derived table and series. Call it in the component's body: its effects and resources follow the component's life.

import { computed, effect, onCleanup, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { LiveBench } from '$lib/api/liveBench';
import { Resource } from '$lib/api/resource';
import type { BenchEntry } from '$lib/api/types';
import { AXES, AXIS_NAME, isEntryActive } from '$lib/bench/rules';
import type { XYSeries } from '$lib/charts/xy';
import { metric, num } from '$lib/format';
import { router } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import { assignSlots, bestAsPrinted, classRows, compareShown, conditionRows, dotDomain, labelTradeoff, SLOT_PATTERN, SLOTS } from '../charts';
import { COMPARE_MAX, compareIds, finished, intervalDomain, stale } from '../leaderboard';
import { BenchResults } from '../results';
import { benchUnavailable } from '../support';
import { compareCaveats, intervalsVerdict, scoreText } from '../words';
import type { DotKey } from './DotRows';

const SHAPES = ['circle', 'square', 'diamond', 'triangle'] as const;
export const CLS_SHOWN = 12;

/** A row of the side-by-side table: the score or one axis, each entry's value. */
export interface Row {
  key: string;
  label: string;
  values: (number | null)[];
  fmt: (v: number) => string;
  d: number;
}

/**
 * Distinct, human-readable labels for compared entries (accessible names, chart readouts, phone baseline picker).
 * Unique names stay as-is; colliding names add "earlier version" and/or score, falling back to entry id if still identical.
 */
export function entryLabels(shown: readonly BenchEntry[], old: readonly boolean[]): Map<string, string> {
  const nameCounts = new Map<string, number>();
  for (const e of shown) nameCounts.set(e.name, (nameCounts.get(e.name) ?? 0) + 1);
  const draft = shown.map((e, i) => {
    if ((nameCounts.get(e.name) ?? 0) <= 1) return { id: e.id, label: e.name };
    const parts = [old[i] ? 'earlier version' : null, e.score ? `score ${scoreText(e.score.total)}` : null].filter(Boolean);
    return { id: e.id, label: parts.length ? `${e.name} · ${parts.join(' · ')}` : `${e.name} · ${e.id}` };
  });
  const labelCounts = new Map<string, number>();
  for (const d of draft) labelCounts.set(d.label, (labelCounts.get(d.label) ?? 0) + 1);
  const out = new Map<string, string>();
  for (const d of draft) out.set(d.id, (labelCounts.get(d.label) ?? 0) > 1 ? `${d.label} (${d.id})` : d.label);
  return out;
}

export function compareState() {
  const id = computed(() => router.route.params.id ?? '');
  const ids = computed(() => compareIds(router.route.query));
  const system = new Resource((s) => hub.api.system(s));
  const noBench = computed(() => benchUnavailable(system.data));
  const catalog = new Resource((s) => (noBench.get() ? Promise.resolve(undefined) : hub.api.benchCatalog(s)));
  const live = new LiveBench(() => (noBench.get() ? null : id.get() || null));
  const bench = () => live.bench;
  const unsupported = computed(() => benchUnavailable(system.data, live.error));
  const picked = computed(() => {
    const all = compareShown(ids.get(), live.entries);
    return { shown: all.shown.slice(0, COMPARE_MAX), missing: all.missing };
  });
  const shown = computed(() => picked.get().shown);
  /** In-progress entries named in `?entries=` that have room to join when SSE finishes them (capped at COMPARE_MAX). */
  const pending = computed(() => {
    const room = Math.max(0, COMPARE_MAX - shown.get().length);
    if (!room) return [];
    const byId = new Map(live.entries.map((e) => [e.id, e]));
    const out: BenchEntry[] = [];
    for (const eid of picked.get().missing) {
      const e = byId.get(eid);
      if (e && isEntryActive(e)) {
        out.push(e);
        if (out.length >= room) break;
      }
    }
    return out;
  });
  const canAdd = computed(() => shown.get().length + pending.get().length < COMPARE_MAX);
  /** Unavailable entry IDs pruned from `?entries=` after the benchmark's entries loaded, scoped to the comparison & history entry that produced them. */
  const dropped = signal<string[]>([]);
  const missingCount = computed(() => new Set([...dropped.get(), ...picked.get().missing]).size);
  let prevCtx = untrack(() => `${hub.epoch}:${id.get()}`);
  let lastIdsKey = untrack(() => ids.get().join(','));
  let prunedKey: string | null = null;
  onCleanup(
    router.onEntry(() => {
      prunedKey = null;
      untrack(() => dropped.set([]));
    }),
  );
  effect(() => {
    const ctx = `${hub.epoch}:${id.get()}`;
    const key = ids.get().join(',');
    if (ctx !== prevCtx) {
      prevCtx = ctx;
      lastIdsKey = key;
      prunedKey = null;
      untrack(() => dropped.set([]));
      return;
    }
    if (key !== lastIdsKey) {
      lastIdsKey = key;
      if (key === prunedKey) {
        prunedKey = null;
      } else {
        prunedKey = null;
        untrack(() => dropped.set([]));
      }
    }
  });
  effect(() => {
    const b = bench();
    if (live.loading || live.error || !b || b.id !== id.get()) return;
    const missing = picked.get().missing;
    if (!missing.length) return;
    const activeIds = new Set(live.entries.filter(isEntryActive).map((e) => e.id));
    const unavailable = missing.filter((eid) => !activeIds.has(eid));
    if (!unavailable.length) return;
    const unavailableSet = new Set(unavailable);
    untrack(() => {
      if (router.route.name !== 'bench-compare' || router.location.name !== 'bench-compare' || router.location.params.id !== id.peek()) return;
      const keep = ids.peek().filter((eid) => !unavailableSet.has(eid));
      prunedKey = keep.join(',');
      dropped.set([...new Set([...dropped.peek(), ...unavailable])]);
      setIds(keep);
    });
  });
  const results = new BenchResults(
    () => id.get() || null,
    () => shown.get(),
  );

  effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench()?.name ?? id.get(), path: `/benchmarks/${id.get()}` }, { label: 'Compare' });
  });

  function setIds(next: string[]) {
    if (router.location.name !== 'bench-compare' || router.location.params.id !== id.peek()) return;
    router.go(`/benchmarks/${encodeURIComponent(id.peek())}/compare${next.length ? `?entries=${next.map(encodeURIComponent).join(',')}` : ''}`, { replace: true });
  }
  const candidates = computed(() => {
    const now = new Set(shown.get().map((e) => e.id));
    return live.entries.filter((e) => finished(e) && !now.has(e.id));
  });

  /** Entries graded on an earlier version of the benchmark: never the best of a row, never in the intervals' verdict. */
  const old = computed(() => {
    const b = bench();
    return shown.get().map((e) => !!b && stale(e, b.fingerprint));
  });
  const labels = computed(() => entryLabels(shown.get(), old.get()));
  const labelOf = (e: BenchEntry) => labels.get().get(e.id) ?? e.name;

  // Colours follow the entry, not its place: removing one never repaints the others (RunCompare). Kept from one shown
  // list to the next in a computed (legacy: an effect after the render, so a change painted twice).
  let lastSlots: Record<string, number> = {};
  const slots = computed(() => (lastSlots = assignSlots(shown.get().map((e) => e.id), lastSlots)));
  const slotOf = (eid: string) => slots.get()[eid] ?? Math.max(0, shown.get().findIndex((e) => e.id === eid));
  const color = (eid: string) => `var(--chart-${SLOTS[slotOf(eid)] ?? 1})`;
  const fill = (eid: string) => `var(--chart-fill-${SLOTS[slotOf(eid)] ?? 1})`;
  const keys = computed<DotKey[]>(() => shown.get().map((e) => ({ id: e.id, name: labelOf(e), color: color(e.id), shape: SHAPES[slotOf(e.id)] ?? 'circle' })));

  // The legend: hover or focus a chip to bring its entry forward everywhere; a click keeps it there.
  const hovered = signal<string | null>(null);
  const pinned = signal<string | null>(null);
  effect(() => {
    const cur = shown.get();
    const p = pinned.peek();
    if (p !== null && !cur.some((e) => e.id === p)) pinned.set(null);
    const h = hovered.peek();
    if (h !== null && !cur.some((e) => e.id === h)) hovered.set(null);
  });
  const highlight = computed(() => [hovered.get(), pinned.get()].find((x) => x !== null && shown.get().some((e) => e.id === x)) ?? null);

  const res = computed(() => shown.get().map((e) => results.get(e.id)));
  const loaded = computed(() => shown.get().length > 0 && res.get().every((r) => !!r));
  /** True when at least two finished entries are compared (boolean memo keeps header Menu & SideBySide mounted across entry adds/polls). */
  const canCompare = computed(() => shown.get().length >= 2);
  /** True once at least two entries are shown and all detailed results + condition names have settled. */
  const exportReady = computed(() => !!bench() && canCompare.get() && loaded.get() && !catalog.loading);
  /** An entry whose result could not be loaded: said with a retry, never skeletons forever. */
  const failedResult = computed(() =>
    shown
      .get()
      .map((e) => ({ e, err: results.error(e.id) }))
      .find((x) => x.err !== undefined),
  );
  const caveats = computed(() => {
    const b = bench();
    return b ? compareCaveats(shown.get(), b) : [];
  });
  const nameOf = (cid: string) => catalog.data?.conditions.find((c) => c.id === cid)?.name ?? cid;

  // ---- side by side: the score and its axes, against the baseline (the first) ----
  const rows = computed<Row[]>(() => {
    const s = shown.get();
    return [
      { key: 'score', label: 'Score', values: s.map((e) => e.score!.total), fmt: (v: number) => scoreText(v), d: 1 },
      ...AXES.map((a, k) => ({
        key: a,
        label: a === 'accuracy' ? 'Accuracy · mAP50-95' : AXIS_NAME[a],
        values: s.map((e) => e.score!.axes[k]?.value ?? null),
        fmt: (v: number) => (a === 'accuracy' ? metric(v) : num(v, 2)),
        d: a === 'accuracy' ? 3 : 2,
      })),
    ];
  });
  const best = (key: string, values: (number | null)[], d: number) => {
    const o = old.get();
    return bestAsPrinted(
      values.map((v, i) => (o[i] ? null : v)),
      d,
      key === 'accuracy' ? shown.get().map((e, i) => (o[i] ? null : e.ci95)) : null,
    );
  };
  const pendingIds = () => pending.peek().map((e) => e.id);
  const makeBaseline = (eid: string) => {
    const current = shown.peek().map((e) => e.id);
    setIds([eid, ...current.filter((x) => x !== eid), ...pendingIds()]);
  };
  const rotateBaseline = () => {
    const current = shown.peek().map((e) => e.id);
    if (current.length < 2) return;
    setIds([...current.slice(1), current[0]!, ...pendingIds()]);
  };
  const appendEntry = (eid: string): boolean => {
    const current = shown.peek().map((e) => e.id);
    const active = pendingIds();
    if (current.length + active.length >= COMPARE_MAX || current.includes(eid) || active.includes(eid)) return false;
    setIds([...current, eid, ...active]);
    return true;
  };
  const dropShown = (eid: string) => {
    setIds([...shown.peek().map((e) => e.id).filter((x) => x !== eid), ...pendingIds()]);
  };

  // ---- mAP intervals ----
  const ivDomain = computed(() => intervalDomain(shown.get().flatMap((e) => (e.ci95 ? [e.ci95] : []))));
  const verdict = computed(() => {
    const o = old.get();
    return intervalsVerdict(
      shown
        .get()
        .filter((_, i) => !o[i])
        .map((e) => ({ name: labelOf(e), map: e.map50_95 ?? 0, ci: e.ci95 })),
    );
  });

  // ---- robustness per condition ----
  const condRows = computed(() => {
    const b = bench();
    return b && loaded.get() ? conditionRows(b.plan.conditions.map((c) => c.condition), res.get()) : [];
  });
  const condDomain = computed(() => dotDomain(condRows.get().flatMap((r) => r.values)));
  const noRobust = computed(() => shown.get().filter((_, i) => !res.get()[i]?.robustness));

  // ---- calibration overlaid ----
  const calSeries = computed<XYSeries[]>(() =>
    shown.get().flatMap((e, i) => {
      const c = res.get()[i]?.calibration;
      if (!c) return [];
      const n = c.bins.reduce((s, b) => s + b.count, 0);
      return [
        {
          id: e.id,
          label: labelOf(e),
          color: color(e.id),
          pattern: SLOT_PATTERN[slotOf(e.id)],
          points: c.bins.filter((b) => b.count > 0).map((b) => ({ x: b.confidence!, y: b.precision!, r: 3.5, hollow: b.count < 0.05 * n })),
        },
      ];
    }),
  );
  const noCal = computed(() => shown.get().filter((_, i) => res.get()[i] && !res.get()[i]!.calibration));
  const calCursor = signal<number | null>(null);

  // ---- self-labeling overlaid: precision of the labels kept against the share of objects they find ----
  const slSeries = computed<XYSeries[]>(() =>
    shown.get().flatMap((e, i) => {
      const s = res.get()[i]?.self_labeling;
      if (!s) return [];
      const pts = labelTradeoff(s.curve);
      return [{ id: e.id, label: labelOf(e), color: color(e.id), pattern: SLOT_PATTERN[slotOf(e.id)], points: pts, dots: pts.length === 1, line: pts.length > 1 }];
    }),
  );
  const slMarks = computed(() =>
    shown.get().flatMap((e, i) => {
      const s = res.get()[i]?.self_labeling;
      return s && s.recall > 0 ? [{ x: s.recall, y: s.precision, label: '' }] : [];
    }),
  );

  // ---- per class ----
  const clsRows = computed(() => {
    const b = bench();
    return b && loaded.get() ? classRows(b.classes, b.class_boxes, res.get()) : [];
  });
  const allClasses = signal(false);
  const clsShown = computed(() => (allClasses.get() ? clsRows.get() : clsRows.get().slice(0, CLS_SHOWN)));
  const clsDomain = computed(() => dotDomain(clsRows.get().flatMap((r) => r.values)));

  return {
    id,
    live,
    bench,
    unsupported,
    picked,
    shown,
    pending,
    canAdd,
    dropped,
    missingCount,
    results,
    setIds,
    appendEntry,
    dropShown,
    rotateBaseline,
    candidates,
    slotOf,
    color,
    fill,
    keys,
    hovered,
    pinned,
    highlight,
    res,
    loaded,
    canCompare,
    exportReady,
    failedResult,
    caveats,
    nameOf,
    rows,
    old,
    labelOf,
    best,
    makeBaseline,
    ivDomain,
    verdict,
    condRows,
    condDomain,
    noRobust,
    calSeries,
    noCal,
    calCursor,
    slSeries,
    slMarks,
    clsRows,
    allClasses,
    clsShown,
    clsDomain,
  };
}
