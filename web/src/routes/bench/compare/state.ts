// The state of BenchCompare (../../BenchCompare.tsx), apart from its markup so the screen stays under 1000 lines: the
// entries compared (from the URL), their results, their slots (hue and shape), the legend's highlight and every
// derived table and series. Call it in the component's body: its effects and resources follow the component's life.

import { computed, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import { LiveBench } from '$lib/api/liveBench';
import { Resource } from '$lib/api/resource';
import { AXES, AXIS_NAME } from '$lib/bench/rules';
import type { XYSeries } from '$lib/charts/xy';
import { metric, num } from '$lib/format';
import { router } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import { assignSlots, bestAsPrinted, classRows, compareShown, conditionRows, dotDomain, labelTradeoff, SLOT_PATTERN, SLOTS } from '../charts';
import { COMPARE_MAX, compareIds, finished, intervalDomain, stale } from '../leaderboard';
import { BenchResults } from '../results';
import { cannotBench } from '../support';
import { compareCaveats, intervalsVerdict, scoreText } from '../words.fw';
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

export function compareState() {
  const id = computed(() => router.route.params.id ?? '');
  const ids = computed(() => compareIds(router.route.query));
  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const live = new LiveBench(() => id.get() || null);
  const bench = () => live.bench;
  const unsupported = computed(() => cannotBench(system.data, live.error));
  const picked = computed(() => compareShown(ids.get().slice(0, COMPARE_MAX), live.entries));
  const shown = computed(() => picked.get().shown);
  const results = new BenchResults(
    () => id.get() || null,
    () => shown.get(),
  );

  effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench()?.name ?? id.get(), path: `/benchmarks/${id.get()}` }, { label: 'Compare' });
  });

  function setIds(next: string[]) {
    router.go(`/benchmarks/${encodeURIComponent(id.peek())}/compare${next.length ? `?entries=${next.map(encodeURIComponent).join(',')}` : ''}`, { replace: true });
  }
  const candidates = computed(() => {
    const now = ids.get();
    return live.entries.filter((e) => finished(e) && !now.includes(e.id));
  });

  // Colours follow the entry, not its place: removing one never repaints the others (RunCompare). Kept from one shown
  // list to the next in a computed (legacy: an effect after the render, so a change painted twice).
  let lastSlots: Record<string, number> = {};
  const slots = computed(() => (lastSlots = assignSlots(shown.get().map((e) => e.id), lastSlots)));
  const slotOf = (eid: string) => slots.get()[eid] ?? Math.max(0, shown.get().findIndex((e) => e.id === eid));
  const color = (eid: string) => `var(--chart-${SLOTS[slotOf(eid)] ?? 1})`;
  const fill = (eid: string) => `var(--chart-fill-${SLOTS[slotOf(eid)] ?? 1})`;
  const keys = computed<DotKey[]>(() => shown.get().map((e) => ({ id: e.id, name: e.name, color: color(e.id), shape: SHAPES[slotOf(e.id)] ?? 'circle' })));

  // The legend: hover or focus a chip to bring its entry forward everywhere; a click keeps it there.
  const hovered = signal<string | null>(null);
  const pinned = signal<string | null>(null);
  const highlight = computed(() => [hovered.get(), pinned.get()].find((x) => x !== null && shown.get().some((e) => e.id === x)) ?? null);

  const res = computed(() => shown.get().map((e) => results.get(e.id)));
  const loaded = computed(() => shown.get().length > 0 && res.get().every((r) => !!r));
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
  /** Entries graded on an earlier version of the benchmark: never the best of a row, never in the intervals' verdict. */
  const old = computed(() => {
    const b = bench();
    return shown.get().map((e) => !!b && stale(e, b.fingerprint));
  });
  const best = (key: string, values: (number | null)[], d: number) => {
    const o = old.get();
    return bestAsPrinted(
      values.map((v, i) => (o[i] ? null : v)),
      d,
      key === 'accuracy' ? shown.get().map((e, i) => (o[i] ? null : e.ci95)) : null,
    );
  };
  const makeBaseline = (eid: string) => setIds([eid, ...ids.peek().filter((x) => x !== eid)]);

  // ---- mAP intervals ----
  const ivDomain = computed(() => intervalDomain(shown.get().flatMap((e) => (e.ci95 ? [e.ci95] : []))));
  const verdict = computed(() => {
    const o = old.get();
    return intervalsVerdict(
      shown
        .get()
        .filter((_, i) => !o[i])
        .map((e) => ({ name: e.name, map: e.map50_95 ?? 0, ci: e.ci95 })),
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
          label: e.name,
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
      return [{ id: e.id, label: e.name, color: color(e.id), pattern: SLOT_PATTERN[slotOf(e.id)], points: pts, dots: pts.length === 1, line: pts.length > 1 }];
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
    ids,
    live,
    bench,
    unsupported,
    picked,
    shown,
    results,
    setIds,
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
    failedResult,
    caveats,
    nameOf,
    rows,
    old,
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
