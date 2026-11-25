/**
 * Pure CSV / JSON payload builders for the Benchmarks list, a single benchmark's leaderboard, and BenchCompare.
 */
import type { Bench, BenchEntry, BenchResult, LabelAgent } from '$lib/api/types';
import { AXES } from '$lib/bench/rules';
import type { CsvCell } from '$lib/tableExport';
import { diffAsPrinted } from './charts';
import { ranks, stale } from './leaderboard';
import type { Row } from './compare/state';
import { contestantFacts, sourceKind } from './words';

export interface TableExportPayload {
  filenameBase: string;
  headers: string[];
  rows: CsvCell[][];
  json: unknown;
}

export function benchmarksExport(list: readonly Bench[]): TableExportPayload {
  const headers = [
    'id',
    'name',
    'source',
    'source_kind',
    'images',
    'classes',
    'conditions',
    'entries',
    'active',
    'leader_name',
    'leader_score',
    'leader_partial',
    'created',
  ];
  const items = list.map((b) => ({
    id: b.id,
    name: b.name,
    source: b.source_name,
    source_kind: sourceKind(b.source),
    images: b.images,
    classes: b.classes.length,
    conditions: b.plan.conditions.length,
    entries: b.entries,
    active: b.active,
    leader_name: b.best?.name ?? null,
    leader_score: b.best?.score ?? null,
    leader_partial: b.best?.partial ?? null,
    created: b.created,
  }));
  const rows: CsvCell[][] = items.map((r) => [
    r.id,
    r.name,
    r.source,
    r.source_kind,
    r.images,
    r.classes,
    r.conditions,
    r.entries,
    r.active,
    r.leader_name,
    r.leader_score,
    r.leader_partial,
    r.created,
  ]);
  return {
    filenameBase: 'folo-benchmarks',
    headers,
    rows,
    json: { count: items.length, benchmarks: items },
  };
}

export function leaderboardExport(bench: Bench, entries: readonly BenchEntry[], agents?: readonly LabelAgent[]): TableExportPayload {
  const current = entries.filter((e) => !stale(e, bench.fingerprint));
  const rankMap = ranks(current);
  const headers = [
    'rank',
    'id',
    'name',
    'contestant',
    'status',
    'score',
    'partial',
    'earlier_version',
    ...AXES,
    'map50_95',
    'ci95_low',
    'ci95_high',
    'images',
    'total_images',
    'created',
  ];
  const items = entries.map((e) => {
    const isStale = stale(e, bench.fingerprint);
    const rank = isStale ? null : (rankMap.get(e.id) ?? null);
    const who = contestantFacts(e, agents);
    const contestant = [who.arch, ...who.facts].filter(Boolean).join(' · ');
    const axes = Object.fromEntries(
      AXES.map((a) => [a, e.score?.axes.find((x) => x.axis === a)?.value ?? null]),
    ) as Record<(typeof AXES)[number], number | null>;
    return {
      rank,
      id: e.id,
      name: e.name,
      contestant,
      status: e.status,
      score: e.score?.total ?? null,
      partial: e.score?.partial ?? null,
      earlier_version: isStale,
      axes,
      map50_95: e.map50_95 ?? null,
      ci95: e.ci95 ?? null,
      images: e.images,
      total_images: bench.images,
      created: e.created,
    };
  });
  const rows: CsvCell[][] = items.map((r) => [
    r.rank,
    r.id,
    r.name,
    r.contestant,
    r.status,
    r.score,
    r.partial,
    r.earlier_version,
    ...AXES.map((a) => r.axes[a]),
    r.map50_95,
    r.ci95?.[0] ?? null,
    r.ci95?.[1] ?? null,
    r.images,
    r.total_images,
    r.created,
  ]);
  return {
    filenameBase: `benchmark-${bench.name}-leaderboard`,
    headers,
    rows,
    json: {
      benchmark: {
        id: bench.id,
        name: bench.name,
        fingerprint: bench.fingerprint,
        images: bench.images,
        classes: bench.classes,
      },
      count: items.length,
      entries: items,
    },
  };
}

export function compareExport(
  bench: Bench,
  shown: readonly BenchEntry[],
  tableRows: readonly Row[],
  old: readonly boolean[],
  best: (key: string, values: (number | null)[], d: number) => Set<number>,
  caveats: readonly { text: string }[],
  results: readonly (BenchResult | undefined)[],
): TableExportPayload {
  const headers: string[] = ['measure'];
  for (let i = 0; i < shown.length; i++) {
    const e = shown[i]!;
    headers.push(i === 0 ? `${e.name} (baseline)` : e.name);
    if (i > 0) headers.push(`${e.name} vs baseline`);
  }
  const rows: CsvCell[][] = [];
  const measures = tableRows.map((row) => {
    const winners = best(row.key, row.values, row.d);
    const csvRow: CsvCell[] = [row.label];
    const baseVal = row.values[0] ?? null;
    const scale = 10 ** row.d;
    const entryCells = shown.map((e, i) => {
      const v = row.values[i] ?? null;
      const comparable = !old[i] && !old[0] && v !== null && baseVal !== null && Number.isFinite(v) && Number.isFinite(baseVal);
      // Match the table's `diffAsPrinted`: difference of rounded printed values so CSV/JSON never disagree with the UI.
      const rawRounded = i > 0 && comparable ? (Math.round(v * scale) - Math.round(baseVal * scale)) / scale : null;
      const delta = rawRounded === null ? null : Object.is(rawRounded, -0) ? 0 : rawRounded;
      const deltaPrinted = i > 0 && comparable ? diffAsPrinted(v, baseVal, row.d) : null;
      csvRow.push(v);
      if (i > 0) csvRow.push(delta);
      return {
        id: e.id,
        name: e.name,
        value: v,
        delta,
        delta_printed: deltaPrinted,
        best: winners.has(i),
      };
    });
    rows.push(csvRow);
    return { key: row.key, label: row.label, values: entryCells };
  });

  return {
    filenameBase: `benchmark-${bench.name}-compare`,
    headers,
    rows,
    json: {
      benchmark: { id: bench.id, name: bench.name, fingerprint: bench.fingerprint, images: bench.images },
      baseline_id: shown[0]?.id ?? null,
      caveats: caveats.map((c) => c.text),
      entries: shown.map((e, i) => ({
        id: e.id,
        name: e.name,
        baseline: i === 0,
        earlier_version: old[i] ?? false,
        score: e.score?.total ?? null,
        partial: e.score?.partial ?? null,
        map50_95: e.map50_95 ?? null,
        ci95: e.ci95 ?? null,
        ece: results[i]?.calibration?.ece ?? null,
        self_labeling: results[i]?.self_labeling
          ? {
              threshold: results[i]!.self_labeling!.threshold,
              recall: results[i]!.self_labeling!.recall,
              precision: results[i]!.self_labeling!.precision,
            }
          : null,
      })),
      measures,
    },
  };
}
