/**
 * Pure CSV / JSON / Markdown payload builders for the Benchmarks list, a single benchmark's leaderboard, and BenchCompare.
 */
import type { Bench, BenchEntry, BenchResult, LabelAgent } from '$lib/api/types';
import { AXES } from '$lib/bench/rules';
import { exportTableCsv, exportTableJson, type CsvCell } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import { diffAsPrinted } from './charts';
import { ranks, stale } from './leaderboard';
import type { Row } from './compare/state';
import { contestantFacts, imagesText, keptText, sourceKind } from './words';

export interface CompareConditionRow {
  key: string;
  name: string;
  values: (number | null)[];
}

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

// Escapes dynamic text as literal GitHub-Flavored Markdown table/inline text: collapses newlines and
// escapes backslashes alongside pipes, code spans, emphasis, strikethrough, link brackets, HTML delimiters, and entities.
const escMd = (s: string) => s.replace(/[\r\n]+/g, ' ').replace(/[\\|`*_~[\]<>&]/g, '\\$&');

function conditionDelta(v: number | null, base: number | null, comparable: boolean): { delta: number | null; printed: string | null } {
  if (!comparable || v === null || base === null || !Number.isFinite(v) || !Number.isFinite(base)) {
    return { delta: null, printed: null };
  }
  const diffPts = Math.round(v * 100) - Math.round(base * 100);
  const delta = diffPts === 0 ? 0 : diffPts / 100;
  const printed = diffPts === 0 ? '±0 pp' : `${diffPts > 0 ? '+' : '−'}${Math.abs(diffPts)} pp`;
  return { delta, printed };
}

export function compareExport(
  bench: Bench,
  shown: readonly BenchEntry[],
  tableRows: readonly Row[],
  old: readonly boolean[],
  best: (key: string, values: (number | null)[], d: number) => Set<number>,
  caveats: readonly { text: string }[],
  results: readonly (BenchResult | undefined)[],
  conditions: readonly CompareConditionRow[] = [],
  labelOf?: (e: BenchEntry) => string,
): TableExportPayload {
  const entryName = (e: BenchEntry) => (labelOf ? labelOf(e) : e.name);
  const headers: string[] = ['measure'];
  for (let i = 0; i < shown.length; i++) {
    const name = entryName(shown[i]!);
    headers.push(i === 0 ? `${name} (baseline)` : name);
    if (i > 0) headers.push(`${name} vs baseline`);
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
        name: entryName(e),
        value: v,
        delta,
        delta_printed: deltaPrinted,
        best: winners.has(i),
      };
    });
    rows.push(csvRow);
    return { key: row.key, label: row.label, values: entryCells };
  });

  const conditionItems = conditions.map((cond) => {
    const csvRow: CsvCell[] = [`Condition · ${cond.name}`];
    const baseVal = cond.values[0] ?? null;
    const entryCells = shown.map((e, i) => {
      const v = cond.values[i] ?? null;
      const comparable = i > 0 && !old[i] && !old[0];
      const { delta, printed } = conditionDelta(v, baseVal, comparable);
      csvRow.push(v);
      if (i > 0) csvRow.push(delta);
      return {
        id: e.id,
        name: entryName(e),
        retention: v,
        delta,
        delta_printed: printed,
      };
    });
    rows.push(csvRow);
    return { condition: cond.key, name: cond.name, values: entryCells };
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
        name: entryName(e),
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
      ...(conditionItems.length ? { conditions: conditionItems } : {}),
    },
  };
}

/**
 * Renders a clean GitHub-Flavored Markdown comparison report for BenchCompare:
 * overall score + axis values and baseline deltas, plus per-stress-condition accuracy kept rows.
 */
export function compareMarkdown(
  bench: Bench,
  shown: readonly BenchEntry[],
  tableRows: readonly Row[],
  old: readonly boolean[],
  best: (key: string, values: (number | null)[], d: number) => Set<number>,
  caveats: readonly { text: string }[],
  conditions: readonly CompareConditionRow[] = [],
  labelOf?: (e: BenchEntry) => string,
): string {
  const entryName = (e: BenchEntry) => (labelOf ? labelOf(e) : e.name);
  const baseline = shown[0] ? entryName(shown[0]) : '—';
  const colHeaders: string[] = ['Measure'];
  const colAlign: string[] = ['---'];
  for (let i = 0; i < shown.length; i++) {
    const name = escMd(entryName(shown[i]!));
    colHeaders.push(i === 0 ? `${name} (baseline)` : name);
    colAlign.push('---:');
    if (i > 0) {
      colHeaders.push(`${name} vs baseline`);
      colAlign.push('---:');
    }
  }

  const lines: string[] = [
    `# Benchmark comparison: ${escMd(bench.name)}`,
    '',
    `${imagesText(bench.images)} · ${shown.length} entries · baseline: **${escMd(baseline)}**`,
    '',
    '## Score & axes',
    '',
    `| ${colHeaders.join(' | ')} |`,
    `| ${colAlign.join(' | ')} |`,
  ];

  for (const row of tableRows) {
    const winners = best(row.key, row.values, row.d);
    const baseVal = row.values[0] ?? null;
    const cells: string[] = [escMd(row.label)];
    for (let i = 0; i < shown.length; i++) {
      const v = row.values[i] ?? null;
      const finiteV = v !== null && Number.isFinite(v);
      if (!finiteV) {
        cells.push('—');
      } else {
        const printed = escMd(row.fmt(v));
        cells.push(winners.has(i) ? `**${printed}**` : printed);
      }
      if (i > 0) {
        const finiteBase = baseVal !== null && Number.isFinite(baseVal);
        const d = !old[i] && !old[0] && finiteV && finiteBase ? diffAsPrinted(v, baseVal, row.d) : null;
        cells.push(d ? escMd(d) : '—');
      }
    }
    lines.push(`| ${cells.join(' | ')} |`);
  }

  if (conditions.length) {
    const condHeaders = ['Condition', ...colHeaders.slice(1)];
    lines.push('', '## Stress conditions (accuracy kept)', '', `| ${condHeaders.join(' | ')} |`, `| ${colAlign.join(' | ')} |`);
    for (const cond of conditions) {
      const baseVal = cond.values[0] ?? null;
      const cells: string[] = [escMd(cond.name)];
      for (let i = 0; i < shown.length; i++) {
        const v = cond.values[i] ?? null;
        cells.push(v === null ? '—' : escMd(keptText(v)));
        if (i > 0) {
          const { printed } = conditionDelta(v, baseVal, !old[i] && !old[0]);
          cells.push(printed ? escMd(printed) : '—');
        }
      }
      lines.push(`| ${cells.join(' | ')} |`);
    }
  }

  if (caveats.length) {
    lines.push('', '### Notes', '');
    for (const c of caveats) {
      lines.push(`- ${escMd(c.text)}`);
    }
  }

  return `${lines.join('\n')}\n`;
}

/** Triggers BenchCompare CSV or JSON download and returns an announcement string on success. */
export function runCompareDownload(
  format: 'csv' | 'json',
  bench: Bench | undefined,
  shown: readonly BenchEntry[],
  rows: readonly Row[],
  old: readonly boolean[],
  best: (key: string, values: (number | null)[], d: number) => Set<number>,
  caveats: readonly { text: string }[],
  results: readonly (BenchResult | undefined)[],
  conditions: readonly CompareConditionRow[],
  labelOf?: (e: BenchEntry) => string,
): string | null {
  if (!bench || shown.length < 2) return null;
  const payload = compareExport(bench, shown, rows, old, best, caveats, results, conditions, labelOf);
  const ok =
    format === 'csv'
      ? exportTableCsv(payload.filenameBase, payload.headers, payload.rows)
      : exportTableJson(payload.filenameBase, payload.json);
  return ok ? `Exported comparison of ${shown.length} entries as ${format.toUpperCase()}.` : null;
}

/** Copies BenchCompare Markdown to clipboard, showing toast feedback and returning an announcement string on success. */
export async function runCompareCopyMarkdown(
  bench: Bench | undefined,
  shown: readonly BenchEntry[],
  rows: readonly Row[],
  old: readonly boolean[],
  best: (key: string, values: (number | null)[], d: number) => Set<number>,
  caveats: readonly { text: string }[],
  conditions: readonly CompareConditionRow[],
  labelOf: (e: BenchEntry) => string,
): Promise<string | null> {
  if (!bench || shown.length < 2) return null;
  const md = compareMarkdown(bench, shown, rows, old, best, caveats, conditions, labelOf);
  try {
    await navigator.clipboard.writeText(md);
    toast.success('Markdown copied', `Comparison of ${shown.length} entries copied to clipboard.`, undefined, {
      key: 'bench-compare-export',
    });
    return `Copied comparison of ${shown.length} entries as Markdown.`;
  } catch {
    toast.error(new Error('Your browser blocked clipboard access.'), 'Could not copy Markdown', {
      key: 'bench-compare-export',
    });
    return null;
  }
}
