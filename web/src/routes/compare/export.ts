// Pure CSV & GitHub-Flavored Markdown report builders for Compare runs (`../RunCompare.tsx` and `./CompareTable.tsx`).
// Shares setting row definitions with `CompareTable` so exported configuration & metric cells match the UI verbatim.
import { bestMark, bestWord } from '$lib/bestMark';
import { trainingSeconds } from '$lib/api/progress';
import type { Dataset, Run, RunMetrics } from '$lib/api/types';
import { conf, deviceName, duration, metric } from '$lib/format';
import { FROM_SCRATCH } from '$lib/lineage';
import { exportTableCsv, type CsvCell } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import {
  batchSetting,
  closeMosaicSetting,
  effectiveBatchSetting,
  emaSetting,
  headSetting,
  lrSetting,
  optimizerSetting,
  precisionSetting,
  settingText,
  warmupSetting,
  type Setting,
} from '../run/resolved';
import { DATA_CAVEAT } from './config';

export type CompareSettingRow = [string, (r: Run) => Setting];

export interface RunCompareExportInput {
  chosen: readonly Run[];
  baselineId: string | null;
  datasets: readonly Dataset[] | undefined;
  metrics: ReadonlyMap<string, RunMetrics> | undefined;
  sameData: boolean;
  startOf: (r: Run) => string | null;
}

// Escapes dynamic text as literal GitHub-Flavored Markdown table/inline text.
const escMd = (s: string) => s.replace(/[\r\n]+/g, ' ').replace(/[\\|`*_~[\]<>&]/g, '\\$&');

/** Canonical side-by-side configuration rows shared between `CompareTable` and CSV/Markdown exports. */
export function buildCompareSettingRows(
  datasets: readonly Dataset[] | undefined,
  startOf: (r: Run) => string | null,
): CompareSettingRow[] {
  const plain = (value: string): Setting => ({ value });
  const ident = (value: string): Setting => ({ value, ident: true });
  return [
    ['Dataset', (r) => plain(datasets?.find((d) => d.id === r.config.dataset_id)?.name ?? r.dataset?.name ?? r.config.dataset_id)],
    ['Model', (r) => ident(r.config.arch)],
    ['Start from', (r) => (startOf(r) ? ident(startOf(r) ?? '') : plain(FROM_SCRATCH))],
    ['Epochs', (r) => plain(`${r.epoch}/${r.epochs}`)],
    ['Batch', (r) => batchSetting(r.config, r.resolved)],
    ['Effective batch', (r) => (r.resolved ? effectiveBatchSetting(r.resolved) : plain('—'))],
    ['Image size', (r) => plain(`${r.config.imgsz} px`)],
    ['Optimizer', (r) => optimizerSetting(r.config, r.resolved)],
    ['LR', (r) => lrSetting(r.config, r.resolved)],
    ['Warmup', (r) => (r.resolved ? warmupSetting(r.resolved) : plain('—'))],
    ['EMA', (r) => (r.resolved ? emaSetting(r.resolved) : null) ?? plain('—')],
    ['Mosaic / mixup', (r) => plain(`${r.config.augment.mosaic} / ${r.config.augment.mixup}`)],
    ['Close mosaic', (r) => closeMosaicSetting(r)],
    ['Precision', (r) => precisionSetting(r.config)],
    ['Head', (r) => headSetting(r.config)],
    ['Device', (r) => plain(deviceName(r.device))],
    ['Duration', (r) => plain(r.started ? duration(trainingSeconds(r)) : '—')],
  ];
}

/** Orders compared runs with the active baseline first, preserving chip order for the remaining runs. */
export function orderedRunsForExport(chosen: readonly Run[], baselineId: string | null): Run[] {
  const base = chosen.find((r) => r.id === baselineId) ?? chosen[0];
  if (!base) return [];
  return [base, ...chosen.filter((r) => r.id !== base.id)];
}

/** Disambiguates runs that share a display name by appending a short ID suffix. */
export function runExportLabel(run: Run, all: readonly Run[]): string {
  const dup = all.filter((r) => r.name === run.name).length > 1;
  return dup ? `${run.name} (${run.id.slice(-4)})` : run.name;
}

function datasetNameOf(id: string, datasets: readonly Dataset[] | undefined, runs: readonly Run[]): string {
  return datasets?.find((d) => d.id === id)?.name ?? runs.find((r) => r.config.dataset_id === id)?.dataset?.name ?? id;
}

function bestMarksFor(runs: readonly Run[], sameData: boolean, datasets: readonly Dataset[] | undefined) {
  const marks = new Map<string, { word: string; decimals: number }>();
  for (const dsId of new Set(runs.map((r) => r.config.dataset_id))) {
    const group = runs.filter((r) => r.config.dataset_id === dsId);
    const m = bestMark(group.map((r) => r.best?.map50_95));
    for (const r of group) if (m.decimals > 3) marks.set(r.id, { word: '', decimals: m.decimals });
    for (const i of m.best) {
      marks.set(group[i]!.id, {
        word: sameData ? bestWord(m) : `${bestWord(m)} on ${datasetNameOf(dsId, datasets, runs)}`,
        decimals: m.decimals,
      });
    }
  }
  return marks;
}

function bestEpochVal(run: Run, metrics: ReadonlyMap<string, RunMetrics> | undefined) {
  return run.best ? metrics?.get(run.id)?.epochs.find((e) => e.epoch === run.best?.epoch)?.val : undefined;
}

function formatPrSummary(run: Run, metrics: ReadonlyMap<string, RunMetrics> | undefined): string {
  const v = bestEpochVal(run, metrics);
  if (!v) return '—';
  const base = `${metric(v.precision)} · ${metric(v.recall)}`;
  if (v.precision === 0 && v.recall === 0) return `${base} (no detections)`;
  const at = v.best_f1_conf != null ? `conf ${conf(v.best_f1_conf)}` : 'best-F1 conf';
  return `${base} (${at}, ep ${run.best?.epoch})`;
}

/** Builds RFC-4180 CSV headers, rows, and a safe filename base for Compare runs. */
export function buildRunCompareCsv(input: RunCompareExportInput): {
  filenameBase: string;
  headers: string[];
  rows: CsvCell[][];
} {
  const ordered = orderedRunsForExport(input.chosen, input.baselineId);
  const labels = ordered.map((r) => runExportLabel(r, ordered));
  const headers: string[] = [
    'section',
    'measure',
    'differs',
    ...labels.map((name, i) => (i === 0 ? `${name} (baseline)` : name)),
  ];

  const rows: CsvCell[][] = [];
  const valStatus = ordered.map((r) => r.status);
  rows.push(['result', 'Status', new Set(valStatus).size > 1 ? 'yes' : 'no', ...valStatus]);

  const mapVals = ordered.map((r) => (r.best && Number.isFinite(r.best.map50_95) ? Number(r.best.map50_95.toFixed(4)) : null));
  rows.push(['result', 'Best mAP50-95', new Set(mapVals.map(String)).size > 1 ? 'yes' : 'no', ...mapVals]);

  const bestEpVals = ordered.map((r) => r.best?.epoch ?? null);
  rows.push(['result', 'Best epoch', new Set(bestEpVals.map(String)).size > 1 ? 'yes' : 'no', ...bestEpVals]);

  const precVals = ordered.map((r) => {
    const v = bestEpochVal(r, input.metrics);
    return v && Number.isFinite(v.precision) ? Number(v.precision.toFixed(4)) : null;
  });
  rows.push(['result', 'Precision (best-F1)', new Set(precVals.map(String)).size > 1 ? 'yes' : 'no', ...precVals]);

  const recVals = ordered.map((r) => {
    const v = bestEpochVal(r, input.metrics);
    return v && Number.isFinite(v.recall) ? Number(v.recall.toFixed(4)) : null;
  });
  rows.push(['result', 'Recall (best-F1)', new Set(recVals.map(String)).size > 1 ? 'yes' : 'no', ...recVals]);

  const confVals = ordered.map((r) => {
    const v = bestEpochVal(r, input.metrics);
    if (!v || (v.precision === 0 && v.recall === 0)) return null;
    return v.best_f1_conf != null && Number.isFinite(v.best_f1_conf) ? Number(v.best_f1_conf.toFixed(4)) : null;
  });
  rows.push(['result', 'Best-F1 conf', new Set(confVals.map(String)).size > 1 ? 'yes' : 'no', ...confVals]);

  const settingRows = buildCompareSettingRows(input.datasets, input.startOf);
  for (const [label, fn] of settingRows) {
    const texts = ordered.map((r) => settingText(fn(r)));
    const differs = new Set(texts).size > 1;
    rows.push(['setting', label, differs ? 'yes' : 'no', ...texts]);
  }

  const baseName = ordered[0]?.name ?? 'runs';
  const filenameBase = `runs-compare-${baseName}-${ordered.length}`;
  return { filenameBase, headers, rows };
}

/** Builds a clean GitHub-Flavored Markdown comparison report for Compare runs. */
export function buildRunCompareMarkdown(input: RunCompareExportInput): string {
  const ordered = orderedRunsForExport(input.chosen, input.baselineId);
  if (!ordered.length) return '';
  const labels = ordered.map((r) => runExportLabel(r, ordered));
  const baselineLabel = labels[0]!;
  const marks = bestMarksFor(ordered, input.sameData, input.datasets);

  const colHeaders = ['Measure', ...labels.map((name, i) => (i === 0 ? `${escMd(name)} (baseline)` : escMd(name)))];
  const colAlign = ['---', ...labels.map(() => '---')];
  const tableHead = (lead: string) => [`| ${[lead, ...colHeaders.slice(1)].join(' | ')} |`, `| ${colAlign.join(' | ')} |`];

  const lines: string[] = [
    '# Run comparison',
    '',
    `${ordered.length} ${ordered.length === 1 ? 'run' : 'runs'} · baseline: **${escMd(baselineLabel)}**`,
  ];
  if (!input.sameData && ordered.length > 1) {
    lines.push('', `> ${escMd(DATA_CAVEAT)}`);
  }

  lines.push('', '## Results', '', ...tableHead('Measure'));
  lines.push(`| Status | ${ordered.map((r) => escMd(r.status)).join(' | ')} |`);
  lines.push(
    `| Best mAP50-95 | ${ordered
      .map((r) => {
        const mark = marks.get(r.id);
        const val = r.best && mark ? r.best.map50_95.toFixed(mark.decimals) : metric(r.best?.map50_95);
        return mark?.word ? `**${escMd(val)}** (${escMd(mark.word)})` : escMd(val);
      })
      .join(' | ')} |`,
  );
  lines.push(`| Best epoch | ${ordered.map((r) => (r.best ? String(r.best.epoch) : '—')).join(' | ')} |`);
  lines.push(`| Precision · recall | ${ordered.map((r) => escMd(formatPrSummary(r, input.metrics))).join(' | ')} |`);

  const settingRows = buildCompareSettingRows(input.datasets, input.startOf);
  const evaluated = settingRows.map(([label, fn]) => {
    const values = ordered.map((r) => settingText(fn(r)));
    return { label, values, differs: new Set(values).size > 1 };
  });
  const diffRows = ordered.length > 1 ? evaluated.filter((r) => r.differs) : evaluated;
  const sameRows = ordered.length > 1 ? evaluated.filter((r) => !r.differs) : [];

  if (diffRows.length) {
    lines.push('', '## Configuration differences', '', ...tableHead('Setting'));
    for (const row of diffRows) {
      const baseText = row.values[0]!;
      const cells = row.values.map((v, i) => (i > 0 && v !== baseText ? `**${escMd(v)}**` : escMd(v)));
      lines.push(`| ${escMd(row.label)} | ${cells.join(' | ')} |`);
    }
  }

  if (sameRows.length) {
    lines.push('', '## Identical settings', '', ...tableHead('Setting'));
    for (const row of sameRows) {
      lines.push(`| ${escMd(row.label)} | ${row.values.map(escMd).join(' | ')} |`);
    }
  }

  return `${lines.join('\n')}\n`;
}

/** Downloads Compare runs as formula-safe RFC-4180 CSV and returns an announcement string on success. */
export function exportRunCompareCsv(input: RunCompareExportInput): string | null {
  if (input.chosen.length < 2) return null;
  const { filenameBase, headers, rows } = buildRunCompareCsv(input);
  const ok = exportTableCsv(filenameBase, headers, rows, 'run-compare-export');
  if (!ok) return null;
  toast.success('Comparison CSV exported', `Saved comparison of ${input.chosen.length} runs.`, undefined, {
    key: 'run-compare-export',
  });
  return `Exported comparison of ${input.chosen.length} runs as CSV.`;
}

/** Copies Compare runs as GitHub-Flavored Markdown to the clipboard with toast feedback. */
export async function copyRunCompareMarkdown(input: RunCompareExportInput): Promise<string | null> {
  if (input.chosen.length < 2) return null;
  const md = buildRunCompareMarkdown(input);
  try {
    await navigator.clipboard.writeText(md);
    toast.success('Markdown copied', `Comparison of ${input.chosen.length} runs copied to clipboard.`, undefined, {
      key: 'run-compare-export',
    });
    return `Copied comparison of ${input.chosen.length} runs as Markdown.`;
  } catch {
    toast.error(new Error('Your browser blocked clipboard access.'), 'Could not copy Markdown', {
      key: 'run-compare-export',
    });
    return null;
  }
}
