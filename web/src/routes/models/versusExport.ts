/**
 * Pure Markdown table and CSV report builders for `Models → Versus` (`Engine speed`), exporting the exact loaded
 * reference benchmark rows plus any live measurements reported by this server (`GET /api/system/throughput`).
 */
import type { ThroughputTable } from '$lib/api/types';
import { advantage, BENCH_SOURCE, unlike, unlikeCondition, versusYolo26n, type Versus as VersusRow } from '$lib/benchmarks';
import { deviceName, num } from '$lib/format';
import type { CsvCell } from '$lib/tableExport';
import { latencyMs, latencyRange, measuredLatency, measuredTraining } from '$lib/throughput';

export interface VersusReportRow {
  source: 'reference' | 'server';
  metric: string;
  unit: string;
  folo: string;
  yolo: string;
  comparison: string;
  note: string;
}

const fmtVal = (v: VersusRow, x: number) => num(x, v.unit === 'ms' ? (x < 10 ? 2 : 1) : 0);

function leadText(v: VersusRow): string {
  const a = advantage(v);
  const ahead = a >= 1;
  if (v.unit === 'MiB') return `${Math.round(Math.abs(1 - v.folo / v.yolo) * 100)}% ${ahead ? 'less' : 'more'} memory`;
  return ahead ? `${a.toFixed(2)}× faster` : `${(1 / a).toFixed(2)}× slower`;
}

/** Builds the exact flat list of loaded `Versus` rows (reference rows + optional live server rows). */
export function buildVersusReport(table?: ThroughputTable | null): { rows: VersusReportRow[]; serverDevice: string | null } {
  const rows: VersusReportRow[] = versusYolo26n.map((v) => {
    const diff = unlike[v.metric];
    return {
      source: 'reference',
      metric: v.metric,
      unit: v.unit,
      folo: fmtVal(v, v.folo),
      yolo: fmtVal(v, v.yolo),
      comparison: diff ? `— (${unlikeCondition(diff)})` : leadText(v),
      note: v.note ?? '',
    };
  });

  const lat = measuredLatency(table ?? null, 'folo-n');
  const train = measuredTraining(table ?? null, 'folo-n');
  const serverDevice = table && (lat || train) ? deviceName(table.device_name) : null;

  if (lat) {
    rows.push({
      source: 'server',
      metric: 'GPU end to end, batch 1',
      unit: 'ms',
      folo: latencyMs(lat.latency_ms),
      yolo: 'not run here',
      comparison: 'Measured here',
      note: `${latencyRange(lat.latency_ms_low, lat.latency_ms_high)} · ${num(lat.samples)} predictions${lat.precision === 'f32' ? '' : ` · ${lat.precision}`}`,
    });
  }
  if (train) {
    rows.push({
      source: 'server',
      metric: `Training, batch ${train.batch}`,
      unit: 'img/s',
      folo: `${num(train.imgs_per_s)} img/s`,
      yolo: 'not run here',
      comparison: 'Measured here',
      note: `${train.mosaic ? `${num(train.mosaic.low)}–${num(train.mosaic.high)} img/s · ` : ''}${num(train.samples)} samples · whole trainer`,
    });
  }

  return { rows, serverDevice };
}

const escMd = (s: string) => s.replaceAll('|', '\\|').replace(/[\r\n]+/g, ' ');

/** Renders a clean GitHub-Flavored Markdown table for pasting into PRs, issues, or notebooks. */
export function versusMarkdown(table?: ThroughputTable | null): string {
  const { rows, serverDevice } = buildVersusReport(table);
  const refRows = rows.filter((r) => r.source === 'reference');
  const serverRows = rows.filter((r) => r.source === 'server');

  const lines: string[] = [
    `# Engine speed (${BENCH_SOURCE})`,
    '',
    '| Measured | folo-n | YOLO26n | folo-n vs YOLO26n | Notes |',
    '| --- | ---: | ---: | --- | --- |',
  ];
  for (const r of refRows) {
    lines.push(`| ${escMd(`${r.metric} (${r.unit})`)} | ${escMd(r.folo)} | ${escMd(r.yolo)} | ${escMd(r.comparison)} | ${escMd(r.note)} |`);
  }
  if (serverRows.length) {
    lines.push('', `## Measured on this server (${escMd(serverDevice ?? 'local')})`, '');
    lines.push('| Measured | folo-n | YOLO26n | Status | Notes |');
    lines.push('| --- | ---: | ---: | --- | --- |');
    for (const r of serverRows) {
      lines.push(`| ${escMd(r.metric)} | ${escMd(r.folo)} | ${escMd(r.yolo)} | ${escMd(r.comparison)} | ${escMd(r.note)} |`);
    }
  }
  return `${lines.join('\n')}\n`;
}

/** Builds structured CSV headers and rows for `exportTableCsv`. */
export function versusCsv(table?: ThroughputTable | null): { filenameBase: string; headers: string[]; rows: CsvCell[][] } {
  const { rows, serverDevice } = buildVersusReport(table);
  return {
    filenameBase: 'folo-engine-speed-versus',
    headers: ['source', 'device', 'metric', 'unit', 'folo_n', 'yolo26n', 'comparison', 'note'],
    rows: rows.map((r) => [
      r.source,
      r.source === 'server' ? (serverDevice ?? '') : 'RTX 4080 SUPER / i7-14700KF',
      r.metric,
      r.unit,
      r.folo,
      r.yolo,
      r.comparison,
      r.note,
    ]),
  };
}
