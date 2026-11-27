/**
 * Dataset Health & Class-Balance Report export builders (CSV & JSON) for `StatsTab`.
 * Uses Studio's shared `$lib/tableExport` helpers (RFC-4180 quoting, spreadsheet formula neutralization, and
 * object-URL cleanup) and derives every figure strictly from `Dataset`, `DatasetStats`, and optional `IssueCounts`.
 */
import type { Dataset, DatasetStats, IssueCounts, IssueKind, IssueSeverity } from '$lib/api/types';
import { exportTableCsv, exportTableJson, type CsvCell } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import { FEW_BOXES, healthChecks, TINY_SIZE } from './health';
import { computeDatasetInsights, type DatasetInsightsData } from './insights';
import { KIND_LABEL, KINDS } from './issues';

export const DATASET_REPORT_CSV_HEADERS = [
  'section',
  'name',
  'count',
  'share_pct',
  'severity',
  'detail',
] as const;

const SEVERITY_ORDER: readonly IssueSeverity[] = ['error', 'warning', 'info'];

function round2(n: number): number {
  return Number(n.toFixed(2));
}

function classStatus(count: number): 'missing' | 'rare' | 'ok' {
  if (count === 0) return 'missing';
  if (count < FEW_BOXES) return 'rare';
  return 'ok';
}

/** Derives sorted per-class rows directly from `computeDatasetInsights` so report and UI counts never diverge. */
function classListFromInsights(insights: DatasetInsightsData) {
  return [
    ...insights.presentClasses.map((c) => ({
      class_id: c.cls,
      name: c.name,
      boxes: c.count,
      share_pct: round2(c.sharePct),
      status: classStatus(c.count),
    })),
    ...insights.missingClasses.map((m) => ({
      class_id: m.cls,
      name: m.name,
      boxes: 0,
      share_pct: 0,
      status: 'missing' as const,
    })),
  ].sort((a, b) => b.boxes - a.boxes || a.class_id - b.class_id);
}

/** Iterates nonzero `counts.by_kind` entries ordered by `KINDS` severity without duplicating the severity mapping. */
function checkIssueEntries(counts?: IssueCounts): { kind: IssueKind; label: string; severity: IssueSeverity; count: number }[] {
  if (!counts) return [];
  const out: { kind: IssueKind; label: string; severity: IssueSeverity; count: number }[] = [];
  for (const severity of SEVERITY_ORDER) {
    for (const kind of KINDS[severity]) {
      const n = counts.by_kind[kind] ?? 0;
      if (n > 0) out.push({ kind, label: KIND_LABEL[kind], severity, count: n });
    }
  }
  return out;
}

/** Raw count of boxes in bins strictly at or below `TINY_SIZE` (matching `shareBelow` in `./health`). */
function tinyBoxCount(stats: DatasetStats): number {
  let below = 0;
  stats.box_size_hist.counts.forEach((c, i) => {
    if ((stats.box_size_hist.edges[i + 1] ?? Infinity) <= TINY_SIZE + 1e-12) below += c;
  });
  return below;
}

/** Builds a structured JSON health, split, class-balance, and check-issues report for a dataset. */
export function buildDatasetHealthReportJson(
  dataset: Dataset,
  stats: DatasetStats,
  counts?: IssueCounts,
) {
  const insights = computeDatasetInsights(dataset, stats, counts);
  const checks = healthChecks(dataset, stats, counts);
  const findings = checks.filter((h) => h.severity !== 'ok');
  const classes = classListFromInsights(insights);
  const byKindEntries = checkIssueEntries(counts);

  return {
    dataset: {
      id: dataset.id,
      name: dataset.name,
      format: dataset.format,
      path: dataset.path,
      created: dataset.created,
    },
    summary: {
      total_images: insights.totalImages,
      total_boxes: insights.totalBoxes,
      boxes_per_image: insights.boxesPerImage !== null ? round2(insights.boxesPerImage) : null,
      empty_images: insights.emptyImages,
      empty_image_pct: round2(insights.emptyPct),
      broken_labels: insights.brokenLabels,
      class_count: dataset.classes.length,
      present_classes: insights.presentClasses.length,
      missing_classes: insights.missingClasses.length,
      imbalance_ratio: insights.imbalanceRatio,
      tiny_boxes: tinyBoxCount(stats),
      tiny_box_pct: round2(insights.tinyBoxPct),
      finding_count: findings.length,
    },
    splits: {
      train: dataset.splits.train,
      val: dataset.splits.val,
      test: dataset.splits.test,
      train_pct: round2(insights.splits.find((s) => s.split === 'train')?.pct ?? 0),
      val_pct: round2(insights.splits.find((s) => s.split === 'val')?.pct ?? 0),
      test_pct: round2(insights.splits.find((s) => s.split === 'test')?.pct ?? 0),
    },
    classes,
    health_findings: findings.map((f) => ({
      id: f.id,
      severity: f.severity,
      title: f.title,
      detail: f.detail,
      ...(f.files !== undefined ? { files: f.files } : {}),
    })),
    check_issues: counts
      ? {
          error: counts.error,
          warning: counts.warning,
          info: counts.info,
          by_kind: byKindEntries,
        }
      : null,
  };
}

/** Builds RFC-4180 CSV rows covering summary metrics, split breakdown, class counts, health findings, and check issues. */
export function buildDatasetHealthReportCsvRows(
  dataset: Dataset,
  stats: DatasetStats,
  counts?: IssueCounts,
): CsvCell[][] {
  const insights = computeDatasetInsights(dataset, stats, counts);
  const checks = healthChecks(dataset, stats, counts);
  const findings = checks.filter((h) => h.severity !== 'ok');
  const classes = classListFromInsights(insights);
  const rows: CsvCell[][] = [];

  rows.push(['summary', 'total_images', insights.totalImages, '', '', `${dataset.format} dataset`]);
  rows.push([
    'summary',
    'total_boxes',
    insights.totalBoxes,
    '',
    '',
    insights.boxesPerImage !== null ? `${insights.boxesPerImage.toFixed(2)} boxes/image` : '',
  ]);
  rows.push([
    'summary',
    'empty_images',
    insights.emptyImages,
    round2(insights.emptyPct),
    insights.emptyWarn ? 'warning' : 'ok',
    'images without boxes',
  ]);
  rows.push([
    'summary',
    'broken_labels',
    insights.brokenLabels,
    '',
    insights.brokenLabels > 0 ? 'danger' : 'ok',
    insights.brokenLabels > 0 ? 'unparseable label files' : 'all labels parse',
  ]);
  if (insights.imbalanceRatio !== null) {
    rows.push([
      'summary',
      'class_imbalance_ratio',
      insights.imbalanceRatio,
      '',
      insights.imbalanced ? 'warning' : 'ok',
      insights.mostCommon && insights.rarest
        ? `${insights.mostCommon.name} (${insights.mostCommon.count}) vs ${insights.rarest.name} (${insights.rarest.count})`
        : '',
    ]);
  }
  rows.push([
    'summary',
    'tiny_boxes',
    tinyBoxCount(stats),
    round2(insights.tinyBoxPct),
    insights.tinyWarn ? 'warning' : 'ok',
    'boxes under 5% of image side',
  ]);

  for (const sp of insights.splits) {
    rows.push([
      'split',
      sp.split,
      sp.count,
      round2(sp.pct),
      sp.split === 'val' && sp.count === 0 ? 'warning' : sp.split === 'train' && sp.count === 0 ? 'danger' : 'ok',
      `${sp.split} split images`,
    ]);
  }

  for (const c of classes) {
    const sev = c.status === 'missing' ? 'danger' : c.status === 'rare' ? 'warning' : 'ok';
    const detail =
      c.status === 'missing' ? 'no boxes' : c.status === 'rare' ? `fewer than ${FEW_BOXES} boxes` : 'represented';
    rows.push(['class', c.name, c.boxes, c.share_pct, sev, detail]);
  }

  if (findings.length === 0) {
    rows.push(['health', 'No problems found', 0, '', 'ok', 'Labels parse, every class is represented and class balance is within 5×.']);
  } else {
    for (const f of findings) {
      rows.push(['health', f.title, f.files ?? '', '', f.severity, f.detail]);
    }
  }

  for (const entry of checkIssueEntries(counts)) {
    rows.push(['check_issue', entry.label, entry.count, '', entry.severity, entry.kind]);
  }

  return rows;
}

function reportFilenameBase(dataset: Dataset): string {
  return `dataset-health-${dataset.name || dataset.id}`;
}

/** Downloads the dataset health & class-balance report as RFC-4180 CSV with toast feedback. */
export function exportDatasetReportCsv(
  dataset: Dataset,
  stats: DatasetStats,
  counts?: IssueCounts,
): boolean {
  const rows = buildDatasetHealthReportCsvRows(dataset, stats, counts);
  const ok = exportTableCsv(reportFilenameBase(dataset), DATASET_REPORT_CSV_HEADERS, rows, 'dataset-report-export');
  if (ok) {
    toast.success(
      'Health report CSV exported',
      `${dataset.name} · ${dataset.classes.length} ${dataset.classes.length === 1 ? 'class' : 'classes'}, ${rows.length} rows.`,
      undefined,
      { key: 'dataset-report-export' },
    );
  }
  return ok;
}

/** Downloads the dataset health & class-balance report as pretty-printed JSON with toast feedback. */
export function exportDatasetReportJson(
  dataset: Dataset,
  stats: DatasetStats,
  counts?: IssueCounts,
): boolean {
  const payload = buildDatasetHealthReportJson(dataset, stats, counts);
  const ok = exportTableJson(reportFilenameBase(dataset), payload, 'dataset-report-export');
  if (ok) {
    const f = payload.summary.finding_count;
    toast.success(
      'Health report JSON exported',
      `${dataset.name} · ${payload.summary.class_count} ${payload.summary.class_count === 1 ? 'class' : 'classes'}, ${f} ${f === 1 ? 'finding' : 'findings'}.`,
      undefined,
      { key: 'dataset-report-export' },
    );
  }
  return ok;
}
