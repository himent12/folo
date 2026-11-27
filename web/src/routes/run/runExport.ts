// Pure run-detail export builders (CSV/JSON of real loaded per-epoch metrics, run config, and per-class AP)
// and shell-safe checkpoint CLI command builders (`folo val` / `folo predict`, matching docs/CLI.md).
import Download from '$icons/download';
import FileJson from '$icons/file-json';
import type { Dataset, Epoch, Model, Run } from '$lib/api/types';
import type { EvalSource } from '$lib/evaluated';
import { withoutInexactSeed } from '$lib/seed';
import { exportTableCsv, exportTableJson, type CsvCell } from '$lib/tableExport';
import { quote, weightsRef } from '$lib/trainCli';
import type { MenuItem } from '$lib/ui/Menu';

export interface RunPerClassExportRow {
  class: number;
  name: string;
  map50_95: number;
  map50: number;
  precision: number;
  recall: number;
  delta_prev: number | null;
}

export interface RunMetricsExportPayload {
  exported_at: string;
  run: {
    id: string;
    name: string;
    status: Run['status'];
    device: string;
    dataset_id: string;
    dataset_name: string | null;
    model_id: string | null;
    created: string;
    started: string | null;
    finished: string | null;
    epoch: number;
    epochs: number;
    best: Run['best'];
  };
  config: Partial<Run['config']>;
  resolved: Run['resolved'] | null;
  epochs: Array<{
    epoch: number;
    duration_s: number;
    train_loss: Epoch['train_loss'];
    val: {
      map50: number;
      map50_95: number;
      precision: number;
      recall: number;
      best_f1_conf?: number | null;
    };
    per_class: Array<{
      class: number;
      name: string;
      map50: number;
      map50_95: number;
      precision: number;
      recall: number;
    }>;
  }>;
  latest_per_class: RunPerClassExportRow[];
}

export const RUN_METRICS_CSV_HEADERS = [
  'epoch',
  'duration_s',
  'train_box_loss',
  'train_cls_loss',
  'train_dfl_loss',
  'train_total_loss',
  'val_map50',
  'val_map50_95',
  'val_precision',
  'val_recall',
  'val_best_f1_conf',
] as const;

export const RUN_PER_CLASS_CSV_HEADERS = [
  'class_id',
  'class_name',
  'ap50_95',
  'ap50',
  'precision',
  'recall',
  'delta_prev',
] as const;

const className = (classes: readonly string[], id: number) => classes[id] ?? `class ${id}`;

/** Computes the latest epoch's per-class AP table sorted by AP50-95 descending, with delta vs previous epoch. */
export function buildLatestPerClass(epochs: readonly Epoch[], classes: readonly string[]): RunPerClassExportRow[] {
  const last = epochs[epochs.length - 1];
  if (!last || last.val.per_class.length === 0) return [];
  const prev = epochs[epochs.length - 2];
  return last.val.per_class
    .map((c) => {
      const prevRow = prev?.val.per_class.find((p) => p.class === c.class);
      return {
        class: c.class,
        name: className(classes, c.class),
        map50_95: c.map50_95,
        map50: c.map50,
        precision: c.precision,
        recall: c.recall,
        delta_prev: prevRow ? c.map50_95 - prevRow.map50_95 : null,
      };
    })
    .sort((a, b) => b.map50_95 - a.map50_95 || a.class - b.class);
}

/** Tabular rows for per-epoch metrics CSV export (real loaded epoch records only). */
export function buildRunMetricsCsvRows(epochs: readonly Epoch[]): CsvCell[][] {
  return epochs.map((e) => [
    e.epoch,
    e.duration_s,
    e.train_loss.box,
    e.train_loss.cls,
    e.train_loss.dfl,
    e.train_loss.total,
    e.val.map50,
    e.val.map50_95,
    e.val.precision,
    e.val.recall,
    e.val.best_f1_conf ?? '',
  ]);
}

/** Tabular rows for the latest epoch's per-class AP CSV export. */
export function buildPerClassCsvRows(epochs: readonly Epoch[], classes: readonly string[]): CsvCell[][] {
  return buildLatestPerClass(epochs, classes).map((r) => [
    r.class,
    r.name,
    r.map50_95,
    r.map50,
    r.precision,
    r.recall,
    r.delta_prev ?? '',
  ]);
}

/** Structured JSON export payload combining run metadata, request config, resolved settings, per-epoch metrics, and per-class AP. */
export function buildRunMetricsJson(
  run: Run,
  epochs: readonly Epoch[],
  classes: readonly string[],
  datasetName?: string,
  exportedAt = new Date().toISOString(),
): RunMetricsExportPayload {
  return {
    exported_at: exportedAt,
    run: {
      id: run.id,
      name: run.name,
      status: run.status,
      device: run.device,
      dataset_id: run.config.dataset_id,
      dataset_name: datasetName ?? run.dataset?.name ?? null,
      model_id: run.model_id,
      created: run.created,
      started: run.started,
      finished: run.finished,
      epoch: run.epoch,
      epochs: run.epochs,
      best: run.best,
    },
    config: withoutInexactSeed(run.config),
    resolved: run.resolved ?? null,
    epochs: epochs.map((e) => ({
      epoch: e.epoch,
      duration_s: e.duration_s,
      train_loss: { ...e.train_loss },
      val: {
        map50: e.val.map50,
        map50_95: e.val.map50_95,
        precision: e.val.precision,
        recall: e.val.recall,
        ...(e.val.best_f1_conf !== undefined ? { best_f1_conf: e.val.best_f1_conf } : {}),
      },
      per_class: e.val.per_class.map((c) => ({
        class: c.class,
        name: className(classes, c.class),
        map50: c.map50,
        map50_95: c.map50_95,
        precision: c.precision,
        recall: c.recall,
      })),
    })),
    latest_per_class: buildLatestPerClass(epochs, classes),
  };
}

/** Triggers a CSV download of per-epoch metrics for `run`. */
export function downloadRunMetricsCsv(run: Run, epochs: readonly Epoch[]): boolean {
  return exportTableCsv(
    `${run.name || run.id}-metrics`,
    RUN_METRICS_CSV_HEADERS,
    buildRunMetricsCsvRows(epochs),
    `run-export-csv-${run.id}`,
  );
}

/** Triggers a CSV download of the latest epoch's per-class AP table for `run`. */
export function downloadRunPerClassCsv(run: Run, epochs: readonly Epoch[], classes: readonly string[]): boolean {
  return exportTableCsv(
    `${run.name || run.id}-per-class`,
    RUN_PER_CLASS_CSV_HEADERS,
    buildPerClassCsvRows(epochs, classes),
    `run-export-per-class-${run.id}`,
  );
}

/** Triggers a JSON download of full run config, per-epoch metrics, and per-class AP for `run`. */
export function downloadRunMetricsJson(
  run: Run,
  epochs: readonly Epoch[],
  classes: readonly string[],
  datasetName?: string,
): boolean {
  return exportTableJson(
    `${run.name || run.id}-metrics`,
    buildRunMetricsJson(run, epochs, classes, datasetName),
    `run-export-json-${run.id}`,
  );
}

export type CheckpointCliMode = 'val' | 'predict';

export interface CheckpointCliBundle {
  val: string;
  predict: string;
  /** Filename placeholder when the checkpoint's file path is unknown. */
  modelPlaceholder: string | null;
  /** Dataset placeholder when the dataset's path is unknown (e.g. removed dataset). */
  dataPlaceholder: string | null;
}

function flagArg(flag: string, value: string): string {
  return value.startsWith('-') ? quote(`${flag}=${value}`) : `${flag} ${quote(value)}`;
}

/**
 * Builds shell-safe `folo val` and `folo predict` commands for a run's best checkpoint (matching docs/CLI.md).
 * Returns `null` when the run has no saved checkpoint model yet.
 */
export function buildCheckpointCli(
  run: Run,
  model: Model | undefined,
  dataset: Dataset | undefined,
  evalOn: EvalSource | null = null,
): CheckpointCliBundle | null {
  if (!run.model_id) return null;
  const fallbackName = model?.name ?? `${run.name || run.id}-best`;
  const ref = model
    ? weightsRef(run.model_id, model)
    : ({ kind: 'placeholder', value: `<path to ${fallbackName}.folo>`, file: `${fallbackName}.folo` } as const);

  const modelPart = ref.kind === 'placeholder' ? `--model ${ref.value}` : flagArg('--model', ref.value);
  const modelPlaceholder = ref.kind === 'placeholder' ? ref.file : null;

  const dataPlaceholder = dataset?.path ? null : `${run.dataset?.name ?? run.config.dataset_id}`;
  const dataPart = dataset?.path ? flagArg('--data', dataset.path) : `--data <path to ${dataPlaceholder}>`;

  const valParts = ['folo val', modelPart, dataPart];
  if (evalOn?.split === 'train') valParts.push('--split train');

  const predictParts = ['folo predict', modelPart, '<images>'];

  return {
    val: valParts.join(' '),
    predict: predictParts.join(' '),
    modelPlaceholder,
    dataPlaceholder,
  };
}

/** Shared export menu items for Run detail (`⋯` menu and Per-class panel header) when epochs are loaded. */
export function runExportMenuItems(
  run: Run,
  getEpochs: () => readonly Epoch[],
  getClasses: () => readonly string[],
  getDatasetName: () => string | undefined,
  hasPerClass: boolean,
): MenuItem[] {
  const items: MenuItem[] = [
    {
      label: 'Export metrics (CSV)',
      icon: Download,
      onclick: () => void downloadRunMetricsCsv(run, getEpochs()),
    },
  ];
  if (hasPerClass) {
    items.push({
      label: 'Export per-class AP (CSV)',
      icon: Download,
      onclick: () => void downloadRunPerClassCsv(run, getEpochs(), getClasses()),
    });
  }
  items.push({
    label: 'Export metrics (JSON)',
    icon: FileJson,
    onclick: () => void downloadRunMetricsJson(run, getEpochs(), getClasses(), getDatasetName()),
  });
  return items;
}
