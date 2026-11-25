/**
 * Pure CSV / JSON payload builder for the Models list (`#/models`), respecting current group filter and sort order.
 */
import type { Model } from '$lib/api/types';
import type { ScoreContext } from '$lib/scoreSummary';
import type { CsvCell } from '$lib/tableExport';

export interface ExportModelGroup {
  key: string;
  label: string;
  ctx: ScoreContext;
  ranked: boolean;
  rows: readonly Model[];
}

export interface ModelsExportPayload {
  filenameBase: string;
  headers: string[];
  rows: CsvCell[][];
  json: unknown;
}

export function modelsExport(
  groups: readonly ExportModelGroup[],
  bestMarks: ReadonlyMap<string, string>,
  sourceName: (m: Model) => string,
): ModelsExportPayload {
  const headers = [
    'group',
    'dataset',
    'split',
    'ranked',
    'mark',
    'id',
    'name',
    'arch',
    'imgsz',
    'params',
    'classes',
    'bytes',
    'map50_95',
    'map50',
    'source_run',
    'created',
  ];
  const items: Array<{
    group: string;
    dataset: string | null;
    split: string | null;
    ranked: boolean;
    mark: string | null;
    id: string;
    name: string;
    arch: string;
    imgsz: number;
    params: number;
    classes: number;
    bytes: number;
    map50_95: number | null;
    map50: number | null;
    source_run: string | null;
    created: string;
  }> = [];
  const rows: CsvCell[][] = [];

  for (const g of groups) {
    for (const m of g.rows) {
      const mark = bestMarks.get(m.id) ?? null;
      const src = sourceName(m) || null;
      const item = {
        group: g.label,
        dataset: g.ctx.datasetName ?? null,
        split: g.ctx.split ?? null,
        ranked: g.ranked,
        mark,
        id: m.id,
        name: m.name,
        arch: m.arch,
        imgsz: m.imgsz,
        params: m.params,
        classes: m.classes.length,
        bytes: m.bytes,
        map50_95: m.metrics?.map50_95 ?? null,
        map50: m.metrics?.map50 ?? null,
        source_run: src,
        created: m.created,
      };
      items.push(item);
      rows.push([
        item.group,
        item.dataset,
        item.split,
        item.ranked,
        item.mark,
        item.id,
        item.name,
        item.arch,
        item.imgsz,
        item.params,
        item.classes,
        item.bytes,
        item.map50_95,
        item.map50,
        item.source_run,
        item.created,
      ]);
    }
  }

  return {
    filenameBase: 'folo-models',
    headers,
    rows,
    json: { count: items.length, groups: groups.length, models: items },
  };
}
