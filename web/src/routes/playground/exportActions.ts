// High-level export orchestrators for Playground: builds export filenames and runs annotated PNG or structured JSON
// downloads for both Single and Compare A/B modes with toast feedback.
import type { DatasetImage as Img, Model, Prediction } from '$lib/api/types';
import { toast } from '$lib/ui/toast';
import type { CompareViewMode } from './compare';
import { compareBadgeText, exportComparePng, exportDetectionsJson, exportSinglePng } from './export';
import type { Source } from './source';
import type { Det } from './viewState';

export interface RunPngExportInput {
  compare: boolean;
  compareView: CompareViewMode;
  splitPct: number;
  source: Source;
  src: string | null;
  dims: { width: number; height: number };
  shortName: string;
  modelA: Model | null;
  modelB: Model | null;
  predA: Prediction | null;
  predB: Prediction | null;
  boxesA: readonly Det[];
  boxesB: readonly Det[];
  classesA: readonly string[];
  classesB: readonly string[];
  truthBoxes: readonly Img['boxes'][number][];
  labels: boolean;
  scores: boolean;
}

export function buildExportBase(shortName: string, compare: boolean, modelAName?: string, modelBName?: string): string {
  const stem = shortName.replace(/\.[^.]+$/, '') || 'playground';
  if (compare) return `${stem}-${modelAName ?? 'a'}-vs-${modelBName ?? 'b'}`;
  return `${stem}-${modelAName ?? 'detections'}`;
}

export async function runPngExport(input: RunPngExportInput): Promise<boolean> {
  const filenameBase = buildExportBase(input.shortName, input.compare, input.modelA?.name, input.modelB?.name);
  try {
    const ok = input.compare
      ? await exportComparePng({
          source: input.source,
          src: input.src,
          width: input.dims.width,
          height: input.dims.height,
          view: input.compareView,
          splitPct: input.splitPct,
          layerA: {
            boxes: input.boxesA,
            truth: input.truthBoxes,
            classes: input.classesA,
            labels: input.labels,
            scores: input.scores,
            caption: compareBadgeText('A', input.modelA?.name ?? 'Model A', input.boxesA.length, input.predA?.timing.total_ms ?? null),
          },
          layerB: {
            boxes: input.boxesB,
            truth: input.truthBoxes,
            classes: input.classesB,
            labels: input.labels,
            scores: input.scores,
            caption: compareBadgeText('B', input.modelB?.name ?? 'Model B', input.boxesB.length, input.predB?.timing.total_ms ?? null),
          },
          filenameBase,
        })
      : await exportSinglePng({
          source: input.source,
          src: input.src,
          width: input.dims.width,
          height: input.dims.height,
          layer: {
            boxes: input.boxesA,
            truth: input.truthBoxes,
            classes: input.classesA,
            labels: input.labels,
            scores: input.scores,
          },
          filenameBase,
        });
    if (ok) toast.success('Annotated PNG exported', input.compare ? 'Saved Model A/B comparison image.' : 'Saved annotated detections image.');
    return ok;
  } catch (err) {
    toast.error(err instanceof Error ? err : new Error('Could not export annotated PNG.'), 'Export failed');
    return false;
  }
}

export function runJsonExport(opts: {
  shortName: string;
  compare: boolean;
  modelAName?: string;
  modelBName?: string;
  payload: unknown;
}): boolean {
  if (!opts.payload) return false;
  const filenameBase = buildExportBase(opts.shortName, opts.compare, opts.modelAName, opts.modelBName);
  const ok = exportDetectionsJson(filenameBase, opts.payload);
  if (ok) toast.success('JSON exported', opts.compare ? 'Saved Model A/B comparison JSON.' : 'Saved detections JSON.');
  return ok;
}
