// Frame-consistent PNG and JSON export controller for Playground (Single & Compare A/B modes).
// Ensures exports stay disabled while predictions are in flight or stale relative to the active frame
// (including scrubbed video previews before Detect frame) so exports never mix boxes from different frames.
import { computed, signal, type Signal } from 'fw';
import type { DatasetImage as Img, Model, Prediction } from '$lib/api/types';
import type { CompareViewMode } from './compare';
import { runJsonExport, runPngExport } from './exportActions';
import { sourceImageKey, type Source } from './source';
import type { TrackTrail } from './tracking';
import type { Det } from './viewState';
import type { ZoomState } from './zoomState';

export interface UsePlaygroundExportInput {
  source: Signal<Source | null>;
  src: Signal<string | null>;
  dims: () => { width: number; height: number } | null;
  shortName: () => string;
  modelA: () => Model | null;
  modelB: () => Model | null;
  predA: Signal<Prediction | null>;
  predB: Signal<Prediction | null>;
  predSourceKeyA: Signal<string | null>;
  predSourceKeyB: Signal<string | null>;
  resultA: () => { boxes: Det[]; ms: number } | null;
  resultB: () => { boxes: Det[]; ms: number } | null;
  rowsA: () => { b: Det; i: number }[];
  rowsB: () => { b: Det; i: number }[];
  trailsA?: () => readonly TrackTrail[] | undefined;
  classesA: () => string[];
  classesB: () => string[];
  truthOnly: () => Set<number> | null;
  truthBoxes: () => Img['boxes'];
  labels: Signal<boolean>;
  scores: Signal<boolean>;
  predictingA: Signal<boolean>;
  predictingB: Signal<boolean>;
  errorA: Signal<unknown>;
  errorB: Signal<unknown>;
  weightsMissingA: Signal<boolean>;
  weightsMissingB: Signal<boolean>;
  compare: Signal<boolean>;
  compareView: Signal<CompareViewMode>;
  splitPct: Signal<number>;
  zoom: ZoomState;
  activeJson: () => unknown;
  isUnsyncedVideoPreview: () => boolean;
  /** True while actively streaming webcam or playing auto-detected video (prevents export buttons strobing in the header). */
  isLiveStreaming: () => boolean;
}

export function usePlaygroundExport(o: UsePlaygroundExportInput) {
  const exportingPng = signal(false);

  const curSourceKey = computed(() => {
    const s = o.source.get();
    return s ? sourceImageKey(s) : null;
  });
  const predAMatches = computed(() => {
    const key = curSourceKey.get();
    return !!o.resultA() && key !== null && o.predSourceKeyA.get() === key;
  });
  const predBMatches = computed(() => {
    const key = curSourceKey.get();
    return !!o.resultB() && key !== null && o.predSourceKeyB.get() === key;
  });

  const canExport = computed(() => {
    const s = o.source.get();
    if (!s || !o.dims() || o.isLiveStreaming() || o.isUnsyncedVideoPreview() || o.predictingA.get()) return false;
    if (o.compare.get()) {
      if (o.predictingB.get()) return false;
      const aOk = predAMatches.get() || (!o.resultA() && (!!o.errorA.get() || o.weightsMissingA.get()));
      const bOk = predBMatches.get() || (!o.resultB() && (!!o.errorB.get() || o.weightsMissingB.get()));
      return aOk && bOk && (predAMatches.get() || predBMatches.get());
    }
    return predAMatches.get();
  });

  async function exportPngWithView(compareViewOverride?: CompareViewMode) {
    const s = o.source.peek();
    const d = o.dims();
    if (!s || !d || exportingPng.peek() || !canExport.peek()) return;
    const key = sourceImageKey(s);
    const hasA = o.predSourceKeyA.peek() === key && !!o.predA.peek();
    const hasB = o.predSourceKeyB.peek() === key && !!o.predB.peek();
    exportingPng.set(true);
    try {
      await runPngExport({
        compare: o.compare.peek(),
        compareView: compareViewOverride ?? o.compareView.peek(),
        splitPct: o.splitPct.peek(),
        viewportW: o.zoom.viewportW.peek(),
        viewportH: o.zoom.viewportH.peek(),
        zoomZ: o.zoom.z.peek(),
        zoomNx: o.zoom.nx.peek(),
        source: s,
        src: o.src.peek(),
        dims: d,
        shortName: o.shortName(),
        modelA: o.modelA(),
        modelB: o.modelB(),
        predA: hasA ? o.predA.peek() : null,
        predB: hasB ? o.predB.peek() : null,
        boxesA: hasA ? o.rowsA().map((r) => r.b) : [],
        boxesB: hasB ? o.rowsB().map((r) => r.b) : [],
        trailsA: hasA ? o.trailsA?.() : undefined,
        classesA: o.classesA(),
        classesB: o.classesB(),
        truthOnly: o.truthOnly(),
        truthBoxes: o.truthBoxes(),
        labels: o.labels.peek(),
        scores: o.scores.peek(),
      });
    } finally {
      exportingPng.set(false);
    }
  }

  function onExportPng() {
    return exportPngWithView();
  }

  function onExportCompareSideBySidePng() {
    return exportPngWithView('side-by-side');
  }

  function onExportJson() {
    if (!canExport.peek()) return;
    runJsonExport({
      shortName: o.shortName(),
      compare: o.compare.peek(),
      modelAName: o.modelA()?.name,
      modelBName: o.modelB()?.name,
      payload: o.activeJson(),
    });
  }

  return {
    exportingPng,
    predAMatches,
    predBMatches,
    canExport,
    onExportPng,
    onExportCompareSideBySidePng,
    onExportJson,
  };
}
