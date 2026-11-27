// Reactive controller for Playground Model A/B Visual Compare state.
// Strictly inert when `compare` is false (zero network calls or browser inference work for Model B).
import { computed, effect, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Model, Prediction } from '$lib/api/types';
import { filterDetections } from '$lib/detect/nms';
import { isDetectorLoadError, type BrowserEngine } from '$lib/inference/browser';
import type { ModelChoice } from '$lib/scoreSummary';
import { askAllWhenCut } from './allCandidates';
import { comparePredictionJson, pickDefaultModelB, summarizeComparison, type CompareViewMode } from './compare';
import { ask, FETCH, noServerWeights, type RunOn } from './predict';
import { sourceImageKey, type Source } from './source';
import { classCounts, detectionIndex, visibleClasses, visibleRows, type Det } from './viewState';
import { createZoomState, resetZoomState } from './zoomState';

export interface CompareControllerInput {
  ranked: () => Model[];
  choices: () => ModelChoice<Model>[];
  modelA: () => Model | null;
  modelAId: () => string | null;
  chooseModelA: (id: string) => void;
  runOnA: Signal<RunOn>;
  predA: Signal<Prediction | null>;
  hoverA: Signal<Det | null>;
  pickedA: Signal<Det | null>;
  resultA: () => { boxes: Det[]; ms: number } | null;
  rowsA: () => { b: Det; i: number }[];
  o2oA: () => boolean;
  source: Signal<Source | null>;
  conf: Signal<number>;
  iou: Signal<number>;
  maxDet: Signal<number>;
  hidden: Signal<Set<number>>;
  classesA: () => string[];
  browserOff: () => string | null;
  localWeights: Signal<Map<string, File>>;
  dropLocalB: (id: string, f: File, e: unknown) => void;
  badFileB: Signal<{ name: string; why: string | null } | null>;
  engine: BrowserEngine;
}

export function createCompareController(input: CompareControllerInput) {
  const compare = signal(false);
  const compareView = signal<CompareViewMode>('side-by-side');
  const splitPct = signal(50);
  const inspectSide = signal<'a' | 'b'>('a');
  const pickedModelB = signal<string | null>(null);
  const runOnB = signal<RunOn>('server');
  const predB = signal<Prediction | null>(null);
  const predictingB = signal(false);
  const errorB = signal<unknown>(null);
  const weightsMissingB = signal(false);
  const retryTickB = signal(0);
  const hoverB = signal<Det | null>(null);
  const pickedB = signal<Det | null>(null);
  const zoom = createZoomState();

  let reuseSwapA: string | null = null;
  let reuseSwapB: string | null = null;
  const swapKey = (id: string, s: Source, on: RunOn) => `${id}|${sourceImageKey(s)}|${on}`;

  const modelB = computed((): Model | null => {
    const list = input.ranked();
    if (!list.length) return null;
    const wanted = pickedModelB.get();
    if (wanted) {
      const found = list.find((m) => m.id === wanted);
      if (found) return found;
    }
    return pickDefaultModelB(input.modelA(), list);
  });
  const modelBId = computed(() => modelB.get()?.id ?? null);
  const o2oB = computed(() => modelB.get()?.head === 'o2o');
  const metaB = computed(() => input.choices().find((c) => c.model.id === modelBId.get())?.meta ?? null);

  // Subscribe only to Model B's effective local weights file so Model A updates never restart Model B.
  const activeFileB = computed(() => {
    const on = runOnB.get();
    const id = modelBId.get();
    if (on === 'server' || !id) return null;
    return input.localWeights.get().get(id) ?? null;
  });

  effect(() => {
    if (input.browserOff() && untrack(() => runOnB.get()) !== 'server') runOnB.set('server');
  });

  effect(() => {
    void modelBId.get();
    void runOnB.get();
    input.badFileB.set(null);
  });

  function toggleCompare(next?: boolean) {
    const on = next ?? !compare.peek();
    if (on === compare.peek()) return;
    if (on) {
      runOnB.set(input.runOnA.peek());
      if (!pickedModelB.peek()) {
        const def = pickDefaultModelB(input.modelA(), input.ranked());
        if (def) pickedModelB.set(def.id);
      }
      compare.set(true);
    } else {
      compare.set(false);
      inspectSide.set('a');
      hoverB.set(null);
      pickedB.set(null);
      predictingB.set(false);
    }
  }

  function swapModels() {
    const aId = input.modelAId();
    const bId = modelBId.peek();
    if (!aId || !bId) return;
    const s = input.source.peek();
    const prevRunA = input.runOnA.peek();
    const prevRunB = runOnB.peek();
    const prevPredA = input.predA.peek();
    const prevPredB = predB.peek();

    // Reuse already-fetched predictions for the swapped sides so Swap does not fire 2 redundant requests.
    if (s && prevPredB && prevPredB.model_id === bId && (prevPredB.backend ?? 'server') === prevRunB) {
      reuseSwapA = swapKey(bId, s, prevRunB);
    }
    if (s && prevPredA && prevPredA.model_id === aId && (prevPredA.backend ?? 'server') === prevRunA) {
      reuseSwapB = swapKey(aId, s, prevRunA);
    }

    // Set Model B before choosing Model A so immediate URL sync writes both swapped IDs atomically.
    pickedModelB.set(aId);
    input.runOnA.set(prevRunB);
    runOnB.set(prevRunA);
    input.predA.set(prevPredB);
    predB.set(prevPredA);
    input.hoverA.set(null);
    input.pickedA.set(null);
    hoverB.set(null);
    pickedB.set(null);
    input.chooseModelA(bId);
  }

  function consumeReuseA(id: string, s: Source, on: RunOn): boolean {
    const match = reuseSwapA !== null && reuseSwapA === swapKey(id, s, on);
    reuseSwapA = null;
    return match;
  }

  function clearResults() {
    reuseSwapA = null;
    reuseSwapB = null;
    predB.set(null);
    errorB.set(null);
    weightsMissingB.set(false);
    hoverB.set(null);
    pickedB.set(null);
    resetZoomState(zoom, false);
  }

  // Model B prediction effect: strictly skipped when Compare mode is OFF (zero extra network calls).
  effect(() => {
    const onCompare = compare.get();
    const id = modelBId.get();
    const s = input.source.get();
    const on = runOnB.get();
    const file = activeFileB.get();
    void retryTickB.get();
    if (!onCompare || !id || !s) {
      untrack(() => {
        predictingB.set(false);
        if (!onCompare) {
          reuseSwapB = null;
          errorB.set(null);
          weightsMissingB.set(false);
          inspectSide.set('a');
          hoverB.set(null);
          pickedB.set(null);
        }
      });
      return;
    }
    const reused = untrack(() => {
      const hit = reuseSwapB !== null && reuseSwapB === swapKey(id, s, on);
      reuseSwapB = null;
      return hit;
    });
    if (reused) {
      untrack(() => {
        predictingB.set(false);
        errorB.set(null);
        weightsMissingB.set(false);
      });
      return;
    }
    const ctrl = new AbortController();
    untrack(() => {
      predictingB.set(true);
      errorB.set(null);
      weightsMissingB.set(false);
      const p = predB.get();
      if (p && (p.model_id !== id || (p.backend ?? 'server') !== on)) {
        predB.set(null);
        hoverB.set(null);
        pickedB.set(null);
      }
    });
    ask({ api: hub.api, engine: input.engine, model: id, source: s, on, file, options: FETCH, signal: ctrl.signal }).then(
      (p) => {
        if (ctrl.signal.aborted) return;
        predB.set(p);
        hoverB.set(null);
        pickedB.set(null);
        predictingB.set(false);
      },
      (e: unknown) => {
        if (ctrl.signal.aborted) return;
        predictingB.set(false);
        if (file && on !== 'server' && isDetectorLoadError(e)) {
          predB.set(null);
          input.dropLocalB(id, file, e);
          return;
        }
        if (on !== 'server' && !file && noServerWeights(e)) {
          predB.set(null);
          weightsMissingB.set(true);
          return;
        }
        predB.set(null);
        errorB.set(e);
      },
    );
    return () => ctrl.abort();
  });

  // Complete cut candidate list for Model B only while Compare mode is active.
  askAllWhenCut({
    pred: predB,
    conf: () => input.conf.get(),
    model: () => (compare.get() ? modelBId.get() : null),
    source: () => (compare.get() ? input.source.get() : null),
    runOn: () => runOnB.get(),
    file: () => activeFileB.peek(),
    busy: () => predictingB.get(),
    engine: input.engine,
    weightsMissing: () => weightsMissingB.set(true),
  });

  const resultB = computed(() => {
    const p = predB.get();
    if (!p) return null;
    const t0 = performance.now();
    const boxes = filterDetections(p.boxes, { conf: input.conf.get(), iou: o2oB.get() ? 1 : input.iou.get(), maxDet: input.maxDet.get() });
    return { boxes, ms: performance.now() - t0 };
  });
  const keptB = computed(() => resultB.get()?.boxes ?? []);
  const classesB = computed(() => modelB.get()?.classes ?? []);

  // Map hidden class names from Model A onto Model B's class indices so hiding "circle" hides it on both sides.
  const hiddenB = computed(() => {
    const h = input.hidden.get();
    if (!h.size) return new Set<number>();
    const clsA = input.classesA();
    const hiddenNames = new Set([...h].map((i) => clsA[i] ?? `class ${i}`));
    const clsB = classesB.get();
    const out = new Set<number>();
    clsB.forEach((name, idx) => {
      if (hiddenNames.has(name)) out.add(idx);
    });
    return out;
  });

  const onlyB = computed(() => visibleClasses(classesB.get(), hiddenB.get()));
  const rowsB = computed(() => visibleRows(keptB.get(), hiddenB.get()));
  const countsB = computed(() => classCounts(keptB.get()));
  const maxCountB = computed(() => Math.max(1, ...countsB.get().map((c) => c[1])));
  const selectedIdxB = computed(() => detectionIndex(pickedB.get(), keptB.get(), hiddenB.get()));
  const activeIdxB = computed(() => detectionIndex(hoverB.get(), keptB.get(), hiddenB.get()) ?? selectedIdxB.get());

  const summary = computed(() =>
    summarizeComparison({
      boxesA: input.rowsA().map((r) => r.b),
      boxesB: rowsB.get().map((r) => r.b),
      predA: input.predA.get(),
      predB: predB.get(),
    }),
  );

  const compareJson = computed(() => {
    const h = input.hidden.get();
    const clsA = input.classesA();
    const hiddenClasses = h.size ? [...h].sort((a, b) => a - b).map((c) => clsA[c] ?? `class ${c}`) : undefined;
    return comparePredictionJson({
      view: compareView.get(),
      predA: input.predA.get(),
      predB: predB.get(),
      hasResultA: !!input.resultA(),
      hasResultB: !!resultB.get(),
      boxesA: input.rowsA().map((r) => r.b),
      boxesB: rowsB.get().map((r) => r.b),
      modelAName: input.modelA()?.name ?? input.modelAId() ?? 'Model A',
      modelBName: modelB.get()?.name ?? modelBId.get() ?? 'Model B',
      runOnA: input.runOnA.get(),
      runOnB: runOnB.get(),
      o2oA: input.o2oA(),
      o2oB: o2oB.get(),
      filter: {
        conf: input.conf.get(),
        iou: input.iou.get(),
        maxDet: input.maxDet.get(),
        hiddenClasses,
      },
      summary: summary.get(),
    });
  });

  return {
    compare,
    compareView,
    splitPct,
    inspectSide,
    pickedModelB,
    modelB,
    modelBId,
    o2oB,
    metaB,
    runOnB,
    predB,
    predictingB,
    errorB,
    weightsMissingB,
    retryTickB,
    hoverB,
    pickedB,
    zoom,
    resultB,
    keptB,
    classesB,
    hiddenB,
    onlyB,
    rowsB,
    countsB,
    maxCountB,
    selectedIdxB,
    activeIdxB,
    summary,
    compareJson,
    toggleCompare,
    swapModels,
    consumeReuseA,
    clearResults,
  };
}
