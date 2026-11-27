// Reactive session controller for Playground multi-image batch drop, filmstrip navigation, sequential Run All,
// abort-safe cancellation (on Stop, model/backend/threshold change, mode switch, or unmount), and object-URL ownership.
import { computed, effect, onCleanup, onWindow, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Model, Prediction } from '$lib/api/types';
import {
  DEFAULT_INFERENCE_CONFIG,
  inferenceConfigKey,
  inferenceMetaOf,
  isDetectorLoadError,
  type BrowserEngine,
  type InferenceConfig,
} from '$lib/inference/browser';
import { isOom } from '$lib/oom';
import { toast } from '$lib/ui/toast';
import { isPredictionCutAt, markCompletionFailed, markPredictionComplete, needsCompletionAt } from './allCandidates';
import {
  batchItemSource,
  batchPredKey,
  batchQaCounts,
  exportBatchCsv,
  exportBatchJson,
  exportBatchSummaryCsv,
  filterBatchBoxes,
  filterBatchItems,
  summarizeBatch,
  type BatchFilter,
  type BatchItem,
  type BatchQaFilter,
} from './batch';
import { takeImage } from './intake';
import { ALL, ask, FETCH, noServerWeights, type RunOn } from './predict';
import { sourceImageKey, type Source } from './source';

export interface UseBatchSessionInput {
  engine: BrowserEngine;
  model: () => Model | null;
  modelId: () => string | null;
  runOn: Signal<RunOn>;
  /** Effective inference enhancement config (standard / SAHI / TTA) for Model A's backend. */
  inference?: () => InferenceConfig;
  localWeights: Signal<Map<string, File>>;
  compare: Signal<boolean>;
  o2o: () => boolean;
  conf: Signal<number>;
  iou: Signal<number>;
  maxDet: Signal<number>;
  hidden: Signal<Set<number>>;
  classes: () => string[];
  source: Signal<Source | null>;
  pred: Signal<Prediction | null>;
  predSourceKey: Signal<string | null>;
  predicting: Signal<boolean>;
  error: Signal<unknown>;
  weightsMissing: Signal<boolean>;
  hover: Signal<Prediction['boxes'][number] | null>;
  picked: Signal<Prediction['boxes'][number] | null>;
  /** Shared source-choice token factory from `usePlaygroundSource` so batch and still-image decodes race cleanly. */
  choose: () => () => boolean;
  /** Synchronously aborts any in-flight single-image prediction before `Run all` dispatches batch work. */
  cancelSinglePredict: () => void;
  /** Called when a multi-file selection yields only 1 valid image (falls back cleanly to single-image mode). */
  onSingleFile: (source: Extract<Source, { kind: 'file' }>) => void;
  /** Called when activating a batch image on Playground's stage without tearing down the batch. */
  onActivateFrame: (source: Extract<Source, { kind: 'file' }>, cached: Prediction | null, bumpChoice?: boolean) => void;
  /** Called when entering batch mode to tear down any active video/webcam session. */
  onEnterBatch: () => void;
  /** Drops an unreadable local .folo weights file for Model A. */
  dropLocal: (id: string, f: File, e: unknown) => void;
}

const ARROW_BLOCKED_SELECTOR =
  'input, textarea, select, [contenteditable], [role="slider"], [role="radiogroup"], [role="tablist"], [role="menu"], [role="listbox"], tbody, [data-media-bar]';

let nextBatchId = 0;

function errorText(e: unknown): string {
  if (e instanceof Error && e.message) return e.message;
  return 'Prediction failed';
}

export function useBatchSession(o: UseBatchSessionInput) {
  const items = signal<BatchItem[]>([]);
  const activeIndex = signal(0);
  const runningAll = signal(false);
  const runningIndex = signal<number | null>(null);
  const qaFilter = signal<BatchQaFilter>({ kind: 'all' });

  const active = computed(() => items.get().length >= 2);
  const filter = computed<BatchFilter>(() => ({
    conf: o.conf.get(),
    iou: o.iou.get(),
    maxDet: o.maxDet.get(),
    o2o: o.o2o(),
  }));
  const summary = computed(() => summarizeBatch(items.get(), filter.get(), o.hidden.get()));
  const qaCounts = computed(() => batchQaCounts(items.get(), filter.get(), o.hidden.get(), o.classes()));
  const visibleEntries = computed(() => filterBatchItems(items.get(), qaFilter.get(), filter.get(), o.hidden.get()));
  const exportableCount = computed(() =>
    visibleEntries.get().reduce((n, entry) => n + (entry.item.status === 'done' && entry.item.pred ? 1 : 0), 0),
  );
  const summaryExportableCount = computed(() =>
    visibleEntries
      .get()
      .reduce((n, entry) => n + ((entry.item.status === 'done' && entry.item.pred) || entry.item.status === 'error' ? 1 : 0), 0),
  );

  const activeLocalFile = computed(() => {
    const on = o.runOn.get();
    const id = o.modelId();
    if (on === 'server' || !id) return null;
    return o.localWeights.get().get(id) ?? null;
  });

  let destroyed = false;
  let runSeq = 0;
  let runCtrl: AbortController | null = null;
  /** Source key of the active batch item after Stop or a Run All halt/error (suppresses immediate auto-restart). */
  let stoppedKey: string | null = null;
  let exportCtrl: AbortController | null = null;

  function revokeAll(list: readonly BatchItem[]) {
    for (const item of list) URL.revokeObjectURL(item.url);
  }

  function updateItem(index: number, patch: Partial<BatchItem>) {
    items.set((cur) => {
      if (index < 0 || index >= cur.length) return cur;
      const next = cur.slice();
      next[index] = { ...cur[index]!, ...patch };
      return next;
    });
  }

  function stopRunAll(markStopped = true) {
    const wasRunning = runningAll.peek();
    runSeq++;
    runCtrl?.abort();
    runCtrl = null;
    exportCtrl?.abort();
    exportCtrl = null;
    runningAll.set(false);
    runningIndex.set(null);
    items.set((cur) =>
      cur.some((it) => it.status === 'running')
        ? cur.map((it) => (it.status === 'running' ? { ...it, status: it.pred ? 'done' : 'idle' } : it))
        : cur,
    );
    if (wasRunning && markStopped) {
      const s = o.source.peek();
      stoppedKey = s ? sourceImageKey(s) : null;
      o.predicting.set(false);
    }
  }

  function clear() {
    stoppedKey = null;
    stopRunAll(false);
    qaFilter.set({ kind: 'all' });
    const prev = items.peek();
    if (prev.length) {
      items.set([]);
      activeIndex.set(0);
      revokeAll(prev);
    }
  }

  /** Invalidates all cached batch predictions (called when model, backend, or local weights change). */
  function invalidatePredictions() {
    stoppedKey = null;
    stopRunAll(false);
    qaFilter.set({ kind: 'all' });
    items.set((cur) =>
      cur.length ? cur.map((it) => ({ ...it, pred: null, predKey: null, status: 'idle', error: null })) : cur,
    );
  }

  function currentInference(): InferenceConfig {
    return o.inference ? o.inference() : DEFAULT_INFERENCE_CONFIG;
  }

  function currentKey(): string | null {
    const id = o.modelId();
    return id ? batchPredKey(id, o.runOn.peek(), inferenceConfigKey(untrack(currentInference))) : null;
  }

  function selectIndex(index: number) {
    const list = items.peek();
    if (list.length < 2) return;
    const clamped = Math.max(0, Math.min(list.length - 1, index));
    const cur = o.source.peek();
    if (clamped === activeIndex.peek() && cur?.kind === 'file' && cur.url === list[clamped]!.url) {
      return;
    }
    stoppedKey = null;
    const item = list[clamped]!;
    activeIndex.set(clamped);
    const key = currentKey();
    const cached = item.pred && key && item.predKey === key ? item.pred : null;
    o.onActivateFrame(batchItemSource(item), cached);
  }

  function setQaFilter(next: BatchQaFilter) {
    exportCtrl?.abort();
    exportCtrl = null;
    qaFilter.set(next);
    if (next.kind === 'all') return;
    const vis = visibleEntries.peek();
    if (vis.length > 0 && !vis.some((e) => e.index === activeIndex.peek())) {
      selectIndex(vis[0]!.index);
    }
  }

  function step(delta: -1 | 1) {
    const list = items.peek();
    if (list.length < 2) return;
    if (qaFilter.peek().kind === 'all') {
      const next = Math.max(0, Math.min(list.length - 1, activeIndex.peek() + delta));
      if (next !== activeIndex.peek()) selectIndex(next);
      return;
    }
    const vis = visibleEntries.peek();
    if (vis.length === 0) return;
    const pos = vis.findIndex((e) => e.index === activeIndex.peek());
    if (pos < 0) {
      selectIndex(delta > 0 ? vis[0]!.index : vis[vis.length - 1]!.index);
      return;
    }
    const nextPos = Math.max(0, Math.min(vis.length - 1, pos + delta));
    if (nextPos !== pos) selectIndex(vis[nextPos]!.index);
  }

  /**
   * Decodes 2+ dropped/picked image files into a filmstrip batch using the shared `choose()` source-choice token.
   * If a newer single-image/dataset/batch choice supersedes this load, all created object URLs are revoked cleanly.
   */
  async function loadFiles(files: readonly File[]) {
    if (!files.length) return;
    const isCurrent = o.choose();
    const decoded: BatchItem[] = [];
    const failed: string[] = [];

    for (const file of files) {
      if (destroyed || !isCurrent()) {
        revokeAll(decoded);
        return;
      }
      const res = await takeImage(file, file.name);
      if (destroyed || !isCurrent()) {
        revokeAll(decoded);
        return;
      }
      if ('error' in res) {
        failed.push(res.error);
        continue;
      }
      decoded.push({
        id: `batch_${++nextBatchId}`,
        file: res.file,
        name: res.name,
        url: URL.createObjectURL(res.file),
        width: res.width,
        height: res.height,
        pred: null,
        predKey: null,
        status: 'idle',
        error: null,
      });
    }

    if (destroyed || !isCurrent()) {
      revokeAll(decoded);
      return;
    }

    if (decoded.length === 0) {
      toast.error(new Error(failed[0] ?? 'None of the selected files could be opened as images.'), 'Unsupported image', { key: 'image' });
      return;
    }

    if (failed.length) {
      toast.info(
        `Skipped ${failed.length} unsupported ${failed.length === 1 ? 'file' : 'files'}`,
        failed[0]!,
      );
    }

    if (decoded.length === 1) {
      const only = decoded[0]!;
      clear();
      o.onSingleFile({
        kind: 'file',
        file: only.file,
        name: only.name,
        url: only.url,
        width: only.width,
        height: only.height,
      });
      return;
    }

    stopRunAll(false);
    stoppedKey = null;
    qaFilter.set({ kind: 'all' });
    const prev = items.peek();
    o.onEnterBatch();
    items.set(decoded);
    activeIndex.set(0);
    if (prev.length) revokeAll(prev);
    o.onActivateFrame(batchItemSource(decoded[0]!), null, false);
  }

  /** True when the user stopped Run All on this unrun source (suppresses immediate re-fetch until they navigate or re-run). */
  function wasStoppedFor(s: Source): boolean {
    return stoppedKey !== null && stoppedKey === sourceImageKey(s);
  }

  /** Returns a valid cached prediction for `source` when navigating back to an already-inferred batch item. */
  function getCachedPred(modelId: string, s: Source, runOn: RunOn, inferenceKey = 'standard'): Prediction | null {
    if (s.kind !== 'file' || !s.sharedUrl) return null;
    const list = items.peek();
    const item = list.find((it) => it.file === s.file);
    if (!item || !item.pred) return null;
    return item.predKey === batchPredKey(modelId, runOn, inferenceKey) ? item.pred : null;
  }

  /** Mirrors single-image inference completions/errors onto the matching batch item so filmstrip badges stay live. */
  function recordSingleResult(s: Source, modelId: string, runOn: RunOn, pred: Prediction | null, err: unknown, inferenceKey = 'standard') {
    if (s.kind !== 'file' || !s.sharedUrl) return;
    const idx = items.peek().findIndex((it) => it.file === s.file);
    if (idx < 0) return;
    if (pred) {
      updateItem(idx, { pred, predKey: batchPredKey(modelId, runOn, inferenceKey), status: 'done', error: null });
    } else if (err) {
      updateItem(idx, { pred: null, predKey: null, status: 'error', error: errorText(err) });
    }
  }

  /** Mirrors single-image predicting state onto the active batch item when not in Run All. */
  function recordSingleBusy(s: Source, busy: boolean) {
    if (s.kind !== 'file' || !s.sharedUrl || runningAll.peek()) return;
    const list = items.peek();
    const idx = list.findIndex((it) => it.file === s.file);
    if (idx < 0) return;
    const cur = list[idx]!;
    if (busy && cur.status !== 'done') {
      updateItem(idx, { status: 'running', error: null });
    } else if (!busy && cur.status === 'running') {
      updateItem(idx, { status: cur.pred ? 'done' : 'idle' });
    }
  }

  /** Fetches full candidate set for a cut prediction (`FETCH.max_det` hit above current `conf`). */
  async function completeIfCut(
    itemSource: Extract<Source, { kind: 'file' }>,
    p: Prediction,
    id: string,
    on: RunOn,
    localFile: File | null,
    inference: InferenceConfig,
    signal: AbortSignal,
  ): Promise<{ pred: Prediction; haltError?: unknown }> {
    if (!needsCompletionAt(p, o.conf.peek())) return { pred: p };
    try {
      const all = await ask({
        api: hub.api,
        engine: o.engine,
        model: id,
        source: itemSource,
        on,
        file: localFile,
        options: ALL,
        inference,
        signal,
      });
      if (signal.aborted) return { pred: p };
      const next = { ...all, timing: p.timing, cold: p.cold, ...(p.inference ? { inference: p.inference } : {}) };
      markPredictionComplete(next);
      return { pred: next };
    } catch (e: unknown) {
      if (signal.aborted) return { pred: p };
      markCompletionFailed(p);
      if (isOom(e) || (on !== 'server' && !localFile && noServerWeights(e)) || (localFile && on !== 'server' && isDetectorLoadError(e))) {
        return { pred: p, haltError: e };
      }
      return { pred: p };
    }
  }

  /** Sequentially runs inference on every pending/failed batch item with full abort safety. */
  async function runAll() {
    const id = o.modelId();
    const list = items.peek();
    if (!id || list.length < 2 || runningAll.peek()) return;

    const on = o.runOn.peek();
    const inf = untrack(currentInference);
    const key = batchPredKey(id, on, inferenceConfigKey(inf));
    const localFile = on === 'server' ? null : (o.localWeights.peek().get(id) ?? null);

    // If all items are already complete for this model/backend, nothing to run.
    if (list.every((it) => it.status === 'done' && it.pred && it.predKey === key)) return;

    // Cancel any in-flight single-image or export completion request synchronously before dispatching batch work.
    o.cancelSinglePredict();
    exportCtrl?.abort();
    exportCtrl = null;
    items.set((cur) =>
      cur.some((it) => it.status === 'running')
        ? cur.map((it) => (it.status === 'running' ? { ...it, status: it.pred ? 'done' : 'idle' } : it))
        : cur,
    );
    o.predicting.set(false);
    o.weightsMissing.set(false);
    o.error.set(null);

    stoppedKey = null;
    const seq = ++runSeq;
    runCtrl?.abort();
    const ctrl = (runCtrl = new AbortController());
    runningAll.set(true);
    let haltedOnNotice = false;

    try {
      for (let i = 0; i < items.peek().length; i++) {
        if (destroyed || seq !== runSeq || ctrl.signal.aborted) break;
        const item = items.peek()[i];
        if (!item) break;
        if (item.status === 'done' && item.pred && item.predKey === key) continue;

        runningIndex.set(i);
        updateItem(i, { status: 'running', error: null });
        if (i === activeIndex.peek()) {
          o.predicting.set(true);
          o.error.set(null);
          o.weightsMissing.set(false);
        }

        const itemSource = batchItemSource(item);
        const frameKey = sourceImageKey(itemSource);
        try {
          const raw = await ask({
            api: hub.api,
            engine: o.engine,
            model: id,
            source: itemSource,
            on,
            file: localFile,
            options: FETCH,
            inference: inf,
            signal: ctrl.signal,
          });
          if (destroyed || seq !== runSeq || ctrl.signal.aborted) break;
          const completed = await completeIfCut(itemSource, raw, id, on, localFile, inf, ctrl.signal);
          if (destroyed || seq !== runSeq || ctrl.signal.aborted) break;
          const p = completed.pred;
          updateItem(i, { pred: p, predKey: key, status: 'done', error: null });
          if (i === activeIndex.peek()) {
            o.pred.set(p);
            o.predSourceKey.set(frameKey);
            o.hover.set(null);
            o.picked.set(null);
            o.error.set(null);
            o.weightsMissing.set(false);
            o.predicting.set(false);
          }
          if (completed.haltError) {
            haltedOnNotice = true;
            const he = completed.haltError;
            if (localFile && on !== 'server' && isDetectorLoadError(he)) {
              o.dropLocal(id, localFile, he);
            } else if (on !== 'server' && !localFile && noServerWeights(he)) {
              o.weightsMissing.set(true);
            } else if (isOom(he)) {
              o.error.set(he);
              toast.error(he, 'Batch stopped: out of GPU memory');
            }
            break;
          }
        } catch (e: unknown) {
          if (destroyed || seq !== runSeq || ctrl.signal.aborted) break;
          if (localFile && on !== 'server' && isDetectorLoadError(e)) {
            updateItem(i, { status: 'idle', error: null });
            haltedOnNotice = true;
            if (i === activeIndex.peek()) {
              o.predicting.set(false);
              o.pred.set(null);
              o.predSourceKey.set(null);
            }
            o.dropLocal(id, localFile, e);
            break;
          }
          if (on !== 'server' && !localFile && noServerWeights(e)) {
            updateItem(i, { status: 'idle', error: null });
            haltedOnNotice = true;
            o.weightsMissing.set(true);
            if (i === activeIndex.peek()) {
              o.predicting.set(false);
              o.pred.set(null);
              o.predSourceKey.set(null);
            }
            break;
          }
          updateItem(i, { pred: null, predKey: null, status: 'error', error: errorText(e) });
          if (i === activeIndex.peek()) {
            o.predicting.set(false);
            o.pred.set(null);
            o.predSourceKey.set(null);
            o.error.set(e);
          }
          if (isOom(e)) {
            haltedOnNotice = true;
            o.error.set(e);
            if (i !== activeIndex.peek()) {
              toast.error(e, 'Batch stopped: out of GPU memory');
            }
            break;
          }
        }
      }
    } finally {
      if (seq === runSeq) {
        runCtrl = null;
        const activeItem = items.peek()[activeIndex.peek()];
        const curSrc = o.source.peek();
        if (curSrc && (haltedOnNotice || (activeItem?.status === 'error' && !!o.error.peek()))) {
          stoppedKey = sourceImageKey(curSrc);
        }
        runningAll.set(false);
        runningIndex.set(null);
        items.set((cur) =>
          cur.some((it) => it.status === 'running')
            ? cur.map((it) => (it.status === 'running' ? { ...it, status: it.pred ? 'done' : 'idle' } : it))
            : cur,
        );
        if (!destroyed) o.predicting.set(false);
      }
    }
  }

  function detectionCountFor(item: BatchItem): { count: number; cut: boolean } | null {
    if (item.status !== 'done' || !item.pred) return null;
    const f = filter.get();
    return {
      count: filterBatchBoxes(item.pred, f, o.hidden.get()).length,
      cut: isPredictionCutAt(item.pred, f.conf),
    };
  }

  /**
   * Completes any capped batch predictions before exporting; captures click-time QA filter, thresholds, and hidden
   * classes so mid-flight changes never silently export a different subset. When a QA filter is active, every completed
   * capped item is completed first so newly revealed matches are never omitted.
   */
  function runExport(kind: 'json' | 'csv' | 'summary-csv') {
    exportCtrl?.abort();
    exportCtrl = null;
    const snap = {
      modelId: o.modelId(),
      modelName: o.model()?.name ?? null,
      runOn: o.runOn.peek(),
      inf: untrack(currentInference),
      filter: { ...filter.peek() },
      hidden: new Set(o.hidden.peek()),
      classes: [...untrack(o.classes)],
      qaFilter: qaFilter.peek(),
    };
    const key = snap.modelId ? batchPredKey(snap.modelId, snap.runOn, inferenceConfigKey(snap.inf)) : null;

    const buildInput = () => {
      const all = items.peek();
      const exportItems =
        snap.qaFilter.kind === 'all'
          ? all
          : filterBatchItems(all, snap.qaFilter, snap.filter, snap.hidden).map((e) => e.item);
      const provisionalFilter =
        snap.qaFilter.kind !== 'all' &&
        all.some((it) => it.status === 'done' && isPredictionCutAt(it.pred, snap.filter.conf));
      return {
        items: exportItems,
        modelId: snap.modelId,
        modelName: snap.modelName,
        runOn: snap.runOn,
        inference: inferenceMetaOf(snap.inf),
        filter: snap.filter,
        hidden: snap.hidden,
        classes: snap.classes,
        qaFilter: snap.qaFilter,
        ...(provisionalFilter ? { provisionalFilter: true } : {}),
      };
    };

    const doExport = () => {
      const input = buildInput();
      if (kind === 'json') return exportBatchJson(input);
      if (kind === 'summary-csv') return exportBatchSummaryCsv(input);
      return exportBatchCsv(input);
    };

    // Complete every eligible capped item in the batch so filtered exports never miss boxes hidden past FETCH.max_det.
    const cutIndices = items
      .peek()
      .map((item, index) => (item.status === 'done' && needsCompletionAt(item.pred, snap.filter.conf) ? index : -1))
      .filter((index) => index >= 0);

    if (cutIndices.length === 0) {
      doExport();
      return;
    }

    const localFile = snap.modelId && snap.runOn !== 'server' ? (o.localWeights.peek().get(snap.modelId) ?? null) : null;
    if (!snap.modelId || !key) {
      doExport();
      return;
    }

    const ctrl = (exportCtrl = new AbortController());
    void (async () => {
      for (const idx of cutIndices) {
        if (destroyed || ctrl.signal.aborted) return;
        const item = items.peek()[idx];
        if (!item?.pred || item.predKey !== key) continue;
        const itemSource = batchItemSource(item);
        const { pred: next } = await completeIfCut(itemSource, item.pred, snap.modelId!, snap.runOn, localFile, snap.inf, ctrl.signal);
        if (destroyed || ctrl.signal.aborted) return;
        if (next !== item.pred) {
          updateItem(idx, { pred: next });
          if (idx === activeIndex.peek() && o.pred.peek() === item.pred) {
            o.pred.set(next);
          }
        }
      }
      if (!destroyed && !ctrl.signal.aborted) doExport();
    })();
  }

  function onExportJson() {
    runExport('json');
  }

  function onExportCsv() {
    runExport('csv');
  }

  function onExportSummaryCsv() {
    runExport('summary-csv');
  }

  // Keep stage `predicting` synced with whether the currently selected thumbnail is the one running in `Run all`.
  effect(() => {
    if (!runningAll.get()) return;
    const curIdx = activeIndex.get();
    const runIdx = runningIndex.get();
    untrack(() => o.predicting.set(runIdx !== null && runIdx === curIdx));
  });

  // Abort and invalidate cached batch predictions whenever model, backend, inference mode, or local weights change.
  let lastModelKey: string | null = null;
  let lastLocalFile: File | null = null;
  effect(() => {
    const id = o.modelId();
    const on = o.runOn.get();
    const infKey = inferenceConfigKey(currentInference());
    const f = activeLocalFile.get();
    const nextKey = id ? batchPredKey(id, on, infKey) : null;
    if (lastModelKey !== null && (nextKey !== lastModelKey || f !== lastLocalFile)) {
      untrack(() => {
        if (items.peek().length) invalidatePredictions();
      });
    }
    lastModelKey = nextKey;
    lastLocalFile = f;
  });

  // Abort any in-flight Run All cleanly when thresholds or Compare A/B mode change (cached raw predictions stay intact).
  let lastThreshKey: string | null = null;
  effect(() => {
    const k = `${o.conf.get()}|${o.iou.get()}|${o.maxDet.get()}`;
    if (lastThreshKey !== null && k !== lastThreshKey && untrack(() => runningAll.peek())) {
      untrack(() => stopRunAll());
    }
    lastThreshKey = k;
  });

  let lastCompare: boolean | null = null;
  effect(() => {
    const c = o.compare.get();
    // Cancel in-flight batch without setting manual Stop marker so Model A resumes cleanly alongside Model B.
    if (lastCompare !== null && c !== lastCompare && untrack(() => runningAll.peek())) {
      untrack(() => stopRunAll(false));
    }
    lastCompare = c;
  });

  // Global Left/Right keyboard stepping when a batch is active and focus is not inside an interactive arrow-handling control.
  onWindow('keydown', (e: KeyboardEvent) => {
    if (e.defaultPrevented || !active.peek()) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    if (document.querySelector('dialog[open]')) return;
    const el = e.target as Element | null;
    if (el?.closest?.(ARROW_BLOCKED_SELECTOR)) return;
    e.preventDefault();
    step(e.key === 'ArrowLeft' ? -1 : 1);
  });

  onCleanup(() => {
    destroyed = true;
    clear();
  });

  return {
    items,
    activeIndex,
    active,
    runningAll,
    runningIndex,
    summary,
    qaFilter,
    qaCounts,
    visibleEntries,
    exportableCount,
    summaryExportableCount,
    classes: o.classes,
    loadFiles,
    selectIndex,
    setQaFilter,
    step,
    runAll,
    stopRunAll,
    clear,
    getCachedPred,
    wasStoppedFor,
    recordSingleResult,
    recordSingleBusy,
    detectionCountFor,
    onExportJson,
    onExportCsv,
    onExportSummaryCsv,
  };
}

export type BatchSession = ReturnType<typeof useBatchSession>;
