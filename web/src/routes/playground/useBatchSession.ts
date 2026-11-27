// Reactive session controller for Playground multi-image batch drop, filmstrip navigation, sequential Run All,
// abort-safe cancellation (on Stop, model/backend/threshold change, mode switch, or unmount), and object-URL ownership.
import { computed, effect, onCleanup, onWindow, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Model, Prediction } from '$lib/api/types';
import { isDetectorLoadError, type BrowserEngine } from '$lib/inference/browser';
import { isOom } from '$lib/oom';
import { toast } from '$lib/ui/toast';
import { isPredictionCutAt, markCompletionFailed, markPredictionComplete } from './allCandidates';
import {
  batchItemSource,
  batchPredKey,
  exportBatchCsv,
  exportBatchJson,
  filterBatchBoxes,
  summarizeBatch,
  type BatchFilter,
  type BatchItem,
} from './batch';
import { takeImage } from './intake';
import { ALL, ask, FETCH, noServerWeights, type RunOn } from './predict';
import { sourceImageKey, type Source } from './source';

export interface UseBatchSessionInput {
  engine: BrowserEngine;
  model: () => Model | null;
  modelId: () => string | null;
  runOn: Signal<RunOn>;
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

  const active = computed(() => items.get().length >= 2);
  const filter = computed<BatchFilter>(() => ({
    conf: o.conf.get(),
    iou: o.iou.get(),
    maxDet: o.maxDet.get(),
    o2o: o.o2o(),
  }));
  const summary = computed(() => summarizeBatch(items.get(), filter.get(), o.hidden.get()));

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
    items.set((cur) =>
      cur.length ? cur.map((it) => ({ ...it, pred: null, predKey: null, status: 'idle', error: null })) : cur,
    );
  }

  function currentKey(): string | null {
    const id = o.modelId();
    return id ? batchPredKey(id, o.runOn.peek()) : null;
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

  function step(delta: -1 | 1) {
    const list = items.peek();
    if (list.length < 2) return;
    const next = Math.max(0, Math.min(list.length - 1, activeIndex.peek() + delta));
    if (next !== activeIndex.peek()) selectIndex(next);
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
  function getCachedPred(modelId: string, s: Source, runOn: RunOn): Prediction | null {
    if (s.kind !== 'file' || !s.sharedUrl) return null;
    const list = items.peek();
    const item = list.find((it) => it.file === s.file);
    if (!item || !item.pred) return null;
    return item.predKey === batchPredKey(modelId, runOn) ? item.pred : null;
  }

  /** Mirrors single-image inference completions/errors onto the matching batch item so filmstrip badges stay live. */
  function recordSingleResult(s: Source, modelId: string, runOn: RunOn, pred: Prediction | null, err: unknown) {
    if (s.kind !== 'file' || !s.sharedUrl) return;
    const idx = items.peek().findIndex((it) => it.file === s.file);
    if (idx < 0) return;
    if (pred) {
      updateItem(idx, { pred, predKey: batchPredKey(modelId, runOn), status: 'done', error: null });
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
    signal: AbortSignal,
  ): Promise<{ pred: Prediction; haltError?: unknown }> {
    if (!isPredictionCutAt(p, o.conf.peek())) return { pred: p };
    try {
      const all = await ask({
        api: hub.api,
        engine: o.engine,
        model: id,
        source: itemSource,
        on,
        file: localFile,
        options: ALL,
        signal,
      });
      if (signal.aborted) return { pred: p };
      const next = { ...all, timing: p.timing, cold: p.cold };
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
    const key = batchPredKey(id, on);
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
            signal: ctrl.signal,
          });
          if (destroyed || seq !== runSeq || ctrl.signal.aborted) break;
          const completed = await completeIfCut(itemSource, raw, id, on, localFile, ctrl.signal);
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

  function makeExportInput() {
    return {
      items: items.peek(),
      modelId: o.modelId(),
      modelName: o.model()?.name ?? null,
      runOn: o.runOn.peek(),
      filter: filter.peek(),
      hidden: o.hidden.peek(),
      classes: o.classes(),
    };
  }

  /** Completes any cut batch predictions before exporting; runs synchronously when no item is cut at current conf. */
  function runExport(kind: 'json' | 'csv') {
    const confNow = o.conf.peek();
    const list = items.peek();
    const cutIndices = list
      .map((it, idx) => (it.status === 'done' && isPredictionCutAt(it.pred, confNow) ? idx : -1))
      .filter((idx) => idx >= 0);

    const doExport = () => (kind === 'json' ? exportBatchJson(makeExportInput()) : exportBatchCsv(makeExportInput()));
    if (cutIndices.length === 0) {
      doExport();
      return;
    }

    const id = o.modelId();
    const on = o.runOn.peek();
    const key = id ? batchPredKey(id, on) : null;
    const localFile = id && on !== 'server' ? (o.localWeights.peek().get(id) ?? null) : null;
    if (!id || !key) {
      doExport();
      return;
    }

    exportCtrl?.abort();
    const ctrl = (exportCtrl = new AbortController());
    void (async () => {
      for (const idx of cutIndices) {
        if (destroyed || ctrl.signal.aborted) return;
        const item = items.peek()[idx];
        if (!item?.pred || item.predKey !== key) continue;
        const itemSource = batchItemSource(item);
        const { pred: next } = await completeIfCut(itemSource, item.pred, id, on, localFile, ctrl.signal);
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

  // Keep stage `predicting` synced with whether the currently selected thumbnail is the one running in `Run all`.
  effect(() => {
    if (!runningAll.get()) return;
    const curIdx = activeIndex.get();
    const runIdx = runningIndex.get();
    untrack(() => o.predicting.set(runIdx !== null && runIdx === curIdx));
  });

  // Abort and invalidate cached batch predictions whenever model, backend, or local weights change.
  let lastModelKey: string | null = null;
  let lastLocalFile: File | null = null;
  effect(() => {
    const id = o.modelId();
    const on = o.runOn.get();
    const f = activeLocalFile.get();
    const nextKey = id ? batchPredKey(id, on) : null;
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
    loadFiles,
    selectIndex,
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
  };
}

export type BatchSession = ReturnType<typeof useBatchSession>;
