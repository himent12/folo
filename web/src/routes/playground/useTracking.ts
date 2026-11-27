// Reactive ByteTrack controller for Playground live video & webcam sessions.
// Holds one persistent `folo-wasm` `FoloTracker` instance per active media session, resets it on seek, source change,
// camera restart, model change, or toggle off, and always calls `free()` on exit/unmount (zero WASM memory leaks).
import { computed, effect, onCleanup, signal, untrack } from 'fw';
import type { PredictBox, Prediction } from '$lib/api/types';
import { filterDetections } from '$lib/detect/nms';
import type { BrowserEngine, Tracker } from '$lib/inference/browser';
import { trackerOptionsForConf, TrackTrailHistory, type TrackTrail } from './tracking';
import type { MediaMode } from './useMediaSession';

export interface UseTrackingInput {
  engine: BrowserEngine;
  browserOff: () => string | null;
  mediaMode: () => MediaMode;
  /** Stable key for the active clip or camera stream (changes on source switch / clip replacement). */
  mediaKey: () => string | null;
  modelId: () => string | null;
  classes: () => readonly string[];
  pred: () => Prediction | null;
  conf: () => number;
  iou: () => number;
  maxDet: () => number;
  o2o: () => boolean;
}

const NO_TRACKER_TIP = 'This folo-wasm build does not include FoloTracker. Rebuild folo-wasm to enable Track IDs.';
const NO_WASM_TIP = 'Track IDs requires the folo-wasm browser runtime.';

export function useTracking(input: UseTrackingInput) {
  const enabled = signal(false);
  const loading = signal(false);
  const activeCount = signal(0);
  const trails = signal<TrackTrail[]>([]);
  const trackedFrame = signal<{ pred: Prediction; boxes: PredictBox[] } | null>(null);
  const trackerReadyTick = signal(0);

  let tracker: Tracker | null = null;
  let trackerConf: number | null = null;
  let trackerSeq = 0;
  let alive = true;
  let lastProcessedPred: Prediction | null = null;
  const trailHistory = new TrackTrailHistory();

  const supported = computed(() => !input.browserOff() && input.engine.capabilities.tracker !== false);
  const disabledReason = computed(() => {
    if (input.browserOff()) return NO_WASM_TIP;
    if (input.engine.capabilities.tracker === false) return NO_TRACKER_TIP;
    return null;
  });

  function clearOutputs(ignoreCurrentPred = false) {
    trailHistory.clear();
    trails.set([]);
    activeCount.set(0);
    trackedFrame.set(null);
    lastProcessedPred = ignoreCurrentPred ? untrack(() => input.pred()) : null;
  }

  function freeTracker() {
    trackerSeq++;
    const prev = tracker;
    tracker = null;
    trackerConf = null;
    loading.set(false);
    if (prev) {
      try {
        prev.free();
      } catch {
        // Ignore already-freed WASM handles during teardown.
      }
    }
  }

  /** Resets the persistent tracker's tracks & motion trails without reallocating the WASM handle. */
  function reset() {
    clearOutputs(true);
    if (tracker) {
      try {
        tracker.reset();
      } catch {
        freeTracker();
      }
    }
  }

  async function ensureTracker(wantConf: number, classes: readonly string[]) {
    if (!alive) return;
    if (tracker && trackerConf !== null && Math.abs(trackerConf - wantConf) <= 1e-4) return;
    const seq = ++trackerSeq;
    // Free any prior handle before allocating a replacement (e.g. after confidence threshold change).
    if (tracker) {
      const old = tracker;
      tracker = null;
      trackerConf = null;
      try {
        old.free();
      } catch {
        // Ignore teardown errors.
      }
    }
    loading.set(true);
    try {
      const created = await input.engine.createTracker(trackerOptionsForConf(wantConf, classes));
      if (!alive || seq !== trackerSeq || !enabled.peek() || input.mediaMode() === 'none') {
        try {
          created?.free();
        } catch {
          // Ignore teardown errors.
        }
        return;
      }
      if (!created) {
        enabled.set(false);
        clearOutputs();
        return;
      }
      tracker = created;
      trackerConf = wantConf;
      trackerReadyTick.set((t) => t + 1);
    } catch {
      if (!alive || seq !== trackerSeq) return;
      enabled.set(false);
      clearOutputs();
    } finally {
      if (alive && seq === trackerSeq) loading.set(false);
    }
  }

  function setEnabled(next: boolean) {
    if (next && !supported.peek()) return;
    const prev = enabled.peek();
    if (prev === next) return;
    enabled.set(next);
    if (!next) {
      reset();
    } else {
      lastProcessedPred = null;
    }
  }

  // Free WASM handle when leaving video/webcam mode; reset when the clip/source key or active model changes.
  let lastMediaKey: string | null = null;
  let lastModelId: string | null = null;
  effect(() => {
    const mode = input.mediaMode();
    const key = input.mediaKey();
    const mid = input.modelId();
    untrack(() => {
      if (mode === 'none') {
        clearOutputs();
        freeTracker();
        lastMediaKey = null;
        return;
      }
      if (lastMediaKey !== null && key !== lastMediaKey) {
        reset();
      }
      if (lastModelId !== null && mid !== lastModelId) {
        reset();
      }
      lastMediaKey = key;
      lastModelId = mid;
    });
  });

  // Lazily construct or re-configure the persistent FoloTracker while tracking is enabled in media mode.
  effect(() => {
    const on = enabled.get();
    const mode = input.mediaMode();
    const c = input.conf();
    const cls = input.classes();
    if (!on || mode === 'none') return;
    untrack(() => {
      if (tracker && trackerConf !== null && Math.abs(trackerConf - c) > 1e-4) {
        clearOutputs();
      }
      void ensureTracker(c, cls);
    });
  });

  // Feed each new live prediction into FoloTracker exactly once (never double-stepping ByteTrack on UI re-renders).
  effect(() => {
    const on = enabled.get();
    const mode = input.mediaMode();
    const p = input.pred();
    const c = input.conf();
    const iou = input.iou();
    const maxDet = input.maxDet();
    const o2o = input.o2o();
    void trackerReadyTick.get();
    if (!on || mode === 'none' || !p) {
      untrack(() => {
        if (!p) trackedFrame.set(null);
      });
      return;
    }
    untrack(() => {
      if (!tracker || p === lastProcessedPred) return;
      lastProcessedPred = p;
      // Preserve ByteTrack's two-stage low-confidence occlusion recovery by keeping candidates down to `min(0.1, conf)`.
      const low = Math.min(0.1, Math.max(0.01, c));
      const candidates = filterDetections(p.boxes, { conf: low, iou: o2o ? 1 : iou, maxDet });
      const tracked = tracker.update(candidates);
      const boxes: PredictBox[] = tracked.map((t) => ({
        class: t.class,
        name: t.name,
        score: t.score,
        x1: t.x1,
        y1: t.y1,
        x2: t.x2,
        y2: t.y2,
        track_id: t.track_id,
      }));
      trackedFrame.set({ pred: p, boxes });
      trails.set(trailHistory.update(tracked));
      activeCount.set(tracked.length);
    });
  });

  onCleanup(() => {
    alive = false;
    clearOutputs();
    freeTracker();
  });

  /** Returns tracked boxes matching `p` when tracking is active for that frame, else null. */
  function boxesFor(p: Prediction | null): PredictBox[] | null {
    if (!enabled.get() || !p) return null;
    const frame = trackedFrame.get();
    return frame && frame.pred === p ? frame.boxes : null;
  }

  return {
    enabled,
    loading,
    supported,
    disabledReason,
    activeCount,
    trails,
    setEnabled,
    reset,
    boxesFor,
  };
}

export type TrackingController = ReturnType<typeof useTracking>;
