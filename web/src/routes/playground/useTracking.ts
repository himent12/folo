// Reactive ByteTrack controller for Playground live video & webcam sessions.
// Holds one persistent `folo-wasm` `FoloTracker` instance per active media session, resets it on seek, source change,
// camera restart, model change, explicit Clear tracks, or toggle off, records bounded frame-indexed track observations
// for JSON/CSV trajectory export, avoids double-stepping duplicate/re-filtered frames, and always calls `free()` on
// exit/unmount (zero WASM memory leaks).
import { computed, effect, onCleanup, signal, untrack } from 'fw';
import { ApiError } from '$lib/api/client';
import type { PredictBox, Prediction } from '$lib/api/types';
import { effectiveMinConf, filterDetections, thresholdForClass } from '$lib/detect/nms';
import type { BrowserEngine, Tracker } from '$lib/inference/browser';
import { toast } from '$lib/ui/toast';
import { exportTracksCsv as downloadTracksCsv, exportTracksJson as downloadTracksJson } from './export';
import { perClassConfKey, type PerClassConfMap } from './perClassConf';
import {
  buildTrackExportJson,
  EMPTY_TRACK_SUMMARY,
  MAX_TRACK_OBSERVATIONS,
  MAX_TRACK_SUMMARIES,
  trackerOptionsForConf,
  TrackHistoryRecorder,
  TrackTrailHistory,
  type TrackObservation,
  type TrackSessionSummary,
  type TrackTrail,
} from './tracking';
import type { MediaMode } from './useMediaSession';

export interface UseTrackingInput {
  engine: BrowserEngine;
  browserOff: () => string | null;
  mediaMode: () => MediaMode;
  /** Stable key for the active clip or camera stream (changes on source switch / clip replacement / camera restart). */
  mediaKey: () => string | null;
  /** Stable key identifying the current unique video/webcam frame (unchanged across Detect-frame retries & candidate completion). */
  frameKey?: () => string | null;
  /** Frame index and timestamp (ms) for the current captured frame. */
  frameMeta?: () => { frame: number; timestampMs: number } | null;
  /** True when `pred()` belongs to the currently active media frame (prevents attaching stale N-1 boxes to frame N). */
  predMatchesFrame?: () => boolean;
  /** Human-readable clip or stream name used for trajectory export filenames & metadata. */
  sourceName?: () => string;
  modelId: () => string | null;
  /** Display name of the active model used for trajectory export filenames & metadata. */
  modelName?: () => string | null;
  classes: () => readonly string[];
  pred: () => Prediction | null;
  conf: () => number;
  perClassConf?: () => PerClassConfMap;
  iou: () => number;
  maxDet: () => number;
  o2o: () => boolean;
  /** Optional upper bound on retained observations (defaults to `MAX_TRACK_OBSERVATIONS`). */
  maxObservations?: number;
}

const NO_TRACKER_TIP = 'This folo-wasm build does not include FoloTracker. Rebuild folo-wasm to enable Track IDs.';
const NO_WASM_TIP = 'Track IDs requires the folo-wasm browser runtime.';

export function useTracking(input: UseTrackingInput) {
  const enabled = signal(false);
  const loading = signal(false);
  const loadError = signal<string | null>(null);
  const activeCount = signal(0);
  const trails = signal<TrackTrail[]>([]);
  const summary = signal<TrackSessionSummary>(EMPTY_TRACK_SUMMARY);
  const trackedFrame = signal<{ pred: Prediction; frameKey: string | null; threshKey: string; boxes: PredictBox[] } | null>(null);
  const trackerReadyTick = signal(0);

  let tracker: Tracker | null = null;
  let trackerConf: number | null = null;
  let trackerThreshKey = '';
  let trackerSeq = 0;
  let alive = true;
  let lastProcessedPred: Prediction | null = null;
  let lastProcessedFrameKey: string | null = null;
  let fallbackFrame = 0;
  const activeTrackedClasses = new Set<number>();
  const trailHistory = new TrackTrailHistory();
  const trackHistory = new TrackHistoryRecorder(input.maxObservations ?? MAX_TRACK_OBSERVATIONS, MAX_TRACK_SUMMARIES);

  const supported = computed(() => !input.browserOff() && !loadError.get() && input.engine.capabilities.tracker !== false);
  const disabledReason = computed(() => {
    if (input.browserOff()) return NO_WASM_TIP;
    if (loadError.get()) return loadError.get();
    if (input.engine.capabilities.tracker === false) return NO_TRACKER_TIP;
    return null;
  });

  function clearOutputs(ignorePred: Prediction | null = null) {
    trailHistory.clear();
    trackHistory.clear();
    activeTrackedClasses.clear();
    fallbackFrame = 0;
    trails.set([]);
    activeCount.set(0);
    summary.set(EMPTY_TRACK_SUMMARY);
    trackedFrame.set(null);
    lastProcessedPred = ignorePred;
    lastProcessedFrameKey = null;
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

  /** Resets the persistent tracker's tracks, motion trails, and recorded trajectory history without reallocating WASM. */
  function reset(ignorePred: Prediction | null = untrack(() => input.pred())) {
    clearOutputs(ignorePred);
    if (tracker) {
      try {
        tracker.reset();
      } catch {
        freeTracker();
      }
    }
  }

  async function ensureTracker(wantConf: number) {
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
      const created = await input.engine.createTracker(trackerOptionsForConf(wantConf));
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
        toast.info('Track IDs unavailable', NO_TRACKER_TIP, undefined, { key: 'playground-tracking' });
        return;
      }
      tracker = created;
      trackerConf = wantConf;
      trackerReadyTick.set((t) => t + 1);
    } catch (err) {
      if (!alive || seq !== trackerSeq) return;
      enabled.set(false);
      clearOutputs();
      if (err instanceof ApiError && err.code === 'wasm_not_built') {
        loadError.set(NO_WASM_TIP);
        toast.info('Track IDs unavailable', NO_WASM_TIP, undefined, { key: 'playground-tracking' });
      } else {
        toast.error(err, 'Could not start Track IDs', { key: 'playground-tracking' });
      }
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
      lastProcessedFrameKey = null;
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
        const cur = input.pred();
        reset(cur && cur.model_id === lastModelId ? cur : null);
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
    const pcc = input.perClassConf?.();
    const eff = effectiveMinConf(c, pcc);
    const threshKey = `${c}|${perClassConfKey(pcc)}`;
    if (!on || mode === 'none') return;
    untrack(() => {
      const confChanged = tracker && trackerConf !== null && Math.abs(trackerConf - eff) > 1e-4;
      const threshChanged = tracker && threshKey !== trackerThreshKey;
      trackerThreshKey = threshKey;
      if (confChanged) {
        clearOutputs();
      } else if (threshChanged && tracker) {
        reset(null);
        trackerReadyTick.set((t) => t + 1);
      }
      void ensureTracker(eff);
    });
  });

  // Feed each new live frame into FoloTracker exactly once (never double-stepping on UI toggles, slider drags,
  // duplicate Detect-frame clicks on the same timestamp, or askAllWhenCut candidate completion).
  effect(() => {
    const on = enabled.get();
    const mode = input.mediaMode();
    const p = input.pred();
    const fk = input.frameKey ? input.frameKey() : null;
    const matchesFrame = input.predMatchesFrame ? input.predMatchesFrame() : true;
    void trackerReadyTick.get();
    if (!on || mode === 'none' || !p || !matchesFrame) {
      untrack(() => {
        if (!p) {
          trackedFrame.set(null);
          lastProcessedFrameKey = null;
        }
      });
      return;
    }
    untrack(() => {
      if (!tracker || p === lastProcessedPred) return;
      const cur = trackedFrame.peek();
      if (
        fk !== null &&
        fk === lastProcessedFrameKey &&
        cur &&
        cur.frameKey === fk &&
        cur.threshKey === trackerThreshKey &&
        cur.pred.model_id === p.model_id &&
        (cur.pred.backend ?? 'server') === (p.backend ?? 'server')
      ) {
        // Same physical frame & model/backend (e.g. candidate completion or repeat Detect frame): keep existing tracked boxes without advancing ByteTrack.
        lastProcessedPred = p;
        trackedFrame.set({ pred: p, frameKey: fk, threshKey: cur.threshKey, boxes: cur.boxes });
        return;
      }
      lastProcessedPred = p;
      lastProcessedFrameKey = fk;
      const c = input.conf();
      const pcc = input.perClassConf?.();
      const eff = effectiveMinConf(c, pcc);
      const iou = input.iou();
      const maxDet = input.maxDet();
      const o2o = input.o2o();
      const cls = input.classes();
      // Gate per-class eligibility BEFORE NMS and `maxDet` so boxes rejected by a raised override never consume
      // a `maxDet` slot ahead of valid lower-threshold classes, while keeping stage-2 low-confidence recovery.
      const low = Math.min(0.1, eff);
      const eligible = p.boxes.filter((b) => {
        const t = thresholdForClass(b.class, c, pcc);
        if (b.score >= t) return true;
        return b.score >= Math.min(0.1, t) && b.score < eff && activeTrackedClasses.has(b.class);
      });
      const candidates = filterDetections(eligible, { conf: low, iou: o2o ? 1 : iou, maxDet });
      const tracked = tracker.update(candidates);
      for (const t of tracked) activeTrackedClasses.add(t.class);
      const boxes: PredictBox[] = tracked.map((t) => ({
        class: t.class,
        name: cls[t.class] ?? t.name,
        score: t.score,
        x1: t.x1,
        y1: t.y1,
        x2: t.x2,
        y2: t.y2,
        track_id: t.track_id,
      }));
      const meta = input.frameMeta?.();
      fallbackFrame++;
      const frameIndex = meta ? meta.frame : fallbackFrame;
      const timestampMs = meta ? meta.timestampMs : 0;
      const nextSummary = trackHistory.record(frameIndex, timestampMs, tracked, cls);

      trackedFrame.set({ pred: p, frameKey: fk, threshKey: `${c}|${perClassConfKey(pcc)}`, boxes });
      trails.set(trailHistory.update(tracked));
      activeCount.set(tracked.length);
      summary.set(nextSummary);
    });
  });

  onCleanup(() => {
    alive = false;
    clearOutputs();
    freeTracker();
  });

  /** Returns tracked boxes matching `p` when tracking is active for that frame and threshold config, else null. */
  function boxesFor(p: Prediction | null): PredictBox[] | null {
    if (!enabled.get() || !p) return null;
    const frame = trackedFrame.get();
    const curThreshKey = `${input.conf()}|${perClassConfKey(input.perClassConf?.())}`;
    return frame && frame.pred === p && frame.threshKey === curThreshKey ? frame.boxes : null;
  }

  /** Returns a chronological snapshot of recorded frame-indexed track observations. */
  function getObservations(): TrackObservation[] {
    return trackHistory.observations();
  }

  function exportFilenameBase(): string {
    const mode = untrack(() => input.mediaMode());
    const rawSource = untrack(() => input.sourceName?.() ?? '') || (mode === 'webcam' ? 'webcam' : 'video');
    const stem = rawSource.replace(/\.[^.]+$/, '') || 'tracks';
    const mName = untrack(() => input.modelName?.() ?? input.modelId() ?? '');
    return mName ? `${stem}-${mName}-tracks` : `${stem}-tracks`;
  }

  /** Downloads all recorded frame-indexed track observations and session summary as JSON. */
  function exportTracksJson(): boolean {
    const mode = untrack(() => input.mediaMode());
    if (mode === 'none') return false;
    const obs = trackHistory.observations();
    if (!obs.length) return false;
    const payload = buildTrackExportJson({
      sourceName: untrack(() => input.sourceName?.() ?? '') || mode,
      mode,
      modelId: untrack(() => input.modelId()),
      modelName: untrack(() => input.modelName?.() ?? null),
      summary: trackHistory.summary(),
      observations: obs,
    });
    return downloadTracksJson(exportFilenameBase(), payload);
  }

  /** Downloads all recorded frame-indexed track observations as RFC-4180 CSV. */
  function exportTracksCsv(): boolean {
    const obs = trackHistory.observations();
    if (!obs.length) return false;
    return downloadTracksCsv(exportFilenameBase(), obs);
  }

  /** Explicitly clears recorded trajectories, trails, and active tracker state without leaving media mode. */
  function clearTracks() {
    reset();
  }

  return {
    enabled,
    loading,
    supported,
    disabledReason,
    activeCount,
    trails,
    summary,
    setEnabled,
    reset,
    clearTracks,
    boxesFor,
    getObservations,
    exportTracksJson,
    exportTracksCsv,
  };
}

export type TrackingController = ReturnType<typeof useTracking>;
