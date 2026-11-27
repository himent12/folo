// Pure helpers for Playground local video frame scrubbing and live webcam detection:
// file-type detection, frame stepping/clamping, trailing-edge single-flight throttling,
// rolling FPS/latency tracking, offscreen frame capture, and MediaStream cleanup / error copy.

export const VIDEO_ACCEPT = 'video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov';
export const PLAYGROUND_ACCEPT = `image/*,${VIDEO_ACCEPT}`;

/** Standard 30 fps frame duration used for -1 / +1 frame stepping when container metadata exposes no frame rate. */
export const DEFAULT_FRAME_STEP_SEC = 1 / 30;

/** Minimum interval between automatic scrub-triggered inference requests while dragging the timeline. */
export const SCRUB_THROTTLE_MS = 120;

/** Floor delay between consecutive live webcam / playing-video inference ticks after the prior tick settles. */
export const LIVE_LOOP_FLOOR_MS = 30;

/** True when a dropped or picked file is a local video (.mp4, .webm, .mov, .ogv, or video/* MIME). */
export function isVideoFile(file: Pick<File, 'name' | 'type'>): boolean {
  if (file.type && file.type.toLowerCase().startsWith('video/')) return true;
  return /\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name);
}

/** Clamps a media timestamp safely into `[0, duration]`. */
export function clampMediaTime(sec: number, duration: number): number {
  const max = Number.isFinite(duration) && duration > 0 ? duration : 0;
  if (!Number.isFinite(sec) || sec <= 0) return 0;
  return Math.min(max, sec);
}

/** Rounding tolerance for mapping integer-millisecond container timestamps (e.g. 33 ms at 30 fps) to frame indices. */
function frameBoundaryTolerance(step: number): number {
  return Math.min(0.00075, step * 0.15);
}

/**
 * Steps forward or backward by `deltaFrames` (e.g. -1 or +1) snapped to decoded frame boundaries.
 * Adds a small sub-frame nudge past the frame start so 1 ms container PTS rounding never decodes
 * the previous frame twice, and never moves backward when already on the last frame.
 */
export function stepFrameTime(current: number, deltaFrames: number, duration: number, frameStep = DEFAULT_FRAME_STEP_SEC): number {
  const step = Number.isFinite(frameStep) && frameStep > 0 ? frameStep : DEFAULT_FRAME_STEP_SEC;
  const cur = clampMediaTime(current, duration);
  if (!Number.isFinite(deltaFrames) || deltaFrames === 0) return Math.round(cur * 10000) / 10000;
  const total = totalFramesFor(duration, step);
  const maxIdx = total > 0 ? total - 1 : 0;
  const curIdx = Math.min(maxIdx, Math.max(0, Math.floor((cur + frameBoundaryTolerance(step)) / step)));
  const targetIdx = Math.min(maxIdx, Math.max(0, curIdx + deltaFrames));
  if (targetIdx === 0) return 0;
  if (targetIdx === curIdx && deltaFrames > 0) return Math.round(cur * 10000) / 10000;
  const nudge = Math.min(0.002, step * 0.1);
  const raw = targetIdx * step + nudge;
  return Math.round(clampMediaTime(raw, duration) * 10000) / 10000;
}

/** Formats seconds as `m:ss.t` (e.g. `0:04.2`), safe for NaN/negative/infinite inputs. */
export function formatMediaTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00.0';
  const m = Math.floor(sec / 60);
  const rem = sec - m * 60;
  return `${m}:${rem < 10 ? '0' : ''}${rem.toFixed(1)}`;
}

/** 1-based frame index at `sec` for display (`f 2 / 11`), clamped to `[1, totalFrames]` when provided. */
export function frameIndexAt(sec: number, frameStep = DEFAULT_FRAME_STEP_SEC, totalFrames?: number): number {
  if (!Number.isFinite(sec) || sec <= 0) return 1;
  const step = Number.isFinite(frameStep) && frameStep > 0 ? frameStep : DEFAULT_FRAME_STEP_SEC;
  const idx = Math.floor((sec + frameBoundaryTolerance(step)) / step) + 1;
  return totalFrames && totalFrames > 0 ? Math.min(totalFrames, idx) : idx;
}

/** Approximate total frame count for `duration`, or 0 when unknown. */
export function totalFramesFor(duration: number, frameStep = DEFAULT_FRAME_STEP_SEC): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  const step = Number.isFinite(frameStep) && frameStep > 0 ? frameStep : DEFAULT_FRAME_STEP_SEC;
  return Math.max(1, Math.round(duration / step));
}

/** Builds a clean filename for an extracted video frame (used in the toolbar and PNG/JSON exports). */
export function videoFrameName(videoName: string, sec: number): string {
  const stem = videoName.replace(/\.[^.]+$/, '') || 'video';
  const t = Number.isFinite(sec) && sec >= 0 ? sec.toFixed(2) : '0.00';
  return `${stem}@${t}s.jpg`;
}

/** Stops every track on a MediaStream so browser camera indicators turn off immediately. */
export function stopMediaStream(stream: MediaStream | null | undefined): void {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      // Track may already be ended/stopped by the browser.
    }
  }
}

/** Maps getUserMedia / environment failures to a clear, actionable inline message. */
export function describeCameraError(err: unknown): string {
  if (err === null || err === undefined) {
    return 'Camera access is unavailable in this browser or context (requires HTTPS or localhost with camera support).';
  }
  const name = err instanceof Error ? err.name : typeof err === 'object' && 'name' in err ? String((err as { name: unknown }).name) : '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError' || name === 'SecurityError') {
    return 'Camera permission was denied. Allow camera access in your browser settings and try again.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No camera was found on this device.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'The camera is already in use by another application or could not be opened.';
  }
  if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') {
    return 'No camera matched the requested video constraints.';
  }
  if (err instanceof Error && err.message) return err.message;
  return 'Could not start the camera.';
}

export interface FrameThrottleOptions {
  minIntervalMs: number;
  onTrigger: () => void;
  now?: () => number;
  setTimer?: (fn: () => void, ms: number) => ReturnType<typeof setTimeout>;
  clearTimer?: (id: ReturnType<typeof setTimeout>) => void;
}

export interface FrameThrottle {
  /** Requests a frame capture; coalesces rapid scrub events into a single trailing call and never overlaps in-flight work. */
  schedule: (immediate?: boolean) => void;
  /** Marks an inference tick as in flight so new schedules wait for settlement instead of queueing stale frames. */
  beginFlight: () => void;
  /** Marks the in-flight tick settled; flushes any coalesced pending schedule respecting `minIntervalMs`. */
  endFlight: () => void;
  /** True while an inference tick is currently in flight. */
  inFlight: () => boolean;
  /** Cancels any pending timer and clears queued work (call on pause, stop, or unmount). */
  cancel: () => void;
}

/**
 * Coordinates throttled, single-flight frame capture for video scrubbing and live webcam streams.
 * Rapid calls while throttled or in flight collapse to a single trailing capture of the latest frame.
 */
export function createFrameThrottle(opts: FrameThrottleOptions): FrameThrottle {
  const now = opts.now ?? (() => performance.now());
  const setTimer = opts.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
  const clearTimer = opts.clearTimer ?? ((id) => clearTimeout(id));
  const minInterval = Math.max(0, opts.minIntervalMs);

  let timer: ReturnType<typeof setTimeout> | null = null;
  let lastTriggerAt = -Infinity;
  let flight = false;
  let pending = false;
  let pendingImmediate = false;

  function clearPendingTimer() {
    if (timer !== null) {
      clearTimer(timer);
      timer = null;
    }
  }

  function fireNow() {
    clearPendingTimer();
    pending = false;
    pendingImmediate = false;
    lastTriggerAt = now();
    opts.onTrigger();
  }

  function armTrailing(immediate: boolean) {
    if (flight) {
      pending = true;
      pendingImmediate = pendingImmediate || immediate;
      return;
    }
    const elapsed = now() - lastTriggerAt;
    if (immediate || elapsed >= minInterval) {
      fireNow();
      return;
    }
    pending = true;
    if (timer === null) {
      const wait = Math.max(0, minInterval - elapsed);
      timer = setTimer(() => {
        timer = null;
        if (!flight && pending) fireNow();
      }, wait);
    }
  }

  return {
    schedule(immediate = false) {
      armTrailing(immediate);
    },
    beginFlight() {
      flight = true;
    },
    endFlight() {
      if (!flight) return;
      flight = false;
      if (pending) {
        const imm = pendingImmediate;
        pending = false;
        pendingImmediate = false;
        armTrailing(imm);
      }
    },
    inFlight: () => flight,
    cancel() {
      clearPendingTimer();
      flight = false;
      pending = false;
      pendingImmediate = false;
    },
  };
}

export interface FpsSnapshot {
  fps: number;
  latencyMs: number;
}

export interface FpsTracker {
  record: (latencyMs: number, atMs?: number) => FpsSnapshot;
  reset: () => void;
  snapshot: () => FpsSnapshot;
}

/** Rolling window FPS and smoothed inference latency tracker for live webcam and playing video. */
export function createFpsTracker(windowSize = 10): FpsTracker {
  const cap = Math.max(2, windowSize);
  const stamps: number[] = [];
  const latencies: number[] = [];

  function compute(): FpsSnapshot {
    if (latencies.length === 0) return { fps: 0, latencyMs: 0 };
    const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    let fps = 0;
    if (stamps.length >= 2) {
      const span = stamps[stamps.length - 1]! - stamps[0]!;
      fps = span > 0 ? ((stamps.length - 1) * 1000) / span : 0;
    } else if (avgLatency > 0) {
      fps = 1000 / avgLatency;
    }
    return {
      fps: Math.round(fps * 10) / 10,
      latencyMs: Math.round(avgLatency * 10) / 10,
    };
  }

  return {
    record(latencyMs: number, atMs = performance.now()) {
      const safeLatency = Number.isFinite(latencyMs) && latencyMs >= 0 ? latencyMs : 0;
      stamps.push(atMs);
      latencies.push(safeLatency);
      if (stamps.length > cap) stamps.shift();
      if (latencies.length > cap) latencies.shift();
      return compute();
    },
    reset() {
      stamps.length = 0;
      latencies.length = 0;
    },
    snapshot: compute,
  };
}

let captureCanvas: HTMLCanvasElement | null = null;

/** Releases the reusable offscreen capture canvas (called on Playground cleanup and tests). */
export function resetCaptureCanvas(): void {
  if (captureCanvas) {
    captureCanvas.width = 0;
    captureCanvas.height = 0;
    captureCanvas = null;
  }
}

/** Captures the current decoded frame of an HTMLVideoElement into a Blob without per-frame canvas allocations. */
export async function captureVideoFrame(
  video: HTMLVideoElement,
  opts: { mime?: 'image/jpeg' | 'image/png'; quality?: number } = {},
): Promise<{ blob: Blob; width: number; height: number }> {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height || video.readyState < 2) {
    throw new Error('Video frame is not ready yet.');
  }
  if (!captureCanvas) captureCanvas = document.createElement('canvas');
  if (captureCanvas.width !== width || captureCanvas.height !== height) {
    captureCanvas.width = width;
    captureCanvas.height = height;
  }
  const ctx = captureCanvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas is unavailable in this browser.');
  ctx.drawImage(video, 0, 0, width, height);
  const mime = opts.mime ?? 'image/jpeg';
  const quality = opts.quality ?? 0.92;
  const blob = await new Promise<Blob>((resolve, reject) => {
    captureCanvas!.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode video frame.'))), mime, quality);
  });
  return { blob, width, height };
}
