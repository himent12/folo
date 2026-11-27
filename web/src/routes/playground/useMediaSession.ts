// Reactive session controller for Playground Local Video Frame Scrubber and Live Webcam Detection.
// Guarantees single-flight / throttled frame inference (zero stale backlog while scrubbing or streaming)
// and immediate MediaStream + object URL teardown on exit, source switch, unmount, or pagehide.
import { computed, effect, onCleanup, onWindow, signal, untrack } from 'fw';
import {
  captureVideoFrame,
  clampMediaTime,
  createFpsTracker,
  createFrameThrottle,
  DEFAULT_FRAME_STEP_SEC,
  describeCameraError,
  LIVE_LOOP_FLOOR_MS,
  resetCaptureCanvas,
  SCRUB_THROTTLE_MS,
  stepFrameTime,
  stopMediaStream,
  videoFrameName,
} from './media';

export type MediaMode = 'none' | 'video' | 'webcam';

export interface MediaFramePayload {
  blob: Blob;
  name: string;
  width: number;
  height: number;
  /** True during continuous playback / live webcam ticks (preserves previous boxes while next frame runs). */
  live: boolean;
}

export interface UseMediaSessionInput {
  /** Dispatches a decoded video or webcam frame into Playground's active source/inference pipeline. */
  onFrame: (frame: MediaFramePayload) => void;
  /** True while Model A (or Model B in Compare mode) is currently running inference. */
  isPredicting: () => boolean;
  /** Latest inference latency (ms) reported by Playground once a tick finishes, or null if failed. */
  lastLatencyMs: () => number | null;
  /** Called when clearing media mode without replacing it with another source. */
  onStopped?: () => void;
}

export function useMediaSession(input: UseMediaSessionInput) {
  const mode = signal<MediaMode>('none');
  const videoEl = signal<HTMLVideoElement | undefined>(undefined);

  // Local video state
  const videoUrl = signal<string | null>(null);
  const videoName = signal('');
  const duration = signal(0);
  const currentTime = signal(0);
  const playing = signal(false);
  const videoReady = signal(false);
  const seeking = signal(false);
  const videoErr = signal<string | null>(null);
  const autoDetect = signal(true);
  const videoDims = signal<{ width: number; height: number } | null>(null);

  // Live webcam state
  const camActive = signal(false);
  const camStarting = signal(false);
  const camPaused = signal(false);
  const camErr = signal<string | null>(null);
  const camDims = signal<{ width: number; height: number } | null>(null);

  // Live telemetry (shared for playing video and live webcam)
  const fps = signal(0);
  const latencyMs = signal(0);

  const canCamera = computed(() => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia);
  const liveStreaming = computed(() => (mode.get() === 'webcam' && camActive.get() && !camPaused.get()) || (mode.get() === 'video' && playing.get()));

  let alive = true;
  let camSeq = 0;
  let captureSeq = 0;
  let stream: MediaStream | null = null;
  let streamTrackCleanups: (() => void)[] = [];
  let initialVideoCaptured = false;
  let frameSentAt = 0;
  let waitingForInference = false;
  let liveTimer: ReturnType<typeof setTimeout> | null = null;

  const tracker = createFpsTracker(10);

  function clearLiveTimer() {
    if (liveTimer !== null) {
      clearTimeout(liveTimer);
      liveTimer = null;
    }
  }

  function clearTrackListeners() {
    for (const off of streamTrackCleanups) off();
    streamTrackCleanups = [];
  }

  function resetTelemetry() {
    tracker.reset();
    fps.set(0);
    latencyMs.set(0);
    waitingForInference = false;
    clearLiveTimer();
  }

  const scrubThrottle = createFrameThrottle({
    minIntervalMs: SCRUB_THROTTLE_MS,
    onTrigger: () => {
      void captureAndEmit(false);
    },
  });

  async function captureAndEmit(live: boolean): Promise<boolean> {
    const el = videoEl.peek();
    const m = mode.peek();
    if (!alive || !el || m === 'none') return false;
    if (m === 'video' && (!videoReady.peek() || seeking.peek())) return false;
    if (m === 'webcam' && (!camActive.peek() || el.readyState < 2)) return false;

    const seq = ++captureSeq;
    scrubThrottle.beginFlight();
    try {
      const frame = await captureVideoFrame(el, { mime: 'image/jpeg', quality: live ? 0.88 : 0.94 });
      if (!alive || seq !== captureSeq || mode.peek() !== m) {
        scrubThrottle.endFlight();
        return false;
      }
      if (m === 'video') videoDims.set({ width: frame.width, height: frame.height });
      else camDims.set({ width: frame.width, height: frame.height });
      const name = m === 'video' ? videoFrameName(videoName.peek(), el.currentTime) : 'webcam-live.jpg';
      frameSentAt = performance.now();
      waitingForInference = true;
      input.onFrame({ blob: frame.blob, name, width: frame.width, height: frame.height, live });
      // If inference completed synchronously or was skipped (e.g. no model), release flight on next microtask.
      queueMicrotask(() => {
        if (!alive || seq !== captureSeq) return;
        if (!input.isPredicting() && waitingForInference) settleInferenceTick();
      });
      return true;
    } catch (e) {
      scrubThrottle.endFlight();
      if (!alive || seq !== captureSeq) return false;
      if (!live) {
        const msg = e instanceof Error ? e.message : 'Could not read video frame.';
        if (m === 'video') videoErr.set(msg);
        else camErr.set(msg);
      }
      return false;
    }
  }

  function scheduleNextLiveTick() {
    clearLiveTimer();
    if (!alive) return;
    const m = mode.peek();
    const shouldLoop = (m === 'webcam' && camActive.peek() && !camPaused.peek()) || (m === 'video' && playing.peek() && autoDetect.peek());
    if (!shouldLoop || waitingForInference) return;
    liveTimer = setTimeout(() => {
      liveTimer = null;
      if (!alive) return;
      const curMode = mode.peek();
      const stillLoop = (curMode === 'webcam' && camActive.peek() && !camPaused.peek()) || (curMode === 'video' && playing.peek() && autoDetect.peek());
      if (!stillLoop || waitingForInference) return;
      void captureAndEmit(true).then((sent) => {
        if (!sent && alive) scheduleNextLiveTick();
      });
    }, LIVE_LOOP_FLOOR_MS);
  }

  function settleInferenceTick() {
    if (!waitingForInference) return;
    waitingForInference = false;
    scrubThrottle.endFlight();
    const m = mode.peek();
    const isLiveMode = (m === 'webcam' && camActive.peek() && !camPaused.peek()) || (m === 'video' && playing.peek());
    if (isLiveMode) {
      const measured = input.lastLatencyMs() ?? Math.max(0, performance.now() - frameSentAt);
      const snap = tracker.record(measured);
      fps.set(snap.fps);
      latencyMs.set(snap.latencyMs);
      scheduleNextLiveTick();
    }
  }

  // Watch Playground's prediction busy state so single-flight streaming advances strictly after settlement.
  effect(() => {
    const busy = input.isPredicting();
    if (!busy && waitingForInference) {
      untrack(() => settleInferenceTick());
    }
  });

  function stopCameraTracksOnly() {
    camSeq++;
    clearTrackListeners();
    stopMediaStream(stream);
    stream = null;
    const el = videoEl.peek();
    if (el && el.srcObject) {
      el.pause();
      el.srcObject = null;
    }
    camActive.set(false);
    camStarting.set(false);
    camPaused.set(false);
    camDims.set(null);
  }

  function clearVideoState() {
    const el = videoEl.peek();
    if (el && mode.peek() === 'video') {
      el.pause();
      el.removeAttribute('src');
      el.load();
    }
    const old = videoUrl.peek();
    if (old) URL.revokeObjectURL(old);
    videoUrl.set(null);
    videoName.set('');
    duration.set(0);
    currentTime.set(0);
    playing.set(false);
    videoReady.set(false);
    seeking.set(false);
    videoErr.set(null);
    videoDims.set(null);
    initialVideoCaptured = false;
  }

  /** Stops any active video or webcam session and resets mode back to 'none'. */
  function stopAll(notify = false) {
    captureSeq++;
    scrubThrottle.cancel();
    resetTelemetry();
    stopCameraTracksOnly();
    clearVideoState();
    camErr.set(null);
    const wasActive = mode.peek() !== 'none';
    mode.set('none');
    if (notify && wasActive) input.onStopped?.();
  }

  /** Loads a local video file (.mp4/.webm/.mov) directly into Playground's inline frame scrubber. */
  function loadVideo(file: File) {
    captureSeq++;
    scrubThrottle.cancel();
    resetTelemetry();
    stopCameraTracksOnly();
    clearVideoState();
    camErr.set(null);
    videoName.set(file.name);
    videoUrl.set(URL.createObjectURL(file));
    mode.set('video');
  }

  /** Starts live webcam streaming via navigator.mediaDevices.getUserMedia with graceful error fallback. */
  async function startWebcam() {
    captureSeq++;
    scrubThrottle.cancel();
    resetTelemetry();
    clearVideoState();
    stopCameraTracksOnly();
    camErr.set(null);
    mode.set('webcam');

    if (!canCamera.peek()) {
      camErr.set(describeCameraError(null));
      return;
    }

    const seq = ++camSeq;
    camStarting.set(true);
    let acquired: MediaStream | null = null;
    try {
      try {
        acquired = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      } catch (firstErr) {
        if (firstErr instanceof Error && (firstErr.name === 'OverconstrainedError' || firstErr.name === 'ConstraintNotSatisfiedError')) {
          acquired = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } else {
          throw firstErr;
        }
      }

      // Guard against unmount, Stop, or mode switch while getUserMedia was pending.
      if (!alive || seq !== camSeq || mode.peek() !== 'webcam') {
        stopMediaStream(acquired);
        return;
      }

      stream = acquired;
      // If the browser or hardware ends the camera track externally, clean up cleanly.
      const tracks = acquired.getTracks();
      streamTrackCleanups = tracks.map((track) => {
        const onEnded = () => {
          if (!alive || stream !== acquired) return;
          stopCameraTracksOnly();
          resetTelemetry();
          camErr.set('Camera stream ended unexpectedly.');
        };
        track.addEventListener?.('ended', onEnded);
        return () => track.removeEventListener?.('ended', onEnded);
      });

      const el = videoEl.peek();
      if (!el) {
        stopCameraTracksOnly();
        return;
      }
      el.srcObject = acquired;
      try {
        await el.play();
      } catch (playErr) {
        stopCameraTracksOnly();
        if (!alive || seq !== camSeq || mode.peek() !== 'webcam') return;
        throw playErr;
      }
      if (!alive || seq !== camSeq || mode.peek() !== 'webcam') {
        stopCameraTracksOnly();
        return;
      }

      if (el.videoWidth > 0 && el.videoHeight > 0) {
        camDims.set({ width: el.videoWidth, height: el.videoHeight });
      }
      camActive.set(true);
      camPaused.set(false);
      scheduleNextLiveTick();
    } catch (err) {
      if (!alive || seq !== camSeq || mode.peek() !== 'webcam') return;
      camErr.set(describeCameraError(err));
    } finally {
      if (alive && seq === camSeq) camStarting.set(false);
    }
  }

  function stopWebcam() {
    stopAll(true);
  }

  function toggleCamPause() {
    if (mode.peek() !== 'webcam' || !camActive.peek()) return;
    const el = videoEl.peek();
    const nextPaused = !camPaused.peek();
    camPaused.set(nextPaused);
    if (nextPaused) {
      clearLiveTimer();
      el?.pause();
      // Capture one crisp frozen frame for inspection / zoom / export.
      void captureAndEmit(false);
    } else {
      tracker.reset();
      void el?.play().catch(() => {});
      scheduleNextLiveTick();
    }
  }

  function seekTo(sec: number, immediate = false) {
    if (mode.peek() !== 'video') return;
    const clamped = clampMediaTime(sec, duration.peek());
    currentTime.set(clamped);
    const el = videoEl.peek();
    if (!el || !Number.isFinite(clamped)) return;
    if (Math.abs(el.currentTime - clamped) > 0.001) {
      seeking.set(true);
      el.currentTime = clamped;
    } else if (autoDetect.peek()) {
      scrubThrottle.schedule(immediate);
    }
  }

  function stepFrame(delta: -1 | 1) {
    if (mode.peek() !== 'video' || !videoReady.peek()) return;
    const el = videoEl.peek();
    if (el && !el.paused) el.pause();
    playing.set(false);
    clearLiveTimer();
    const next = stepFrameTime(currentTime.peek(), delta, duration.peek(), DEFAULT_FRAME_STEP_SEC);
    seekTo(next, true);
  }

  function togglePlay() {
    if (mode.peek() !== 'video' || !videoReady.peek()) return;
    const el = videoEl.peek();
    if (!el) return;
    if (el.paused) {
      if (duration.peek() > 0 && el.currentTime >= duration.peek() - 0.01) {
        el.currentTime = 0;
        currentTime.set(0);
      }
      tracker.reset();
      void el.play().catch(() => {});
    } else {
      el.pause();
    }
  }

  function detectCurrentFrame() {
    if (mode.peek() === 'none') return;
    scrubThrottle.cancel();
    void captureAndEmit(false);
  }

  // Video element event handlers bound by the hidden/offstage decoder <video> element.
  const videoEvents = {
    onloadedmetadata: (e: Event & { currentTarget: HTMLVideoElement }) => {
      const v = e.currentTarget;
      if (mode.peek() === 'webcam') {
        if (v.videoWidth > 0 && v.videoHeight > 0) camDims.set({ width: v.videoWidth, height: v.videoHeight });
        return;
      }
      duration.set(Number.isFinite(v.duration) ? v.duration : 0);
      if (v.videoWidth > 0 && v.videoHeight > 0) videoDims.set({ width: v.videoWidth, height: v.videoHeight });
      const ready = v.videoWidth > 0 && v.readyState >= 2;
      videoReady.set(ready);
      videoErr.set(null);
    },
    onloadeddata: (e: Event & { currentTarget: HTMLVideoElement }) => {
      const v = e.currentTarget;
      if (mode.peek() === 'webcam') {
        if (v.videoWidth > 0 && v.videoHeight > 0) camDims.set({ width: v.videoWidth, height: v.videoHeight });
        return;
      }
      if (v.videoWidth > 0 && v.videoHeight > 0) videoDims.set({ width: v.videoWidth, height: v.videoHeight });
      const ready = v.videoWidth > 0 && v.readyState >= 2;
      videoReady.set(ready);
      if (ready && !initialVideoCaptured) {
        initialVideoCaptured = true;
        void captureAndEmit(false);
      }
    },
    oncanplay: (e: Event & { currentTarget: HTMLVideoElement }) => {
      const v = e.currentTarget;
      if (mode.peek() !== 'video') return;
      const ready = v.videoWidth > 0 && v.readyState >= 2;
      videoReady.set(ready);
      if (ready && !initialVideoCaptured) {
        initialVideoCaptured = true;
        void captureAndEmit(false);
      }
    },
    onseeking: () => {
      if (mode.peek() === 'video') seeking.set(true);
    },
    onseeked: (e: Event & { currentTarget: HTMLVideoElement }) => {
      if (mode.peek() !== 'video') return;
      const v = e.currentTarget;
      seeking.set(false);
      currentTime.set(v.currentTime);
      videoReady.set(v.videoWidth > 0 && v.readyState >= 2);
      if (autoDetect.peek() && !playing.peek()) {
        scrubThrottle.schedule(false);
      }
    },
    ontimeupdate: (e: Event & { currentTarget: HTMLVideoElement }) => {
      if (mode.peek() === 'video') currentTime.set(e.currentTarget.currentTime);
    },
    onplay: () => {
      if (mode.peek() !== 'video') return;
      playing.set(true);
      if (autoDetect.peek()) scheduleNextLiveTick();
    },
    onpause: () => {
      if (mode.peek() !== 'video') return;
      playing.set(false);
      clearLiveTimer();
      // Snapshot the exact paused frame so zoom, inspection, Compare A/B, and PNG/JSON export match it.
      if (videoReady.peek() && autoDetect.peek()) scrubThrottle.schedule(true);
    },
    onended: () => {
      if (mode.peek() !== 'video') return;
      playing.set(false);
      clearLiveTimer();
      if (videoReady.peek() && autoDetect.peek()) scrubThrottle.schedule(true);
    },
    onerror: () => {
      if (mode.peek() !== 'video') return;
      videoReady.set(false);
      seeking.set(false);
      playing.set(false);
      clearLiveTimer();
      videoErr.set('This browser could not decode the selected video file. Try an MP4 (H.264) or WebM video.');
    },
  };

  onWindow('pagehide', () => stopAll(false));
  onCleanup(() => {
    alive = false;
    stopAll(false);
    resetCaptureCanvas();
  });

  return {
    mode,
    videoEl,
    videoUrl,
    videoName,
    duration,
    currentTime,
    playing,
    videoReady,
    seeking,
    videoErr,
    autoDetect,
    videoDims,
    camActive,
    camStarting,
    camPaused,
    camErr,
    camDims,
    canCamera,
    liveStreaming,
    fps,
    latencyMs,
    videoEvents,
    loadVideo,
    startWebcam,
    stopWebcam,
    stopAll,
    toggleCamPause,
    seekTo,
    stepFrame,
    togglePlay,
    detectCurrentFrame,
  };
}

export type MediaSession = ReturnType<typeof useMediaSession>;
