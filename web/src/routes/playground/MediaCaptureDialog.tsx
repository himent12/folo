/**
 * Capture a single frame into Playground from a local video file (.mp4/.webm/.mov with timeline scrubbing) or live
 * webcam snapshot (with graceful degradation when camera hardware or permissions are unavailable).
 */
import Pause from '$icons/square';
import Play from '$icons/play';
import Scan from '$icons/scan';
import Upload from '$icons/upload';
import { bindable, computed, css, effect, onCleanup, signal } from 'fw';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import Segmented from '$lib/ui/Segmented';
import Slider from '$lib/ui/Slider';

export interface MediaCaptureDialogProps {
  open?: boolean;
  /** Initial local video file when opened by dropping/picking a video file directly. */
  initialVideo?: File | null;
  onCapture: (blob: Blob, name: string) => void;
  onClose?: () => void;
}

type Mode = 'video' | 'camera';
const STEP_SEC = 0.1;

function formatSec(s: number): string {
  if (!Number.isFinite(s) || s < 0) return '0:00.0';
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return `${m}:${rem < 10 ? '0' : ''}${rem.toFixed(1)}`;
}

/** Captures the current decoded frame of an HTMLVideoElement as a PNG blob. */
async function videoFrameToPng(video: HTMLVideoElement): Promise<Blob> {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h || video.readyState < 2) throw new Error('Video frame is not ready yet.');
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas is unavailable.');
  ctx.drawImage(video, 0, 0, w, h);
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG capture failed.'))), 'image/png');
  });
}

export default function MediaCaptureDialog(props: MediaCaptureDialogProps) {
  const open = bindable(props, 'open', false);
  const mode = signal<Mode>('video');

  // Lifecycle and request-sequencing guards prevent stale async streams or captures from committing after close/unmount.
  let alive = true;
  let camSeq = 0;
  let captureSeq = 0;

  // Video file state
  let videoInput!: HTMLInputElement;
  let videoEl: HTMLVideoElement | undefined;
  const videoUrl = signal<string | null>(null);
  const videoName = signal('');
  const duration = signal(0);
  const currentTime = signal(0);
  const playing = signal(false);
  const videoReady = signal(false);
  const seeking = signal(false);
  const videoErr = signal<string | null>(null);

  // Camera state
  let camEl: HTMLVideoElement | undefined;
  let stream: MediaStream | null = null;
  const camActive = signal(false);
  const camBusy = signal(false);
  const camErr = signal<string | null>(null);

  const captureBusy = signal(false);
  const canCamera = computed(() => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia);

  function stopTracks(s: MediaStream | null) {
    if (!s) return;
    for (const t of s.getTracks()) t.stop();
  }

  function stopCamera() {
    camSeq++;
    captureSeq++;
    captureBusy.set(false);
    stopTracks(stream);
    stream = null;
    if (camEl) camEl.srcObject = null;
    camActive.set(false);
    camBusy.set(false);
  }

  function resetVideoPlayback() {
    if (videoEl && !videoEl.paused) videoEl.pause();
    playing.set(false);
    videoReady.set(false);
    seeking.set(false);
    currentTime.set(0);
  }

  function clearVideo() {
    captureSeq++;
    captureBusy.set(false);
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
  }

  function loadVideoFile(f: File) {
    clearVideo();
    mode.set('video');
    videoName.set(f.name);
    videoUrl.set(URL.createObjectURL(f));
  }

  effect(() => {
    if (open.get() && props.initialVideo) {
      loadVideoFile(props.initialVideo);
    }
    if (!open.get()) {
      captureSeq++;
      stopCamera();
      camErr.set(null);
      if (videoEl && !videoEl.paused) videoEl.pause();
      playing.set(false);
      seeking.set(false);
      captureBusy.set(false);
    }
  });

  effect(() => {
    const m = mode.get();
    captureSeq++;
    captureBusy.set(false);
    if (m !== 'camera') stopCamera();
    if (m !== 'video') resetVideoPlayback();
  });

  onCleanup(() => {
    alive = false;
    captureSeq++;
    stopCamera();
    clearVideo();
  });

  async function startCamera() {
    if (!canCamera.peek()) {
      camErr.set('Camera capture is not available in this browser environment.');
      return;
    }
    stopCamera();
    const seq = ++camSeq;
    camErr.set(null);
    camBusy.set(true);
    try {
      const next = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      if (!alive || seq !== camSeq || !open.peek() || mode.peek() !== 'camera') {
        stopTracks(next);
        return;
      }
      stream = next;
      const targetEl = camEl;
      if (targetEl) {
        targetEl.srcObject = next;
        try {
          await targetEl.play();
        } catch (playErr) {
          stopTracks(next);
          if (stream === next) stream = null;
          if (targetEl.srcObject === next) targetEl.srcObject = null;
          if (!alive || seq !== camSeq) return;
          throw playErr;
        }
      }
      if (!alive || seq !== camSeq || !open.peek() || mode.peek() !== 'camera') {
        stopTracks(next);
        if (stream === next) stream = null;
        if (targetEl && targetEl.srcObject === next) targetEl.srcObject = null;
        return;
      }
      camActive.set(true);
    } catch (e) {
      if (!alive || seq !== camSeq) return;
      const msg = e instanceof Error && e.message ? e.message : 'Camera permission was denied or no camera was found.';
      camErr.set(msg);
    } finally {
      if (alive && seq === camSeq) camBusy.set(false);
    }
  }

  function seekTo(sec: number) {
    const max = duration.peek();
    const clamped = Math.max(0, Math.min(max || 0, sec));
    currentTime.set(clamped);
    if (videoEl && Number.isFinite(clamped)) {
      if (Math.abs(videoEl.currentTime - clamped) > 0.001) seeking.set(true);
      videoEl.currentTime = clamped;
    }
  }

  function togglePlay() {
    if (!videoEl || !videoReady.peek()) return;
    if (videoEl.paused) void videoEl.play().catch(() => {});
    else videoEl.pause();
  }

  async function captureCurrent() {
    const activeMode = mode.peek();
    const el = activeMode === 'video' ? videoEl : camEl;
    if (!el || captureBusy.peek()) return;
    if (activeMode === 'video' && (!videoReady.peek() || seeking.peek())) return;
    const seq = ++captureSeq;
    const snapshotName =
      activeMode === 'video'
        ? `${videoName.peek().replace(/\.[^.]+$/, '') || 'video'}-${currentTime.peek().toFixed(1)}s.png`
        : 'camera-snapshot.png';
    captureBusy.set(true);
    videoErr.set(null);
    camErr.set(null);
    try {
      const blob = await videoFrameToPng(el);
      if (!alive || seq !== captureSeq || !open.peek() || mode.peek() !== activeMode) return;
      stopCamera();
      open.set(false);
      props.onCapture(blob, snapshotName);
    } catch (e) {
      if (!alive || seq !== captureSeq) return;
      const msg = e instanceof Error ? e.message : 'Could not capture frame.';
      if (activeMode === 'video') videoErr.set(msg);
      else camErr.set(msg);
    } finally {
      if (alive && seq === captureSeq) captureBusy.set(false);
    }
  }

  const canCapture = computed(() =>
    !captureBusy.get() && (mode.get() === 'video' ? videoReady.get() && !seeking.get() : camActive.get()),
  );

  return (
    <Dialog
      bind:open={open}
      title="Capture frame"
      description="Scrub a local video file or take a camera snapshot to run detection on one frame."
      size="lg"
      onclose={props.onClose}
      footer={() => (
        <>
          <Button variant="ghost" onclick={() => open.set(false)}>
            Cancel
          </Button>
          <Button variant="primary" icon={Scan} disabled={!canCapture.get()} onclick={() => void captureCurrent()}>
            {captureBusy.get() ? 'Capturing…' : 'Use this frame'}
          </Button>
        </>
      )}
    >
      <input
        ref={(el) => (videoInput = el)}
        type="file"
        accept="video/mp4,video/webm,video/quicktime,video/*"
        class="sr-only"
        tabindex={-1}
        aria-hidden="true"
        onchange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) loadVideoFile(f);
          e.currentTarget.value = '';
        }}
      />

      <div class="cap-root">
        <div class="cap-bar">
          <Segmented
            label="Capture source"
            size="sm"
            bind:value={mode}
            options={[
              { value: 'video', label: 'Video file' },
              { value: 'camera', label: 'Camera' },
            ]}
          />
          {mode.get() === 'video' ? (
            <Button size="sm" icon={Upload} onclick={() => videoInput.click()}>
              {videoUrl.get() ? 'Choose another video' : 'Choose video'}
            </Button>
          ) : (
            <Button
              size="sm"
              variant={camActive.get() ? 'secondary' : 'primary'}
              disabled={camBusy.get() || !canCamera.get()}
              onclick={() => (camActive.get() ? stopCamera() : void startCamera())}
            >
              {camBusy.get() ? 'Starting camera…' : camActive.get() ? 'Stop camera' : 'Start camera'}
            </Button>
          )}
        </div>

        {mode.get() === 'video' ? (
          <div class="cap-stage">
            {videoUrl.get() ? (
              <>
                <video
                  ref={(el) => (videoEl = el)}
                  src={videoUrl.get()!}
                  muted
                  playsinline
                  preload="auto"
                  onloadedmetadata={(e) => {
                    const v = e.currentTarget;
                    duration.set(Number.isFinite(v.duration) ? v.duration : 0);
                    videoReady.set(v.videoWidth > 0 && v.readyState >= 2);
                    videoErr.set(null);
                  }}
                  onloadeddata={(e) => {
                    const v = e.currentTarget;
                    videoReady.set(v.videoWidth > 0 && v.readyState >= 2);
                  }}
                  oncanplay={(e) => {
                    const v = e.currentTarget;
                    videoReady.set(v.videoWidth > 0 && v.readyState >= 2);
                  }}
                  onseeking={() => seeking.set(true)}
                  onseeked={(e) => {
                    const v = e.currentTarget;
                    seeking.set(false);
                    currentTime.set(v.currentTime);
                    videoReady.set(v.videoWidth > 0 && v.readyState >= 2);
                  }}
                  ontimeupdate={(e) => {
                    currentTime.set(e.currentTarget.currentTime);
                  }}
                  onplay={() => playing.set(true)}
                  onpause={() => playing.set(false)}
                  onended={() => playing.set(false)}
                  onerror={() => {
                    videoReady.set(false);
                    seeking.set(false);
                    videoErr.set('This browser could not decode the selected video file.');
                  }}
                ></video>
                <div class="scrub">
                  <div class="scrub-controls">
                    <Button
                      size="sm"
                      variant="ghost"
                      icon={playing.get() ? Pause : Play}
                      aria-label={playing.get() ? 'Pause video' : 'Play video'}
                      disabled={!videoReady.get()}
                      onclick={togglePlay}
                    />
                    <Button
                      size="sm"
                      variant="ghost"
                      aria-label="Step back 0.1 seconds"
                      disabled={!videoReady.get() || seeking.get()}
                      onclick={() => seekTo(currentTime.peek() - STEP_SEC)}
                    >
                      -0.1s
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      aria-label="Step forward 0.1 seconds"
                      disabled={!videoReady.get() || seeking.get()}
                      onclick={() => seekTo(currentTime.peek() + STEP_SEC)}
                    >
                      +0.1s
                    </Button>
                    <span class="time num">
                      {formatSec(currentTime.get())} / {formatSec(duration.get())}
                    </span>
                  </div>
                  {duration.get() > 0 ? (
                    <Slider
                      inline
                      label="Position"
                      value={currentTime.get()}
                      min={0}
                      max={duration.get()}
                      step={0.05}
                      format={formatSec}
                      oninput={seekTo}
                    />
                  ) : null}
                </div>
              </>
            ) : (
              <div class="cap-empty">
                <p>Select a local .mp4 or .webm video file to scrub and capture a frame for detection.</p>
                <Button variant="secondary" icon={Upload} onclick={() => videoInput.click()}>
                  Choose video file
                </Button>
              </div>
            )}
            {videoErr.get() ? (
              <p class="cap-err" role="alert">
                {videoErr.get()}
              </p>
            ) : null}
          </div>
        ) : (
          <div class="cap-stage">
            <video ref={(el) => (camEl = el)} muted playsinline class:hidden={!camActive.get()}></video>
            {camActive.get() && camErr.get() ? (
              <p class="cap-err" role="alert">
                {camErr.get()}
              </p>
            ) : null}
            {!camActive.get() ? (
              <div class="cap-empty">
                <p role={camErr.get() ? 'alert' : undefined}>
                  {!canCamera.get()
                    ? 'Camera capture is unavailable in this browser or context (requires HTTPS or localhost with camera hardware).'
                    : camErr.get() ?? 'Start the camera to preview and capture a single frame.'}
                </p>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </Dialog>
  );
}

css`
  .cap-root {
    display: grid;
    gap: var(--space-4);
  }
  .cap-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .cap-stage {
    display: grid;
    gap: var(--space-3);
  }
  video {
    width: 100%;
    max-height: 360px;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    object-fit: contain;
  }
  video.hidden {
    display: none;
  }
  .cap-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    min-height: 200px;
    padding: var(--space-6);
    border-radius: var(--radius-md);
    border: 1px dashed var(--line-default);
    background: var(--bg-inset);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    text-align: center;
  }
  .scrub {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  .scrub-controls {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .time {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .cap-err {
    color: var(--danger-text);
    font-size: var(--type-sm-size);
  }
`;
