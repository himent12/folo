/**
 * Capture a single frame into Playground from a local video file (.mp4/.webm/.mov with frame scrubbing) or live
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
const FRAME_STEP_S = 1 / 30;

function formatSec(s: number): string {
  if (!Number.isFinite(s) || s < 0) return '0:00.0';
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return `${m}:${rem < 10 ? '0' : ''}${rem.toFixed(1)}`;
}

/** Captures the current rendered frame of an HTMLVideoElement as a PNG blob. */
async function videoFrameToPng(video: HTMLVideoElement): Promise<Blob> {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) throw new Error('Video frame is not ready yet.');
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

  // Video file state
  let videoInput!: HTMLInputElement;
  let videoEl: HTMLVideoElement | undefined;
  const videoUrl = signal<string | null>(null);
  const videoName = signal('');
  const duration = signal(0);
  const currentTime = signal(0);
  const playing = signal(false);
  const videoReady = signal(false);
  const videoErr = signal<string | null>(null);

  // Camera state
  let camEl: HTMLVideoElement | undefined;
  let stream: MediaStream | null = null;
  const camActive = signal(false);
  const camBusy = signal(false);
  const camErr = signal<string | null>(null);

  const captureBusy = signal(false);
  const canCamera = computed(() => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia);

  function stopCamera() {
    if (stream) {
      for (const t of stream.getTracks()) t.stop();
      stream = null;
    }
    if (camEl) camEl.srcObject = null;
    camActive.set(false);
    camBusy.set(false);
  }

  function clearVideo() {
    const old = videoUrl.peek();
    if (old) URL.revokeObjectURL(old);
    videoUrl.set(null);
    videoName.set('');
    duration.set(0);
    currentTime.set(0);
    playing.set(false);
    videoReady.set(false);
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
      stopCamera();
      if (videoEl && !videoEl.paused) videoEl.pause();
      playing.set(false);
    }
  });

  effect(() => {
    if (mode.get() !== 'camera') stopCamera();
  });

  onCleanup(() => {
    stopCamera();
    clearVideo();
  });

  async function startCamera() {
    if (!canCamera.peek()) {
      camErr.set('Camera capture is not available in this browser environment.');
      return;
    }
    stopCamera();
    camErr.set(null);
    camBusy.set(true);
    try {
      const next = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      if (!open.peek() || mode.peek() !== 'camera') {
        for (const t of next.getTracks()) t.stop();
        return;
      }
      stream = next;
      if (camEl) {
        camEl.srcObject = next;
        await camEl.play().catch(() => {});
      }
      camActive.set(true);
    } catch (e) {
      const msg = e instanceof Error && e.message ? e.message : 'Camera permission was denied or no camera was found.';
      camErr.set(msg);
    } finally {
      camBusy.set(false);
    }
  }

  function seekTo(sec: number) {
    const max = duration.peek();
    const clamped = Math.max(0, Math.min(max || 0, sec));
    currentTime.set(clamped);
    if (videoEl && Number.isFinite(clamped)) videoEl.currentTime = clamped;
  }

  function togglePlay() {
    if (!videoEl || !videoReady.peek()) return;
    if (videoEl.paused) void videoEl.play().catch(() => {});
    else videoEl.pause();
  }

  async function captureCurrent() {
    const el = mode.peek() === 'video' ? videoEl : camEl;
    if (!el || captureBusy.peek()) return;
    captureBusy.set(true);
    try {
      const blob = await videoFrameToPng(el);
      const name =
        mode.peek() === 'video'
          ? `${videoName.peek().replace(/\.[^.]+$/, '') || 'video'}-${currentTime.peek().toFixed(1)}s.png`
          : `camera-snapshot.png`;
      stopCamera();
      open.set(false);
      props.onCapture(blob, name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Could not capture frame.';
      if (mode.peek() === 'video') videoErr.set(msg);
      else camErr.set(msg);
    } finally {
      captureBusy.set(false);
    }
  }

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
          <Button
            variant="primary"
            icon={Scan}
            disabled={captureBusy.get() || (mode.get() === 'video' ? !videoReady.get() : !camActive.get())}
            onclick={() => void captureCurrent()}
          >
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
            <Button size="sm" variant={camActive.get() ? 'secondary' : 'primary'} disabled={camBusy.get() || !canCamera.get()} onclick={() => (camActive.get() ? stopCamera() : void startCamera())}>
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
                  preload="metadata"
                  onloadedmetadata={(e) => {
                    const v = e.currentTarget;
                    duration.set(Number.isFinite(v.duration) ? v.duration : 0);
                    videoReady.set(v.videoWidth > 0);
                    videoErr.set(null);
                  }}
                  onloadeddata={(e) => {
                    videoReady.set(e.currentTarget.videoWidth > 0);
                  }}
                  ontimeupdate={(e) => {
                    currentTime.set(e.currentTarget.currentTime);
                  }}
                  onplay={() => playing.set(true)}
                  onpause={() => playing.set(false)}
                  onerror={() => {
                    videoReady.set(false);
                    videoErr.set('This browser could not decode the selected video file.');
                  }}
                ></video>
                <div class="scrub">
                  <div class="scrub-controls">
                    <Button size="sm" variant="ghost" icon={playing.get() ? Pause : Play} aria-label={playing.get() ? 'Pause video' : 'Play video'} onclick={togglePlay} />
                    <Button size="sm" variant="ghost" onclick={() => seekTo(currentTime.peek() - FRAME_STEP_S)}>
                      -1f
                    </Button>
                    <Button size="sm" variant="ghost" onclick={() => seekTo(currentTime.peek() + FRAME_STEP_S)}>
                      +1f
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
            {videoErr.get() ? <p class="cap-err">{videoErr.get()}</p> : null}
          </div>
        ) : (
          <div class="cap-stage">
            <video ref={(el) => (camEl = el)} muted playsinline class:hidden={!camActive.get()}></video>
            {!camActive.get() ? (
              <div class="cap-empty">
                <p>
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
