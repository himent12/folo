/**
 * Live canvas renderer for Playground's real-time webcam stream and playing local videos.
 * Paints decoded <video> frames and FOLO detection boxes/chips smoothly via requestAnimationFrame (zero
 * opacity flicker between inference ticks) while rendering a crisp LIVE / FPS / latency HUD and inline
 * error fallback when camera permissions are denied or no camera hardware is available.
 */
import RefreshCw from '$icons/refresh-cw';
import TriangleAlert from '$icons/triangle-alert';
import { css, effect, signal } from 'fw';
import type { ViewBox } from '$lib/detect/DetectionView';
import Button from '$lib/ui/Button';
import { drawAnnotations } from './export';

export interface LiveMediaCanvasProps {
  /** Backing <video> element playing the webcam MediaStream or local video file. */
  video: HTMLVideoElement | undefined;
  /** 'webcam' or 'video'. */
  mode: 'webcam' | 'video';
  /** True when the live stream/playback is actively advancing frames. */
  active: boolean;
  /** True while requesting camera permission / initializing stream. */
  starting?: boolean;
  /** True when live webcam is frozen on one frame. */
  paused?: boolean;
  /** True when showing a scrubbed/paused video preview whose detections have not run yet (Auto off). */
  preview?: boolean;
  /** Monotonic tick bumped on video seeked/loadeddata so paused previews repaint immediately. */
  renderTick?: number;
  /** Inline camera or video decode error message. */
  error?: string | null;
  /** Filtered visible detection boxes in native video pixel coordinates. */
  boxes: readonly ViewBox[];
  classes: readonly string[];
  labels: boolean;
  scores: boolean;
  /** Live rolling inference FPS (0 until first frame completes). */
  fps: number;
  /** Live rolling inference latency in ms. */
  latencyMs: number;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export default function LiveMediaCanvas(props: LiveMediaCanvasProps) {
  const canvasEl = signal<HTMLCanvasElement | undefined>(undefined);

  function paintOnce() {
    const canvas = canvasEl.peek();
    const video = props.video;
    if (!canvas || !video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);
    drawAnnotations(
      ctx,
      {
        boxes: props.boxes,
        classes: props.classes,
        labels: props.labels,
        scores: props.scores,
      },
      { ox: 0, oy: 0, width: w, height: h },
    );
  }

  // Repaint whenever canvas, boxes, labels, scores, renderTick, or active state change; animate smoothly at rAF while streaming.
  effect(() => {
    void canvasEl.get();
    void props.video;
    void props.boxes;
    void props.classes;
    void props.labels;
    void props.scores;
    void props.renderTick;
    const running = props.active && !props.paused && !props.error;
    paintOnce();
    if (!running) return;
    let raf = 0;
    const loop = () => {
      paintOnce();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  });

  return (
    <div class="live-wrap" data-live-media={props.mode}>
      {props.error ? (
        <div class="live-error" role="alert">
          <TriangleAlert size={20} strokeWidth={1.75} aria-hidden="true" />
          <p class="err-title">{props.mode === 'webcam' ? 'Camera unavailable' : 'Video unavailable'}</p>
          <p class="err-msg">{props.error}</p>
          <div class="err-actions">
            {props.mode === 'webcam' && props.onRetry ? (
              <Button size="sm" variant="primary" icon={RefreshCw} onclick={props.onRetry}>
                Try again
              </Button>
            ) : null}
            {props.onDismiss ? (
              <Button size="sm" variant="secondary" onclick={props.onDismiss}>
                Back to Playground
              </Button>
            ) : null}
          </div>
        </div>
      ) : props.starting ? (
        <div class="live-status" role="status">
          <span>Starting camera…</span>
        </div>
      ) : (
        <>
          <canvas ref={canvasEl} class="live-canvas" aria-label={props.mode === 'webcam' ? 'Live webcam detection feed' : 'Playing video detection feed'}></canvas>
          <div class="live-hud num" aria-live="off">
            <span class="dot" class:paused={!!props.paused || !!props.preview} aria-hidden="true"></span>
            <span class="badge-label">
              {props.mode === 'webcam' ? (props.paused ? 'PAUSED' : 'LIVE') : props.preview ? 'PREVIEW' : 'PLAYING'}
            </span>
            {props.fps > 0 && !props.preview ? <span class="metric">{props.fps.toFixed(1)} FPS</span> : null}
            {props.latencyMs > 0 && !props.preview ? <span class="metric">{Math.round(props.latencyMs)} ms</span> : null}
          </div>
        </>
      )}
    </div>
  );
}

css`
  .live-wrap {
    position: relative;
    width: 100%;
    height: 100%;
    display: grid;
    place-items: center;
    background: var(--bg-inset);
    overflow: hidden;
  }
  .live-canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .live-hud {
    position: absolute;
    top: var(--space-3);
    left: var(--space-3);
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: 4px var(--space-2);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
    border: 1px solid var(--line-default);
    box-shadow: var(--shadow-sm);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
    pointer-events: none;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-full);
    background: var(--danger-solid);
  }
  .dot.paused {
    background: var(--warning-solid);
  }
  .badge-label {
    font-weight: var(--weight-semibold);
    letter-spacing: 0.04em;
  }
  .metric {
    color: var(--text-secondary);
  }
  .live-status,
  .live-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    max-width: 420px;
    padding: var(--space-6);
    text-align: center;
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .live-error {
    color: var(--text-primary);
  }
  .live-error :global(svg) {
    color: var(--danger-text);
  }
  .err-title {
    font-size: var(--type-base-size);
    font-weight: var(--weight-medium);
  }
  .err-msg {
    color: var(--text-secondary);
  }
  .err-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
`;
