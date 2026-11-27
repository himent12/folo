/**
 * Inline transport & status bar rendered under Playground's stage when a local video file or live webcam is active.
 * - Video mode: Play/Pause, Step -1 / +1 frame, timeline scrubber, timestamp & frame readout, Auto-detect toggle,
 *   and on-demand Detect frame trigger.
 * - Webcam mode: Live/Paused badge, rolling FPS & latency readout, Pause/Resume, and Stop camera.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import CircleStop from '$icons/circle-stop';
import Play from '$icons/play';
import Scan from '$icons/scan';
import Square from '$icons/square';
import { css } from 'fw';
import Button from '$lib/ui/Button';
import Slider from '$lib/ui/Slider';
import Switch from '$lib/ui/Switch';
import { formatMediaTime, frameIndexAt, totalFramesFor } from './media';
import type { MediaSession } from './useMediaSession';

export interface MediaBarProps {
  media: MediaSession;
  predicting: boolean;
}

export default function MediaBar(props: MediaBarProps) {
  const m = props.media;

  function onVideoKey(e: KeyboardEvent) {
    if (m.mode.peek() !== 'video' || !m.videoReady.peek()) return;
    // Never hijack native keys (Space/Arrow/Enter) from interactive buttons, switches, sliders, or inputs.
    if ((e.target as HTMLElement | null)?.closest('button, input, select, textarea, a, [role="switch"], [role="slider"]')) return;
    if (e.key === ' ' || e.key === 'k') {
      e.preventDefault();
      m.togglePlay();
    } else if (e.key === 'ArrowLeft' || e.key === ',') {
      e.preventDefault();
      m.stepFrame(-1);
    } else if (e.key === 'ArrowRight' || e.key === '.') {
      e.preventDefault();
      m.stepFrame(1);
    }
  }

  return (
    <>
      {m.mode.get() === 'video' ? (
        <div class="media-bar" data-media-bar="video" onkeydown={onVideoKey}>
          <div class="media-row">
            <div class="transport">
              <Button
                size="sm"
                variant="secondary"
                icon={m.playing.get() ? Square : Play}
                aria-label={m.playing.get() ? 'Pause video' : 'Play video'}
                disabled={!m.videoReady.get()}
                onclick={() => m.togglePlay()}
              >
                {m.playing.get() ? 'Pause' : 'Play'}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                icon={ChevronLeft}
                aria-label="Step back 1 frame"
                disabled={!m.videoReady.get() || m.seeking.get()}
                onclick={() => m.stepFrame(-1)}
              >
                −1f
              </Button>
              <Button
                size="sm"
                variant="ghost"
                iconRight={ChevronRight}
                aria-label="Step forward 1 frame"
                disabled={!m.videoReady.get() || m.seeking.get()}
                onclick={() => m.stepFrame(1)}
              >
                +1f
              </Button>
            </div>
            <span class="readout num">
              {formatMediaTime(m.currentTime.get())} / {formatMediaTime(m.duration.get())}
              {(() => {
                const step = m.frameStep.get();
                const total = totalFramesFor(m.duration.get(), step);
                return total > 0 ? (
                  <span class="faint">
                    {' '}
                    · f {frameIndexAt(m.currentTime.get(), step, total)} / {total}
                  </span>
                ) : null;
              })()}
            </span>
            <div class="media-right">
              {m.playing.get() && m.fps.get() > 0 ? (
                <span class="live-stat num">
                  {m.fps.get().toFixed(1)} FPS · {Math.round(m.latencyMs.get())} ms
                </span>
              ) : null}
              <div class="auto-wrap">
                <Switch label="Auto" bind:checked={m.autoDetect} />
              </div>
              <Button
                size="sm"
                variant="ghost"
                icon={Scan}
                disabled={!m.videoReady.get() || m.seeking.get() || props.predicting}
                onclick={() => m.detectCurrentFrame()}
              >
                Detect frame
              </Button>
            </div>
          </div>
          {m.duration.get() > 0 ? (
            <div class="scrubber">
              <Slider
                inline
                label="Timeline"
                value={m.currentTime.get()}
                min={0}
                max={m.duration.get()}
                step={Math.max(0.001, Math.round(m.frameStep.get() * 10000) / 10000)}
                format={formatMediaTime}
                oninput={(v) => m.seekTo(v, false)}
              />
            </div>
          ) : null}
        </div>
      ) : m.mode.get() === 'webcam' ? (
        <div class="media-bar webcam" data-media-bar="webcam">
          <div class="media-row">
            <span class="cam-state num">
              <span class="dot" class:on={m.camActive.get() && !m.camPaused.get()} class:paused={m.camPaused.get()} aria-hidden="true"></span>
              {m.camStarting.get()
                ? 'Starting camera…'
                : m.camErr.get()
                  ? 'Camera unavailable'
                  : m.camPaused.get()
                    ? 'Camera paused'
                    : 'Live webcam'}
            </span>
            {m.camActive.get() && m.fps.get() > 0 ? (
              <span class="live-stat num" data-webcam-fps>
                {m.fps.get().toFixed(1)} FPS · {Math.round(m.latencyMs.get())} ms
              </span>
            ) : null}
            <div class="media-right">
              {m.camActive.get() ? (
                <Button
                  size="sm"
                  variant="secondary"
                  icon={m.camPaused.get() ? Play : Square}
                  onclick={() => m.toggleCamPause()}
                >
                  {m.camPaused.get() ? 'Resume' : 'Pause'}
                </Button>
              ) : null}
              <Button size="sm" variant="destructive" icon={CircleStop} onclick={() => m.stopWebcam()}>
                Stop camera
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

css`
  .media-bar {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-top: 1px solid var(--line-subtle);
    background: var(--bg-panel);
  }
  .media-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
  }
  .transport,
  .media-right {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .media-right {
    margin-left: auto;
    gap: var(--space-2);
  }
  .readout,
  .cam-state,
  .live-stat {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .faint {
    color: var(--text-tertiary);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-full);
    background: var(--text-tertiary);
  }
  .dot.on {
    background: var(--danger-solid);
  }
  .dot.paused {
    background: var(--warning-solid);
  }
  .auto-wrap {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
  }
  .scrubber :global(.slider.inline) {
    grid-template-columns: 64px minmax(0, 1fr) 48px;
  }
  .scrubber :global(output.val) {
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
  }
  @media (max-width: 767px) {
    .media-bar {
      padding: var(--space-2) var(--space-4);
    }
    .media-right {
      margin-left: 0;
    }
  }
`;
