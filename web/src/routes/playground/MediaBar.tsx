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
import { createUniqueId, css, effect, onCleanup, signal } from 'fw';
import Button from '$lib/ui/Button';
import Slider from '$lib/ui/Slider';
import Switch from '$lib/ui/Switch';
import { tooltip } from '$lib/ui/tooltip';
import { formatMediaTime, frameIndexAt, totalFramesFor } from './media';
import type { MediaSession } from './useMediaSession';
import type { TrackingController } from './useTracking';

export interface MediaBarProps {
  media: MediaSession;
  predicting: boolean;
  tracking?: TrackingController;
}

type VideoAction = 'step-back' | 'step-fwd' | 'detect';

export default function MediaBar(props: MediaBarProps) {
  const m = props.media;
  const uid = createUniqueId();
  const barEl = signal<HTMLDivElement | undefined>(undefined);

  const trackControl = () => {
    const tr = props.tracking;
    if (!tr) return null;
    const supported = tr.supported.get();
    const reason = tr.disabledReason.get();
    const on = tr.enabled.get();
    const count = tr.activeCount.get();
    const whyId = `${uid}-track-why`;
    return (
      <>
        {on ? (
          <span class="live-stat num" data-track-count>
            {count} {count === 1 ? 'track' : 'tracks'}
          </span>
        ) : null}
        <span class="auto-wrap track-wrap" class:disabled={!supported} use:tooltip={reason ?? undefined} data-track-toggle>
          <Switch
            label="Track IDs"
            checked={on}
            disabled={!supported}
            describedby={!supported && reason ? whyId : undefined}
            onchange={(v) => tr.setEnabled(v)}
          />
        </span>
        {!supported && reason ? (
          <span class="sr-only" id={whyId}>
            {reason}
          </span>
        ) : null}
      </>
    );
  };
  let pendingFocus: { action: VideoAction; url: string | null; sawDisabled: boolean } | null = null;
  let focusSeq = 0;

  function onUserNav(e: Event) {
    if (!pendingFocus) return;
    // Ignore events targeting the currently armed button itself while it is still focused/activated,
    // and ignore focusin on document.body (user clicks on background fire pointerdown; Tab moves focus to controls).
    const t = e.target as Element | null;
    if (e.type === 'focusin' && t === document.body) return;
    if (t?.closest?.(`button[data-video-action="${pendingFocus.action}"]`)) return;
    cancelPendingFocus();
  }

  function detachNavListeners() {
    document.removeEventListener('pointerdown', onUserNav, true);
    document.removeEventListener('focusin', onUserNav, true);
  }

  function cancelPendingFocus() {
    focusSeq++;
    pendingFocus = null;
    detachNavListeners();
  }

  function armPendingFocus(action: VideoAction, sawDisabled: boolean): number {
    const seq = ++focusSeq;
    pendingFocus = { action, url: m.videoUrl.peek(), sawDisabled };
    document.addEventListener('pointerdown', onUserNav, true);
    document.addEventListener('focusin', onUserNav, true);
    return seq;
  }

  onCleanup(() => cancelPendingFocus());

  // Restore focus after a transient seek or inference cycle clears `disabled` on the activated video transport button,
  // canceled automatically on clip replacement, mode exit, or any user pointer/focus navigation elsewhere.
  effect(() => {
    const mode = m.mode.get();
    const url = m.videoUrl.get();
    const ready = m.videoReady.get();
    const seeking = m.seeking.get();
    const predicting = props.predicting;
    if (!pendingFocus) return;
    if (mode !== 'video' || !ready || url !== pendingFocus.url) {
      cancelPendingFocus();
      return;
    }
    const isEnabled = pendingFocus.action === 'detect' ? !seeking && !predicting : !seeking;
    if (!isEnabled) {
      pendingFocus.sawDisabled = true;
      return;
    }
    if (!pendingFocus.sawDisabled) return;
    const { action, url: expectedUrl } = pendingFocus;
    const seq = ++focusSeq;
    pendingFocus = null;
    detachNavListeners();
    queueMicrotask(() => {
      if (seq !== focusSeq || m.mode.peek() !== 'video' || m.videoUrl.peek() !== expectedUrl) return;
      const active = document.activeElement;
      if (active && active !== document.body) return;
      const btn = barEl.peek()?.querySelector<HTMLButtonElement>(`button[data-video-action="${action}"]:not(:disabled)`);
      btn?.focus({ preventScroll: true });
    });
  });

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
        <div class="media-bar" data-media-bar="video" ref={barEl} onkeydown={onVideoKey}>
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
                data-video-action="step-back"
                disabled={!m.videoReady.get() || m.seeking.get()}
                onclick={(e) => {
                  const hadFocus = document.activeElement === e.currentTarget;
                  m.stepFrame(-1);
                  if (hadFocus && m.seeking.peek()) armPendingFocus('step-back', true);
                }}
              >
                −1f
              </Button>
              <Button
                size="sm"
                variant="ghost"
                iconRight={ChevronRight}
                aria-label="Step forward 1 frame"
                data-video-action="step-fwd"
                disabled={!m.videoReady.get() || m.seeking.get()}
                onclick={(e) => {
                  const hadFocus = document.activeElement === e.currentTarget;
                  m.stepFrame(1);
                  if (hadFocus && m.seeking.peek()) armPendingFocus('step-fwd', true);
                }}
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
              {trackControl()}
              <Button
                size="sm"
                variant="ghost"
                icon={Scan}
                data-video-action="detect"
                disabled={!m.videoReady.get() || m.seeking.get() || props.predicting}
                onblur={(e) => {
                  if (pendingFocus?.action === 'detect' && !(e.currentTarget as HTMLButtonElement).disabled) {
                    cancelPendingFocus();
                  }
                }}
                onclick={(e) => {
                  if (document.activeElement === e.currentTarget) {
                    const seq = armPendingFocus('detect', false);
                    void m.detectCurrentFrame().then((sent) => {
                      if (!sent) {
                        if (pendingFocus && seq === focusSeq) cancelPendingFocus();
                        return;
                      }
                      queueMicrotask(() => {
                        if (pendingFocus && !pendingFocus.sawDisabled && seq === focusSeq && !props.predicting) {
                          cancelPendingFocus();
                        }
                      });
                    });
                  } else {
                    void m.detectCurrentFrame();
                  }
                }}
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
              {trackControl()}
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
  .track-wrap.disabled {
    cursor: not-allowed;
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
