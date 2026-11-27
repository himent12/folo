/**
 * Inline transport & status bar rendered under Playground's stage when a local video file or live webcam is active.
 * - Video mode: Play/Pause, Step -1 / +1 frame, timeline scrubber, timestamp & frame readout, Auto-detect toggle,
 *   and on-demand Detect frame trigger.
 * - Webcam mode: Live/Paused badge, rolling FPS & latency readout, Pause/Resume, and Stop camera.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import CircleStop from '$icons/circle-stop';
import Download from '$icons/download';
import FileJson from '$icons/file-json';
import Play from '$icons/play';
import RotateCcw from '$icons/rotate-ccw';
import Scan from '$icons/scan';
import Square from '$icons/square';
import { createUniqueId, css, effect, For, onCleanup, signal } from 'fw';
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

/** Dedicated component so Switch DOM identity stays stable (no keyboard focus loss when toggled or track count updates). */
function TrackControl(props: { tracking: TrackingController; uid: string }) {
  const tr = props.tracking;
  const whyId = `${props.uid}-track-why`;
  return (
    <>
      {tr.enabled.get() ? (
        <span class="live-stat num" data-track-count>
          {tr.activeCount.get()} {tr.activeCount.get() === 1 ? 'track' : 'tracks'}
        </span>
      ) : null}
      <span
        class="auto-wrap track-wrap"
        class:disabled={!tr.supported.get()}
        use:tooltip={tr.disabledReason.get() ?? undefined}
        data-track-toggle
      >
        <Switch
          label="Track IDs"
          checked={tr.enabled.get()}
          disabled={!tr.supported.get()}
          describedby={!tr.supported.get() && tr.disabledReason.get() ? whyId : undefined}
          onchange={(v) => tr.setEnabled(v)}
        />
      </span>
      {!tr.supported.get() && tr.disabledReason.get() ? (
        <span class="sr-only" id={whyId}>
          {tr.disabledReason.get()}
        </span>
      ) : null}
    </>
  );
}

/** Live ByteTrack session summary & trajectory export controls rendered below transport controls while tracking is on. */
function TrackSummaryBar(props: { tracking: TrackingController }) {
  const tr = props.tracking;
  const s = () => tr.summary.get();
  const topClasses = () => s().by_class.slice(0, 4);
  const extraClasses = () => Math.max(0, s().by_class.length - 4);
  const allClassTitle = () =>
    s()
      .by_class.map((c) => `${c.class_name}: ${c.unique_tracks} unique (${c.observations} obs)`)
      .join(' · ');
  const hasObs = () => s().observations_recorded > 0;
  const canClear = () => hasObs() || s().active_tracks > 0 || s().frames_recorded > 0;

  return (
    <div class="track-strip" data-track-summary>
      <div class="track-stats num">
        <span class="track-metric" data-track-unique>
          {s().active_tracks} active · {s().unique_tracks} unique {s().unique_tracks === 1 ? 'ID' : 'IDs'}
          {hasObs() ? (
            <span class="faint">
              {' '}
              · {s().observations_recorded} {s().observations_recorded === 1 ? 'obs' : 'obs'}
              {s().truncated ? ' (recent cap)' : ''}
            </span>
          ) : null}
        </span>
        {topClasses().length > 0 ? (
          <span class="track-classes" title={allClassTitle()} data-track-classes>
            <For each={topClasses()} key={(c) => c.class_id}>
              {(c) => (
                <span class="track-pill">
                  <span class="truncate">{c().class_name}</span> <b>{c().unique_tracks}</b>
                </span>
              )}
            </For>
            {extraClasses() > 0 ? <span class="track-pill faint">+{extraClasses()} more</span> : null}
          </span>
        ) : null}
      </div>
      <div class="track-actions">
        <Button
          size="sm"
          variant="ghost"
          icon={Download}
          disabled={!hasObs()}
          data-export-tracks="csv"
          aria-label="Export tracks CSV"
          onclick={() => tr.exportTracksCsv()}
        >
          Export tracks CSV
        </Button>
        <Button
          size="sm"
          variant="ghost"
          icon={FileJson}
          disabled={!hasObs()}
          data-export-tracks="json"
          aria-label="Export tracks JSON"
          onclick={() => tr.exportTracksJson()}
        >
          Export tracks JSON
        </Button>
        <Button
          size="sm"
          variant="ghost"
          icon={RotateCcw}
          disabled={!canClear()}
          data-clear-tracks
          aria-label="Clear tracks"
          onclick={() => tr.clearTracks()}
        >
          Clear tracks
        </Button>
      </div>
    </div>
  );
}

export default function MediaBar(props: MediaBarProps) {
  const m = props.media;
  const uid = createUniqueId();
  const barEl = signal<HTMLDivElement | undefined>(undefined);
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
              {props.tracking ? <TrackControl tracking={props.tracking} uid={uid} /> : null}
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
          {props.tracking && props.tracking.enabled.get() ? <TrackSummaryBar tracking={props.tracking} /> : null}
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
              {props.tracking ? <TrackControl tracking={props.tracking} uid={uid} /> : null}
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
          {props.tracking && props.tracking.enabled.get() ? <TrackSummaryBar tracking={props.tracking} /> : null}
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
  .track-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .track-stats {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .track-metric {
    white-space: nowrap;
  }
  .track-classes {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    min-width: 0;
  }
  .track-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    max-width: 140px;
    padding: 0 6px;
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-subtle);
    background: var(--bg-app);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    white-space: nowrap;
  }
  .track-pill b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .track-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1);
    margin-left: auto;
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
    .media-right,
    .track-actions {
      margin-left: 0;
    }
  }
`;
