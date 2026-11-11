/**
 * Narrow screens: the bar fixed under the form with the run's estimate (opening the Summary sheet) and Start, or the
 * change that fits; it steps aside while the Summary's own Start is on screen.
 */
import ChevronUp from '$icons/chevron-up';
import Play from '$icons/play';
import TriangleAlert from '$icons/triangle-alert';
import { bindable, css, type Signal } from 'fw';
import type { Arch, Dataset } from '$lib/api/types';
import { bytes, usdRange } from '$lib/format';
import { useLabel, type OomFix } from '$lib/oom';
import { prefs } from '$lib/prefs';
import { timeRange } from '$lib/throughput';
import { dock } from '$lib/shell/dock';
import { sidebar } from '$lib/shell/sidebarState';
import Button from '$lib/ui/Button';
import Spinner from '$lib/ui/Spinner';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import { tooltip } from '$lib/ui/tooltip';
import type { ShortRun } from './estimate';
import { START_KEYS } from './startKeys';
import { roughBytes, type Shown } from './summary';

export interface StartDockProps {
  startVisible: boolean;
  showWhy: boolean;
  why: { text: string } | null;
  /** The reason Start is off, as the page draws it beside its other Start buttons. */
  whyLine: (id: string, cls?: string) => JSX.Element;
  sumOpen?: boolean;
  shown: Shown;
  estReady: boolean;
  estStale: boolean;
  cloudCost: { lo: number; hi: number } | null;
  memShown: boolean;
  noFit: boolean;
  gpuMem: number | null;
  vram: number;
  short: ShortRun | null;
  dataset: Dataset | undefined;
  arch: Arch;
  fix: OomFix | null;
  fixLoading: boolean;
  applyFix: (f: OomFix) => void;
  submitting: boolean;
  valid: boolean;
}

export default function StartDock(props: StartDockProps) {
  // Bound by the page (`$bindable` in the Svelte original): never undefined.
  const sumOpen = bindable(props, 'sumOpen') as Signal<boolean>;

  // Narrow screens: the summary sits below the form, so keep the essentials and the action in reach.
  // use:dock: the phone tab bar steps aside while this bar is on the page (lib/shell/dock.ts).
  return (
    <div
      use:dock
      data-tip-chrome
      class="dock cta-row"
      class:away={props.startVisible}
      role="region"
      aria-label="Start training"
      inert={props.startVisible}
      style={{ '--dock-left': sidebar.collapsed ? 'var(--size-sidebar-collapsed)' : 'var(--size-sidebar)' }}
    >
      {/* The reason Start is off travels with it (the Summary's line is out of view here), in the estimate's place: one
          row of chrome, not two. */}
      {props.showWhy && props.why ? (
        props.whyLine('dock-why', 'dock-why')
      ) : (
        <button type="button" class="dock-est num" aria-haspopup="dialog" aria-label="Show the run summary" onclick={() => sumOpen.set(true)}>
          <Meta>
            {/* Kept while these settings' answer is out: dimmed, with the spinner the Summary's "Updating…" carries. */}
            <span class:rough={props.shown.rough || !props.estReady} class:stale={props.estStale}>
              {props.estReady && props.estStale ? (
                <span class="dock-spin">
                  <Spinner size={12} />
                </span>
              ) : null}
              {props.estReady ? timeRange(props.shown.lo, props.shown.hi) : 'Estimating…'}
            </span>
            {/* Over the memory: the figure against the GPU's in ink, and the verdict with its mark in the rail's one
                severity. */}
            {props.estReady && props.cloudCost ? <span class:stale={props.estStale}>{usdRange(props.cloudCost.lo, props.cloudCost.hi)}</span> : null}
            {props.estReady && props.memShown ? (
              <span class:stale={props.estStale}>
                {props.noFit && props.gpuMem ? (
                  <>
                    {roughBytes(props.vram)} / {bytes(props.gpuMem, prefs.units)}
                  </>
                ) : (
                  <>{roughBytes(props.vram)} GPU memory</>
                )}
              </span>
            ) : null}
            {props.estReady && props.noFit ? (
              <span class="dock-warn danger">
                <TriangleAlert size={12} aria-hidden="true" />
                Likely won't fit
              </span>
            ) : null}
            {props.short ? (
              <span class="dock-warn">
                <TriangleAlert size={12} aria-hidden="true" />
                {props.short.scratch ? 'Too short to learn' : 'Mostly warmup'}
              </span>
            ) : null}
            {props.dataset ? (
              <span class="faint">
                <Meta line>
                  <span>{props.dataset.name}</span>
                  <Ident value={props.arch} />
                </Meta>
              </span>
            ) : null}
          </Meta>
          <span class="dock-more">
            <ChevronUp size={14} strokeWidth={1.75} aria-hidden="true" />
          </span>
        </button>
      )}
      {props.noFit && (props.fix || props.fixLoading) ? (
        // The fix is the dock's one button; Start, secondary with the risk beside it, is in the summary it opens.
        <Button variant="primary" size="lg" loading={props.fixLoading} onclick={() => props.fix && props.applyFix(props.fix)}>
          {props.fix ? useLabel(props.fix) : 'Finding a fit'}
        </Button>
      ) : (
        <span class="start-tip" use:tooltip={{ text: 'Start training', keys: START_KEYS }}>
          <Button
            type="submit"
            variant="primary"
            size="lg"
            icon={Play}
            loading={props.submitting}
            disabled={!props.valid}
            aria-describedby={props.showWhy ? 'dock-why' : undefined}
          >
            Start training
          </Button>
        </span>
      )}
    </div>
  );
}

css`
  /* A figure for earlier settings while these settings' is out (\`estStale\`): kept, so nothing jumps, but tertiary, so it
     never reads as current. The dock's figures dim the same way. */
  .dock-est .stale {
    color: var(--text-tertiary);
  }
  .dock-spin {
    display: inline-flex;
    margin-right: var(--space-1);
    vertical-align: -2px;
  }
  .dock {
    display: none;
  }
  .start-tip {
    display: inline-flex;
  }
  @media (max-width: 1023px) {
    /* Beside the sidebar (its content may start at the left edge, so the bar starts where the page does). */
    .dock {
      position: fixed;
      left: var(--dock-left, 0);
      right: 0;
      bottom: 0;
      z-index: var(--z-sticky);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: var(--space-4);
      padding: var(--space-3) var(--space-4);
      background: color-mix(in srgb, var(--bg-app) 90%, transparent);
      backdrop-filter: blur(12px) saturate(1.4);
      -webkit-backdrop-filter: blur(12px) saturate(1.4);
      border-top: 1px solid var(--line-subtle);
      transition:
        transform var(--dur-base) var(--ease-out),
        opacity var(--dur-base) var(--ease-out);
    }
    .dock.away {
      transform: translateY(100%);
      opacity: 0;
    }
    .dock-est {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: var(--control-lg);
      padding: 0 var(--space-2);
      border-radius: var(--radius-md);
      color: var(--text-secondary);
      font-size: var(--type-sm-size);
      text-align: left;
    }
    .dock-est:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
    .dock-more {
      display: grid;
      color: var(--text-tertiary);
    }
    /* The dock's warning word in ink; its icon carries the status's hue. */
    .dock-warn {
      color: var(--text-secondary);
    }
    .dock-warn > :global(svg) {
      color: var(--warning-text);
    }
    .dock-warn.danger > :global(svg) {
      color: var(--danger-text);
    }
    .dock-warn {
      display: inline-flex;
      align-items: center;
      gap: var(--space-1);
    }
  }
  .dock-est .faint {
    display: none;
  }
  @media (max-width: 767px) {
    /* Two columns: the estimate wraps on the left, a 44 px Start button holds the right; 16 px gutters. */
    .dock {
      left: 0;
      bottom: calc(var(--tabbar-h) + env(safe-area-inset-bottom));
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--space-3);
      padding: var(--space-2) var(--space-4);
    }
    .dock-est {
      position: relative;
      flex-wrap: wrap;
      gap: 0 var(--space-3);
      min-width: 0;
      margin-left: calc(-1 * var(--space-2));
      padding: var(--space-1) var(--space-6) var(--space-1) var(--space-2);
      line-height: var(--type-sm-line);
    }
    .dock-more {
      position: absolute;
      right: var(--space-2);
      top: 50%;
      transform: translateY(-50%);
    }
    .dock-est .faint {
      display: block;
      flex-basis: 100%;
      font-size: var(--type-sm-size);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
  }
`;
