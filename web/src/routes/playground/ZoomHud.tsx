/**
 * Compact zoom HUD (`-`, percentage/Fit readout, `+`, Reset) driving a shared `ZoomState` for Compare A/B viewports.
 * Rendered at `z-index: 5` above both layers and the split curtain so controls are never clipped or covered.
 */
import Maximize from '$icons/maximize';
import Minus from '$icons/minus';
import Plus from '$icons/plus';
import { computed, css, onCleanup, signal } from 'fw';
import Button from '$lib/ui/Button';
import { tooltip } from '$lib/ui/tooltip';
import { MAX_ZOOM, resetZoomState, ZOOM_STEP, zoomCenter, type ZoomState } from './zoomState';

export interface ZoomHudProps {
  zoom: ZoomState;
}

export default function ZoomHud(props: ZoomHudProps) {
  const hud = signal<HTMLDivElement | undefined>(undefined);
  const zoomed = computed(() => props.zoom.z.get() > 1.001);
  const zoomLabel = computed(() => (zoomed.get() ? `${Math.round(props.zoom.z.get() * 100)}%` : 'Fit'));

  let alive = true;
  onCleanup(() => {
    alive = false;
  });

  function keepFocus(fallback: 'in' | 'out') {
    const box = hud.peek();
    const active = document.activeElement as HTMLElement | null;
    if (!box || !active || !box.contains(active)) return;
    queueMicrotask(() => {
      if (!alive || !box.isConnected) return;
      const now = document.activeElement as HTMLElement | null;
      const lost = !now || now === document.body || (now === active && (active as HTMLButtonElement).disabled);
      if (!lost) return;
      box.querySelector<HTMLButtonElement>(`[data-zoom="${fallback}"]:not(:disabled)`)?.focus({ preventScroll: true });
    });
  }

  function step(factor: number) {
    zoomCenter(props.zoom, factor, true);
    const nz = props.zoom.z.peek();
    if (nz <= 1.001) keepFocus('in');
    else if (nz >= MAX_ZOOM - 0.001) keepFocus('out');
  }

  function reset() {
    resetZoomState(props.zoom, true);
    keepFocus('in');
  }

  return (
    <div class="zoom-hud" ref={hud} role="group" aria-label="Image zoom">
      <span use:tooltip={'Zoom out'}>
        <Button size="sm" variant="ghost" icon={Minus} data-zoom="out" aria-label="Zoom out" disabled={!zoomed.get()} onclick={() => step(1 / ZOOM_STEP)} />
      </span>
      <span class="zoom-tip" use:tooltip={zoomed.get() ? 'Reset zoom (double-click)' : 'Double-click image to zoom'}>
        <button
          type="button"
          class="zoom-pct num"
          disabled={!zoomed.get()}
          aria-label={zoomed.get() ? `Zoom ${Math.round(props.zoom.z.get() * 100)}%, reset to fit` : 'Zoom 100%, fitted'}
          onclick={reset}
        >
          {zoomLabel.get()}
        </button>
      </span>
      <span use:tooltip={'Zoom in'}>
        <Button size="sm" variant="ghost" icon={Plus} data-zoom="in" aria-label="Zoom in" disabled={props.zoom.z.get() >= MAX_ZOOM - 0.001} onclick={() => step(ZOOM_STEP)} />
      </span>
      <span use:tooltip={'Reset zoom (double-click)'}>
        <Button size="sm" variant="ghost" icon={Maximize} aria-label="Reset zoom" disabled={!zoomed.get()} onclick={reset} />
      </span>
    </div>
  );
}

css`
  .zoom-hud {
    position: absolute;
    right: var(--space-2);
    bottom: var(--space-2);
    z-index: 5;
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 2px;
    border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--bg-overlay) 92%, transparent);
    border: 1px solid var(--line-default);
    box-shadow: var(--shadow-md);
  }
  .zoom-hud > span {
    display: inline-flex;
  }
  .zoom-pct {
    min-width: 44px;
    height: var(--control-sm);
    padding: 0 var(--space-1);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-align: center;
    cursor: pointer;
    transition: background-color var(--dur-fast) var(--ease-standard), color var(--dur-fast) var(--ease-standard);
  }
  .zoom-pct:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .zoom-pct:disabled {
    cursor: default;
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .zoom-pct {
      height: 44px;
    }
  }
`;
