/**
 * Interactive zoom and pan viewport for the Playground image canvas.
 * Supports wheel zoom (cursor-anchored), drag-to-pan when zoomed (preserving box clicks under the drag slop
 * threshold), double-click to zoom in or reset back to Fit, compact +/-/Reset controls, and automatic reset when the
 * displayed image source changes.
 */
import Maximize from '$icons/maximize';
import Minus from '$icons/minus';
import Plus from '$icons/plus';
import { computed, css, effect, onCleanup, signal, untrack } from 'fw';
import Button from '$lib/ui/Button';
import { tooltip } from '$lib/ui/tooltip';

export interface ZoomStageProps {
  /** Changing this key resets zoom and pan back to Fit (e.g. when a new image is loaded). */
  resetKey: string | null;
  children?: JSX.Element;
}

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;
const ZOOM_STEP = 1.5;
const DBLCLICK_ZOOM = 2;
const CLICK_SLOP = 4;
const CLIP_FALLBACK_MS = 220;

/** Clamps a center-relative pan offset so the zoomed canvas never pulls away from the stage edge. */
export function clampPan(offset: number, size: number, scale: number): number {
  if (scale <= 1.001 || size <= 0) return 0;
  const limit = (size * (scale - 1)) / 2;
  return Math.max(-limit, Math.min(limit, offset));
}

/** Computes the next (scale, tx, ty) when zooming around a viewport-local point `(px, py)`. */
export function zoomToward(
  current: { z: number; tx: number; ty: number },
  factor: number,
  px: number,
  py: number,
  width: number,
  height: number,
): { z: number; tx: number; ty: number } {
  const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.z * factor));
  if (nz <= 1.001) return { z: 1, tx: 0, ty: 0 };
  if (width <= 0 || height <= 0) return { z: nz, tx: 0, ty: 0 };
  const curTx = clampPan(current.tx, width, current.z);
  const curTy = clampPan(current.ty, height, current.z);
  const dx = px - width / 2;
  const dy = py - height / 2;
  const ux = (dx - curTx) / current.z;
  const uy = (dy - curTy) / current.z;
  return {
    z: nz,
    tx: clampPan(dx - ux * nz, width, nz),
    ty: clampPan(dy - uy * nz, height, nz),
  };
}

export default function ZoomStage(props: ZoomStageProps) {
  const viewport = signal<HTMLDivElement | undefined>(undefined);
  const vw = signal(0);
  const vh = signal(0);
  const z = signal(1);
  const tx = signal(0);
  const ty = signal(0);
  const glide = signal(false);
  const panning = signal(false);
  /** Keeps overflow clipped during the eased return transition back to Fit so scaled content never flashes outside. */
  const clipping = signal(false);

  const zoomed = computed(() => z.get() > 1.001);
  const zoomLabel = computed(() => (zoomed.get() ? `${Math.round(z.get() * 100)}%` : 'Fit'));
  const canvasStyle = computed(() => {
    const scale = z.get();
    if (scale <= 1.001) return undefined;
    const x = clampPan(tx.get(), vw.get(), scale);
    const y = clampPan(ty.get(), vh.get(), scale);
    return { transform: `translate(${x}px, ${y}px) scale(${scale})` };
  });

  let drag: { id: number; x: number; y: number; startTx: number; startTy: number; moved: boolean; captured: boolean } | null = null;
  let windowDragCleanup: (() => void) | null = null;
  let suppressClick = false;
  let suppressTimer: ReturnType<typeof setTimeout> | undefined;
  let clipTimer: ReturnType<typeof setTimeout> | undefined;

  function clearSuppress() {
    suppressClick = false;
    if (suppressTimer !== undefined) {
      clearTimeout(suppressTimer);
      suppressTimer = undefined;
    }
  }

  function clearClipTimer() {
    if (clipTimer !== undefined) {
      clearTimeout(clipTimer);
      clipTimer = undefined;
    }
  }

  function detachWindowDrag() {
    windowDragCleanup?.();
    windowDragCleanup = null;
  }

  function releaseDragCapture(pointerId: number) {
    try {
      viewport.peek()?.releasePointerCapture?.(pointerId);
    } catch {
      // Ignore if capture was already released.
    }
  }

  function armReturnClip() {
    clearClipTimer();
    clipping.set(true);
    clipTimer = setTimeout(() => {
      clipTimer = undefined;
      clipping.set(false);
    }, CLIP_FALLBACK_MS);
  }

  onCleanup(() => {
    clearSuppress();
    clearClipTimer();
    detachWindowDrag();
    if (drag?.captured) releaseDragCapture(drag.id);
    drag = null;
  });

  function stageSize(): { width: number; height: number; left: number; top: number } {
    const el = viewport.peek();
    const r = el?.getBoundingClientRect();
    const width = r && r.width > 0 ? r.width : vw.peek();
    const height = r && r.height > 0 ? r.height : vh.peek();
    if (width > 0 && vw.peek() !== width) vw.set(width);
    if (height > 0 && vh.peek() !== height) vh.set(height);
    return { width, height, left: r?.left ?? 0, top: r?.top ?? 0 };
  }

  function reset(eased = true) {
    if (drag?.captured) releaseDragCapture(drag.id);
    drag = null;
    detachWindowDrag();
    panning.set(false);
    const wasZoomed = z.peek() > 1.001;
    if (eased && wasZoomed) {
      armReturnClip();
    } else {
      clearClipTimer();
      clipping.set(false);
    }
    glide.set(eased);
    z.set(1);
    tx.set(0);
    ty.set(0);
  }

  function zoomAt(factor: number, px?: number, py?: number, eased = true) {
    const { width, height } = stageSize();
    const scale = z.peek();
    const ax = px ?? width / 2;
    const ay = py ?? height / 2;
    const next = zoomToward(
      { z: scale, tx: clampPan(tx.peek(), width, scale), ty: clampPan(ty.peek(), height, scale) },
      factor,
      ax,
      ay,
      width,
      height,
    );
    if (next.z <= 1.001) {
      if (eased && scale > 1.001) armReturnClip();
      else {
        clearClipTimer();
        clipping.set(false);
      }
    } else {
      clearClipTimer();
      clipping.set(false);
    }
    glide.set(eased);
    z.set(next.z);
    tx.set(next.tx);
    ty.set(next.ty);
  }

  // Track viewport size so pan clamping adapts on window resize.
  effect(() => {
    const el = viewport.get();
    if (!el) return;
    const sync = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 0) vw.set(r.width);
      if (r.height > 0) vh.set(r.height);
      const scale = z.peek();
      if (scale > 1.001) {
        tx.set(clampPan(tx.peek(), vw.peek(), scale));
        ty.set(clampPan(ty.peek(), vh.peek(), scale));
      }
    };
    sync();
    const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(sync);
    ro?.observe(el);
    return () => ro?.disconnect();
  });

  // Reset zoom and pan whenever a new image source loads (skip initial mount).
  let lastResetKey = untrack(() => props.resetKey);
  effect(() => {
    const next = props.resetKey;
    if (next === lastResetKey) return;
    lastResetKey = next;
    untrack(() => {
      clearSuppress();
      reset(false);
    });
  });

  // Non-passive wheel listener for cursor-anchored zooming without trapping unzoomed downward page scroll.
  effect(() => {
    const el = viewport.get();
    if (!el) return;
    const onwheel = (e: WheelEvent) => {
      if (e.deltaY === 0) return;
      if (e.target instanceof Element && e.target.closest('.zoom-hud')) return;
      if (!e.ctrlKey && z.peek() <= 1.001 && e.deltaY > 0) return;
      e.preventDefault();
      const { left, top, height } = stageSize();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? Math.max(height, 100) : 1;
      const factor = Math.exp(-e.deltaY * unit * (e.ctrlKey ? 0.01 : 0.002));
      zoomAt(factor, e.clientX - left, e.clientY - top, false);
    };
    const onClickCapture = (e: MouseEvent) => {
      if (!suppressClick) return;
      clearSuppress();
      e.preventDefault();
      e.stopPropagation();
    };
    el.addEventListener('wheel', onwheel, { passive: false });
    el.addEventListener('click', onClickCapture, true);
    return () => {
      el.removeEventListener('wheel', onwheel);
      el.removeEventListener('click', onClickCapture, true);
    };
  });

  function finishDrag(e: PointerEvent) {
    if (!drag || drag.id !== e.pointerId) return;
    const moved = drag.moved;
    const captured = drag.captured;
    drag = null;
    detachWindowDrag();
    panning.set(false);
    if (captured) releaseDragCapture(e.pointerId);
    if (moved) {
      clearSuppress();
      suppressClick = true;
      suppressTimer = setTimeout(clearSuppress, 0);
    }
  }

  function onpointerdown(e: PointerEvent) {
    if (drag || e.button !== 0 || z.peek() <= 1.001) return;
    if (e.target instanceof Element && e.target.closest('.zoom-hud, button, a')) return;
    const { width, height } = stageSize();
    const scale = z.peek();
    drag = {
      id: e.pointerId,
      x: e.clientX,
      y: e.clientY,
      startTx: clampPan(tx.peek(), width, scale),
      startTy: clampPan(ty.peek(), height, scale),
      moved: false,
      captured: false,
    };
    detachWindowDrag();
    window.addEventListener('pointerup', finishDrag);
    window.addEventListener('pointercancel', finishDrag);
    windowDragCleanup = () => {
      window.removeEventListener('pointerup', finishDrag);
      window.removeEventListener('pointercancel', finishDrag);
    };
  }

  function onpointermove(e: PointerEvent) {
    if (!drag || drag.id !== e.pointerId) return;
    if (e.pointerType === 'mouse' && (e.buttons & 1) === 0) {
      finishDrag(e);
      return;
    }
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    if (!drag.moved) {
      if (Math.hypot(dx, dy) <= CLICK_SLOP) return;
      drag.moved = true;
      panning.set(true);
      glide.set(false);
      try {
        viewport.peek()?.setPointerCapture?.(e.pointerId);
        drag.captured = true;
      } catch {
        // Synthetic pointer events in tests may not have an active pointerId.
      }
    }
    const { width, height } = stageSize();
    const scale = z.peek();
    tx.set(clampPan(drag.startTx + dx, width, scale));
    ty.set(clampPan(drag.startTy + dy, height, scale));
  }

  function ondblclick(e: MouseEvent) {
    if (e.target instanceof Element && e.target.closest('.zoom-hud, button, a')) return;
    if (z.peek() > 1.001) {
      reset(true);
      return;
    }
    const { left, top } = stageSize();
    zoomAt(DBLCLICK_ZOOM, e.clientX - left, e.clientY - top, true);
  }

  function oncanvastransitionend(e: TransitionEvent) {
    if (e.target !== e.currentTarget) return;
    if (e.propertyName === 'transform' && z.peek() <= 1.001) {
      clearClipTimer();
      clipping.set(false);
    }
  }

  return (
    <div
      class="zoom-stage"
      class:zoomed={zoomed.get()}
      class:clip={zoomed.get() || clipping.get()}
      class:panning={panning.get()}
      ref={(el) => viewport.set(el)}
      onpointerdown={onpointerdown}
      onpointermove={onpointermove}
      onpointerup={finishDrag}
      onpointercancel={finishDrag}
      onlostpointercapture={finishDrag}
      ondblclick={ondblclick}
    >
      <div class="zoom-canvas" class:glide={glide.get()} style={canvasStyle.get()} ontransitionend={oncanvastransitionend}>
        {props.children}
      </div>
      <div class="zoom-hud" role="group" aria-label="Image zoom">
        <span use:tooltip={'Zoom out'}>
          <Button
            size="sm"
            variant="ghost"
            icon={Minus}
            aria-label="Zoom out"
            disabled={!zoomed.get()}
            onclick={() => zoomAt(1 / ZOOM_STEP)}
          />
        </span>
        <span class="zoom-tip" use:tooltip={zoomed.get() ? 'Reset zoom (double-click)' : 'Double-click image to zoom'}>
          <button
            type="button"
            class="zoom-pct num"
            disabled={!zoomed.get()}
            aria-label={zoomed.get() ? `Zoom ${Math.round(z.get() * 100)}%, reset to fit` : 'Zoom 100%, fitted'}
            onclick={() => reset(true)}
          >
            {zoomLabel.get()}
          </button>
        </span>
        <span use:tooltip={'Zoom in'}>
          <Button
            size="sm"
            variant="ghost"
            icon={Plus}
            aria-label="Zoom in"
            disabled={z.get() >= MAX_ZOOM - 0.001}
            onclick={() => zoomAt(ZOOM_STEP)}
          />
        </span>
        <span use:tooltip={'Reset zoom (double-click)'}>
          <Button
            size="sm"
            variant="ghost"
            icon={Maximize}
            aria-label="Reset zoom"
            disabled={!zoomed.get()}
            onclick={() => reset(true)}
          />
        </span>
      </div>
    </div>
  );
}

css`
  .zoom-stage {
    position: relative;
    isolation: isolate;
    width: 100%;
    height: 100%;
  }
  .zoom-stage.clip {
    overflow: hidden;
  }
  .zoom-stage.zoomed {
    cursor: grab;
    touch-action: none;
  }
  .zoom-stage.panning {
    cursor: grabbing;
    user-select: none;
  }
  .zoom-canvas {
    position: relative;
    z-index: 0;
    isolation: isolate;
    width: 100%;
    height: 100%;
    transform-origin: center center;
  }
  .zoom-canvas.glide {
    transition: transform var(--dur-fast) var(--ease-out);
  }
  @media (prefers-reduced-motion: reduce) {
    .zoom-canvas.glide {
      transition: none;
    }
  }
  .zoom-hud {
    position: absolute;
    right: var(--space-2);
    bottom: var(--space-2);
    z-index: 2;
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
