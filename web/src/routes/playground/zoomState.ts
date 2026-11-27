// Shared zoom/pan state for single and synchronized side-by-side / split Playground viewports.
// Pan offsets are stored normalized by viewport dimensions (`nx = tx / width`, `ny = ty / height`) so two panes stay
// aligned even when their pixel widths differ or when switching between side-by-side and split-overlay views.
import { signal, type Signal } from 'fw';

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;
export const ZOOM_STEP = 1.5;

export interface ZoomState {
  z: Signal<number>;
  nx: Signal<number>;
  ny: Signal<number>;
  glide: Signal<boolean>;
  /** Keeps overflow clipped during eased return-to-Fit transitions so scaled content never flashes outside. */
  clipping: Signal<boolean>;
}

/** Creates an independent or shareable zoom/pan state starting at Fit (100%, centered). */
export function createZoomState(): ZoomState {
  return {
    z: signal(1),
    nx: signal(0),
    ny: signal(0),
    glide: signal(false),
    clipping: signal(false),
  };
}

/** Zooms a shared `ZoomState` around its viewport center by `factor`. */
export function zoomCenter(state: ZoomState, factor: number, eased = true): void {
  const cz = state.z.peek();
  const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, cz * factor));
  state.glide.set(eased);
  if (nz <= 1.001) {
    state.z.set(1);
    state.nx.set(0);
    state.ny.set(0);
    return;
  }
  const ratio = nz / cz;
  state.z.set(nz);
  state.nx.set(clampNormPan(state.nx.peek() * ratio, nz));
  state.ny.set(clampNormPan(state.ny.peek() * ratio, nz));
}

/** Resets a shared `ZoomState` back to Fit (100%, centered). */
export function resetZoomState(state: ZoomState, eased = true): void {
  state.glide.set(eased);
  state.z.set(1);
  state.nx.set(0);
  state.ny.set(0);
}

/** Clamps a normalized pan offset so a zoomed canvas never pulls away from its viewport edge. */
export function clampNormPan(offset: number, scale: number): number {
  if (!Number.isFinite(offset) || !Number.isFinite(scale) || scale <= 1.001) return 0;
  const limit = (scale - 1) / 2;
  return Math.max(-limit, Math.min(limit, offset));
}

/** Clamps a center-relative pixel pan offset so the zoomed canvas never pulls away from the stage edge. */
export function clampPan(offset: number, size: number, scale: number): number {
  if (!Number.isFinite(offset) || !Number.isFinite(size) || !Number.isFinite(scale) || scale <= 1.001 || size <= 0) return 0;
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
  const cz = Number.isFinite(current.z) && current.z > 0 ? Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.z)) : MIN_ZOOM;
  if (!Number.isFinite(factor) || factor <= 0) {
    return { z: cz, tx: clampPan(current.tx, width, cz), ty: clampPan(current.ty, height, cz) };
  }
  const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, cz * factor));
  if (nz <= 1.001) return { z: 1, tx: 0, ty: 0 };
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0 || !Number.isFinite(px) || !Number.isFinite(py)) {
    return { z: nz, tx: 0, ty: 0 };
  }
  const curTx = clampPan(current.tx, width, cz);
  const curTy = clampPan(current.ty, height, cz);
  const dx = px - width / 2;
  const dy = py - height / 2;
  const ux = (dx - curTx) / cz;
  const uy = (dy - curTy) / cz;
  return {
    z: nz,
    tx: clampPan(dx - ux * nz, width, nz),
    ty: clampPan(dy - uy * nz, height, nz),
  };
}
