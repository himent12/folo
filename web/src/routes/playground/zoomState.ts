// Shared zoom/pan state for single and synchronized side-by-side / split Playground viewports.
// Pan offsets (`ix`, `iy`) are stored normalized by the fitted image dimensions (`tx / fittedW`, `ty / fittedH`) when
// image dimensions are known, so switching between Side-by-side and Split keeps the zoomed image landmark centered.
import { signal, type Signal } from 'fw';

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;
export const ZOOM_STEP = 1.5;

export interface ZoomState {
  z: Signal<number>;
  /** Center-relative pan offset normalized by fitted image width (or viewport width when image dims are absent). */
  nx: Signal<number>;
  /** Center-relative pan offset normalized by fitted image height (or viewport height when image dims are absent). */
  ny: Signal<number>;
  glide: Signal<boolean>;
  /** Keeps overflow clipped during eased return-to-Fit transitions so scaled content never flashes outside. */
  clipping: Signal<boolean>;
  /** Latest measured viewport size for Split-overlay image-space export calculations. */
  viewportW: Signal<number>;
  viewportH: Signal<number>;
}

/** Creates an independent or shareable zoom/pan state starting at Fit (100%, centered). */
export function createZoomState(): ZoomState {
  return {
    z: signal(1),
    nx: signal(0),
    ny: signal(0),
    glide: signal(false),
    clipping: signal(false),
    viewportW: signal(0),
    viewportH: signal(0),
  };
}

/** Computes the contained (letterboxed) image rectangle inside a viewport `(vw, vh)`. */
export function fittedImageRect(
  vw: number,
  vh: number,
  imgW?: number,
  imgH?: number,
): { ox: number; oy: number; rw: number; rh: number } {
  if (!vw || !vh) return { ox: 0, oy: 0, rw: 0, rh: 0 };
  if (!imgW || !imgH) return { ox: 0, oy: 0, rw: vw, rh: vh };
  const s = Math.min(vw / imgW, vh / imgH);
  const rw = imgW * s;
  const rh = imgH * s;
  return { ox: (vw - rw) / 2, oy: (vh - rh) / 2, rw, rh };
}

/** Clamps a center-relative pixel pan offset so the zoomed canvas never pulls away from the stage edge. */
export function clampPan(offset: number, size: number, scale: number): number {
  if (!Number.isFinite(offset) || !Number.isFinite(size) || !Number.isFinite(scale) || scale <= 1.001 || size <= 0) return 0;
  const limit = (size * (scale - 1)) / 2;
  return Math.max(-limit, Math.min(limit, offset));
}

/** Converts a normalized image-space pan offset into clamped viewport pixels. */
export function panToPx(norm: number, basis: number, viewportSize: number, scale: number): number {
  if (!Number.isFinite(norm) || basis <= 0 || viewportSize <= 0 || scale <= 1.001) return 0;
  return clampPan(norm * basis, viewportSize, scale);
}

/** Converts a viewport-pixel pan offset into a normalized image-space pan offset. */
export function pxToNormPan(px: number, basis: number, viewportSize: number, scale: number): number {
  if (!Number.isFinite(px) || basis <= 0 || viewportSize <= 0 || scale <= 1.001) return 0;
  return clampPan(px, viewportSize, scale) / basis;
}

/**
 * Converts a live Split-overlay viewport percentage (0..100) into the exact image-space split percentage (0..100),
 * accounting for letterboxing, zoom scale, and horizontal pan so exported Split PNGs cut the same pixels as the viewer.
 */
export function viewportSplitToImagePct(opts: {
  splitPct: number;
  viewportW: number;
  viewportH: number;
  imgW: number;
  imgH: number;
  z: number;
  nx: number;
}): number {
  const pct = Math.max(0, Math.min(100, opts.splitPct));
  const { viewportW: vw, viewportH: vh, imgW, imgH, z, nx } = opts;
  if (!vw || !vh || !imgW || !imgH) return pct;
  const { ox, rw } = fittedImageRect(vw, vh, imgW, imgH);
  if (rw <= 0) return pct;
  const scale = Number.isFinite(z) && z > 1.001 ? z : 1;
  const tx = panToPx(nx, rw, vw, scale);
  const divX = (pct / 100) * vw;
  const canvasX = vw / 2 + (divX - vw / 2 - tx) / scale;
  const imgFrac = (canvasX - ox) / rw;
  return Math.max(0, Math.min(100, imgFrac * 100));
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
  state.nx.set(state.nx.peek() * ratio);
  state.ny.set(state.ny.peek() * ratio);
}

/** Resets a shared `ZoomState` back to Fit (100%, centered). */
export function resetZoomState(state: ZoomState, eased = true): void {
  state.glide.set(eased);
  state.z.set(1);
  state.nx.set(0);
  state.ny.set(0);
}
