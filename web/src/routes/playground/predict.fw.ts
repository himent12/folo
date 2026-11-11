// fw twin of predict.ts (the same values, typed against the fw inference module).
import type { Prediction } from '$lib/api/types';
import type { Backend } from '$lib/inference/browser';

// The server is asked once with loose settings and no suppression (IoU 1 keeps every candidate); the controls below
// re-filter and run NMS in the browser. Suppressing on the server first would lose boxes a looser IoU should keep.
export const FETCH = { conf: 0.01, iou: 1, max_det: 1000 };
export const DEFAULTS = { conf: 0.25, iou: 0.7, maxDet: 300 };

export const BACKEND_NAME: Record<Backend, string> = { webgpu: 'WebGPU', cpu: 'CPU' };

/** The demo has no real networks: the browser path is simulated with the documented speeds (docs/WASM.md). */
export function simulate(p: Prediction, backend: Backend): Prediction {
  const t = backend === 'webgpu' ? { preprocess_ms: 2.5, inference_ms: 11.1, postprocess_ms: 0.2 } : { preprocess_ms: 6, inference_ms: 477, postprocess_ms: 8 };
  return { ...p, backend, device: backend === 'webgpu' ? 'gpu:0' : 'cpu', timing: { ...t, total_ms: t.preprocess_ms + t.inference_ms + t.postprocess_ms } };
}

export const px = (v: number) => Math.round(v);
/** Scores read the same everywhere here (table, the picked box's line, the label chips on the image): 2 decimals. */
export const score = (v: number) => v.toFixed(2);
