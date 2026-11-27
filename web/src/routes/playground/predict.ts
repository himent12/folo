// Server predictions for the Playground: asked once with loose settings (`FETCH`), then filtered in the browser.
import { ApiError, type Api } from '$lib/api/client';
import type { MockApi } from '$lib/api/mock';
import type { PredictInput, PredictOptions, Prediction } from '$lib/api/types';
import {
  inferenceMetaOf,
  toServerTilePayload,
  toServerTtaPayload,
  type Backend,
  type BrowserEngine,
  type BrowserPrecision,
  type InferenceConfig,
} from '$lib/inference/browser';
import { buildWasmDetectOptions } from './folo';
import { imageBlob, sourceImageKey, type Source } from './source';

// The server is asked once with loose settings and no suppression (IoU 1 keeps every candidate); the controls below
// re-filter and run NMS in the browser. Suppressing on the server first would lose boxes a looser IoU should keep.
export const FETCH = { conf: 0.01, iou: 1, max_det: 1000 };
/**
 * Every candidate: the server's own cap before NMS (folo-vision's `NmsConfig::max_candidates`), more than a model has
 * anchors at 1280 px. Asked for only when FETCH's cut matters (`candidateFloor`).
 */
export const ALL = { ...FETCH, max_det: 30_000 };
export const DEFAULTS = { conf: 0.25, iou: 0.7, maxDet: 300 };

export const BACKEND_NAME: Record<Backend, string> = { webgpu: 'WebGPU', cpu: 'CPU' };

export type RunOn = 'server' | Backend;

/** The demo has no real networks: the browser path is simulated with the documented speeds (docs/WASM.md). */
export function simulate(p: Prediction, backend: Backend): Prediction {
  const t = backend === 'webgpu' ? { preprocess_ms: 2.5, inference_ms: 11.1, postprocess_ms: 0.2 } : { preprocess_ms: 6, inference_ms: 477, postprocess_ms: 8 };
  return { ...p, backend, device: backend === 'webgpu' ? 'gpu:0' : 'cpu', timing: { ...t, total_ms: t.preprocess_ms + t.inference_ms + t.postprocess_ms } };
}

/**
 * On a busy image FETCH's answer is full: its `max_det` highest-scoring candidates are not all of them. Filtering
 * them is still exact for a confidence above the lowest score among them (greedy NMS only ever looks at higher
 * scores), but not at or below it. That score, or null when the answer holds every candidate.
 */
export function candidateFloor(p: Prediction, cap = FETCH.max_det): number | null {
  if (p.boxes.length < cap) return null;
  return p.boxes.reduce((m, b) => Math.min(m, b.score), Infinity);
}

/** The server cannot send this model's file (no weights route, or it cannot read it): the user can pick it. */
export function noServerWeights(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 404 || e.status === 405 || e.status === 501) && e.code !== 'model_not_found';
}

/** One prediction of the Playground's image with `options`, wherever it runs: the server, or this browser. */
export function ask(r: {
  api: Api;
  engine: BrowserEngine;
  model: string;
  source: Source;
  on: RunOn;
  /** A .folo the user picked for this model (the browser runs it instead of the server's weights). */
  file: File | null;
  options: typeof FETCH;
  /** Optional per-class confidence overrides forwarded directly to `folo-wasm` on browser runs. */
  perClassConf?: Readonly<Record<number, number>> | null;
  /** Optional inclusive thresholding flag (`score >= conf`) forwarded to `folo-wasm`. */
  confInclusive?: boolean;
  /** Optional browser WASM compute precision override (`auto` | `f32` | `f16` | `int8`). */
  precision?: BrowserPrecision;
  /** Optional SAHI tiled or TTA multi-view inference settings (standard single-pass when omitted). */
  inference?: InferenceConfig;
  signal: AbortSignal;
}): Promise<Prediction> {
  const { api, model, source: s, on, options, inference, signal } = r;
  const input: PredictInput = s.kind === 'file' ? { kind: 'file', file: s.file } : { kind: 'dataset', dataset_id: s.ds.id, image_id: s.img.id };
  // SAHI and TTA apply `max_det` per tile/view BEFORE cross-tile/cross-view fusion, so a 1,000 cap can drop valid
  // boxes even when the merged output has < 1,000 boxes. Use the full candidate budget (`ALL.max_det`) on enhanced
  // runs; fusion merges overlapping duplicates before returning, keeping the payload small and avoiding silent truncation.
  const enhanced = inference?.mode === 'sahi' || inference?.mode === 'tta';
  const effectiveMaxDet = enhanced ? Math.max(options.max_det, ALL.max_det) : options.max_det;
  if (on === 'server') {
    const opts: PredictOptions = { model_id: model, ...options, max_det: effectiveMaxDet };
    if (inference?.mode === 'sahi') {
      opts.tile = toServerTilePayload(inference.tile, effectiveMaxDet);
    } else if (inference?.mode === 'tta') {
      opts.tta = toServerTtaPayload(inference.tta, effectiveMaxDet, options.iou);
    }
    const meta = inferenceMetaOf(inference);
    return api.predict(input, opts, signal).then((p) => (meta ? { ...p, inference: meta } : p));
  }
  if (api.mock) return (api as unknown as MockApi).predictInPage(input, { model_id: model, ...options }).then((p) => simulate(p, on));
  const wasmOpts = buildWasmDetectOptions({
    conf: options.conf,
    iou: options.iou,
    maxDet: effectiveMaxDet,
    perClassConf: r.perClassConf,
    confInclusive: r.confInclusive,
  });
  return (r.file ? Promise.resolve(null) : api.modelWeightsTag(model, signal)).then((tag) => {
    // Superseded while asking for the ETag: do not start loading or running a model nobody waits for.
    if (signal.aborted) throw new DOMException('Superseded', 'AbortError');
    return r.engine.predict({
      modelId: model,
      file: r.file,
      download: () => api.modelWeights(model),
      tag,
      backend: on,
      precision: r.precision ?? inference?.precision,
      image: () => imageBlob(s, signal),
      imageKey: sourceImageKey(s),
      options: {
        conf: wasmOpts.conf ?? options.conf,
        iou: wasmOpts.iou ?? options.iou,
        maxDet: wasmOpts.maxDet ?? effectiveMaxDet,
        ...(wasmOpts.perClassConf ? { perClassConf: wasmOpts.perClassConf } : {}),
        ...(wasmOpts.confInclusive !== undefined ? { confInclusive: wasmOpts.confInclusive } : {}),
      },
      inference,
      signal,
    });
  });
}

export const px = (v: number) => Math.round(v);
/** Scores read the same everywhere here (table, the picked box's line, the label chips on the image): 2 decimals. */
export const score = (v: number) => v.toFixed(2);
