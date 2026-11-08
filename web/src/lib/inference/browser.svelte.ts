// Running FOLO models in this browser: folo-wasm (crates/folo-wasm, docs/WASM.md) on WebGPU or the wasm CPU.
// scripts/build-wasm.(sh|ps1) writes it to web/public/folo-wasm/, which Vite serves as /folo-wasm/. The module is
// fetched once; each model is loaded once per backend (loading parses and fuses it and, on WebGPU, compiles the
// kernels, which takes seconds) and reused for every image. Results have /api/predict's shape plus `backend`.
import { ApiError } from '$lib/api/client';
import type { Prediction } from '$lib/api/types';

export type Backend = 'webgpu' | 'cpu';

/** `detector.info()` (docs/WASM.md). */
export interface DetectorInfo {
  backend: Backend;
  device: string;
  /** The GPU adapter's name, or null when the browser does not disclose it (Chrome does not). */
  adapter: string | null;
  /** Why `auto` chose the CPU (null otherwise). */
  fallback: string | null;
  arch: string;
  imgsz: number;
  classes: string[];
}

interface Detector {
  readonly backend: Backend;
  info(): DetectorInfo;
  detect(image: ImageData, width?: number, height?: number, options?: { conf?: number; iou?: number; maxDet?: number }): Promise<Prediction>;
  free(): void;
}

interface WasmModule {
  default(): Promise<unknown>;
  FoloDetector: { load(bytes: Uint8Array, options: { backend: Backend; modelId: string }): Promise<Detector> };
}

let wasm: Promise<WasmModule> | null = null;

/** The folo-wasm module, fetched and instantiated once (a failure is not cached, so a retry tries again). */
function loadWasm(): Promise<WasmModule> {
  if (!wasm) {
    const p = (async () => {
      let m: WasmModule;
      try {
        m = (await import(/* @vite-ignore */ `${import.meta.env.BASE_URL}folo-wasm/folo_wasm.js`)) as WasmModule;
      } catch {
        throw new ApiError(
          0,
          'wasm_not_built',
          'This build of Studio does not include the in-browser runtime.',
          'Run scripts/build-wasm.sh (or scripts\\build-wasm.ps1) before building or serving Studio.',
        );
      }
      await m.default();
      return m;
    })();
    wasm = p;
    p.catch(() => {
      if (wasm === p) wasm = null;
    });
  }
  return wasm;
}

/** Whether the browser exposes WebGPU at all (a usable device is only known once a model loads on it). */
export function hasWebGpu(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export type Phase = 'idle' | 'runtime' | 'weights' | 'compile' | 'ready';

/** What a phase means, for the waiting state. */
export const PHASE_TEXT: Record<Phase, string> = {
  idle: '',
  runtime: 'Starting the in-browser runtime…',
  weights: 'Downloading the model…',
  compile: 'Loading the model into the browser…',
  ready: 'Running in the browser…',
};

/** Detectors kept loaded at once (an older one is freed: models hold wasm memory and GPU buffers). */
const KEEP = 2;

export class BrowserEngine {
  /** What the engine is doing for the latest request. */
  phase = $state<Phase>('idle');
  /** The detector that ran the latest prediction. */
  info = $state.raw<DetectorInfo | null>(null);

  #detectors = new Map<string, Promise<Detector>>();
  #runtimeReady = false;

  /**
   * Runs `modelId` on an image in the browser. `weightsKey` identifies the weights (a model id, or a file the user
   * picked); `weights` fetches them, only when that key is not loaded on this backend yet.
   */
  async predict(req: {
    modelId: string;
    weightsKey: string;
    weights: () => Promise<ArrayBuffer>;
    backend: Backend;
    image: () => Promise<Blob>;
    options: { conf: number; iou: number; maxDet: number };
  }): Promise<Prediction> {
    const det = await this.#detector(req);
    this.phase = 'ready';
    const pixels = await toImageData(await req.image());
    const p = await det.detect(pixels, undefined, undefined, req.options);
    this.info = det.info();
    return p;
  }

  #detector(req: { modelId: string; weightsKey: string; weights: () => Promise<ArrayBuffer>; backend: Backend }): Promise<Detector> {
    const key = `${req.weightsKey}|${req.backend}`;
    const hit = this.#detectors.get(key);
    if (hit) {
      // Most recently used last.
      this.#detectors.delete(key);
      this.#detectors.set(key, hit);
      return hit;
    }
    const p = (async () => {
      if (!this.#runtimeReady) this.phase = 'runtime';
      const m = await loadWasm();
      this.#runtimeReady = true;
      this.phase = 'weights';
      const bytes = new Uint8Array(await req.weights());
      this.phase = 'compile';
      return m.FoloDetector.load(bytes, { backend: req.backend, modelId: req.modelId });
    })();
    this.#detectors.set(key, p);
    p.catch(() => {
      if (this.#detectors.get(key) === p) this.#detectors.delete(key);
    });
    while (this.#detectors.size > KEEP) {
      const [old, d] = this.#detectors.entries().next().value!;
      this.#detectors.delete(old);
      d.then((x) => x.free(), () => {});
    }
    return p;
  }
}

/** Decodes an image to RGBA pixels (EXIF orientation applied, as browsers display it). */
async function toImageData(blob: Blob): Promise<ImageData> {
  const bmp = await createImageBitmap(blob);
  try {
    const canvas =
      typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(bmp.width, bmp.height) : Object.assign(document.createElement('canvas'), { width: bmp.width, height: bmp.height });
    const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error('This browser cannot read image pixels (no 2D canvas).');
    ctx.drawImage(bmp, 0, 0);
    return ctx.getImageData(0, 0, bmp.width, bmp.height);
  } finally {
    bmp.close();
  }
}
