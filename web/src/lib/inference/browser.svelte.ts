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

export interface Detector {
  readonly backend: Backend;
  info(): DetectorInfo;
  detect(image: ImageData, width?: number, height?: number, options?: { conf?: number; iou?: number; maxDet?: number }): Promise<Prediction>;
  free(): void;
}

export interface WasmModule {
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

  #detectors = new Map<string, Entry>();
  #runtimeReady = false;

  /** `load` provides the folo-wasm module and `decode` an image's pixels (tests pass fakes). */
  constructor(
    readonly load: () => Promise<WasmModule> = loadWasm,
    readonly decode: (image: Blob) => Promise<ImageData> = toImageData,
  ) {}

  /**
   * Runs `modelId` on an image in the browser. `file` is a .folo the user picked for this model; without one the
   * weights come from `download` (the server). Either is read only when that model and source are not loaded on
   * this backend yet.
   */
  async predict(req: {
    modelId: string;
    file: File | null;
    download: () => Promise<ArrayBuffer>;
    /** The server's version of the weights (an ETag): a new one loads them again. */
    tag?: string | null;
    backend: Backend;
    image: () => Promise<Blob>;
    options: { conf: number; iou: number; maxDet: number };
  }): Promise<Prediction> {
    const entry = this.#detector(req);
    entry.users++;
    try {
      const det = await entry.detector;
      this.phase = 'ready';
      const pixels = await this.decode(await req.image());
      const p = await det.detect(pixels, undefined, undefined, req.options);
      this.info = det.info();
      return p;
    } finally {
      entry.users--;
      this.#release(entry);
    }
  }

  #detector(req: { modelId: string; file: File | null; download: () => Promise<ArrayBuffer>; tag?: string | null; backend: Backend }): Entry {
    // One detector per model, weights source (and version) and backend: the same file picked for two models loads
    // twice, since a detector stamps its model id on every result; retrained weights (a run's last.folo changes
    // every epoch) load again.
    const key = `${req.modelId}|${req.file ? `file#${fileId(req.file)}` : `server:${req.tag ?? ''}`}|${req.backend}`;
    const hit = this.#detectors.get(key);
    if (hit) {
      // Most recently used last.
      this.#detectors.delete(key);
      this.#detectors.set(key, hit);
      return hit;
    }
    const detector = (async () => {
      if (!this.#runtimeReady) this.phase = 'runtime';
      const m = await this.load();
      this.#runtimeReady = true;
      this.phase = 'weights';
      const bytes = new Uint8Array(await (req.file ? req.file.arrayBuffer() : req.download()));
      this.phase = 'compile';
      return m.FoloDetector.load(bytes, { backend: req.backend, modelId: req.modelId });
    })();
    const entry: Entry = { detector, users: 0, evicted: false };
    this.#detectors.set(key, entry);
    detector.catch(() => {
      if (this.#detectors.get(key) === entry) this.#detectors.delete(key);
    });
    while (this.#detectors.size > KEEP) {
      const [old, e] = this.#detectors.entries().next().value!;
      this.#detectors.delete(old);
      e.evicted = true;
      this.#release(e);
    }
    return entry;
  }

  /** Frees an evicted detector once no request is using it. */
  #release(e: Entry) {
    if (e.evicted && e.users === 0) e.detector.then((d) => d.free(), () => {});
  }
}

interface Entry {
  detector: Promise<Detector>;
  /** Requests using it right now: an evicted detector is freed only when they are done. */
  users: number;
  evicted: boolean;
}

/** A stable identity per picked file (two files can share name, size and time). */
const fileIds = new WeakMap<Blob, number>();
let nextFileId = 0;
function fileId(f: Blob): number {
  let id = fileIds.get(f);
  if (id === undefined) fileIds.set(f, (id = ++nextFileId));
  return id;
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
