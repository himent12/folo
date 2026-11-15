// Running FOLO models in this browser: folo-wasm (crates/folo-wasm, docs/WASM.md) on WebGPU or the wasm CPU.
// scripts/build-wasm.(sh|ps1) writes it to web/public/folo-wasm/, which Vite serves as /folo-wasm/. The module is
// fetched once; each model is loaded once per backend (loading parses and fuses it and, on WebGPU, compiles the
// kernels, which takes seconds) and reused for every image. Results have /api/predict's shape plus `backend`.
import { ApiError } from '$lib/api/client';
import { hasFeature } from '$lib/api/queue';
import type { Prediction, SystemInfo } from '$lib/api/types';
import { signal } from 'fw';

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
          NO_WASM,
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

/**
 * Whether `/api/system` says this Studio build includes folo-wasm (the one place that reads `web_wasm`). The server
 * speaks only for a page it serves (`servedHere`: same origin, not Vite's dev server, whose files the server cannot
 * see): there, `true` where it lists it, and `false` where a server that has the final list (it lists `run_cli`) leaves
 * it out. Anywhere else (another origin, the dev server, an older server that does not say) it is `null`, and then
 * `probeWasm` asks this page's own files: another server's runtime is not this page's.
 */
export function wasmFromSystem(system: SystemInfo | null | undefined, servedHere = false): boolean | null {
  if (!servedHere) return null;
  if (hasFeature(system, 'web_wasm')) return true;
  return hasFeature(system, 'run_cli') ? false : null;
}

/** Whether the server at `serverUrl` serves this page: the same origin, in a built Studio (not Vite's dev server). */
export function servedBy(serverUrl: string, origin = typeof location === 'undefined' ? '' : location.origin, dev = import.meta.env.DEV): boolean {
  if (dev || !origin) return false;
  try {
    return new URL(serverUrl || origin, origin).origin === origin;
  } catch {
    return false;
  }
}

/** The runtime's files (folo-server `FOLO_WASM_FILES`): the module and its JS glue. */
const WASM_FILES = [
  ['folo-wasm/folo_wasm.js', /javascript/],
  ['folo-wasm/folo_wasm_bg.wasm', /wasm/],
] as const;

let probe: Promise<boolean | null> | null = null;

/**
 * Whether this build serves the in-browser runtime: `true` when both files answer with their own type, `false` when
 * one is confirmed missing (a 404 or 410, or another type: a build without them gets the SPA's index.html), `null`
 * when that is not known (offline, a server error). Only a confirmed answer is kept for the session; an unknown one
 * is asked again on the next call (the next visit to Playground, or on reconnecting). `fetcher` is for tests.
 */
export function probeWasm(fetcher: typeof fetch = fetch): Promise<boolean | null> {
  if (probe) return probe;
  const p = Promise.all(
    WASM_FILES.map(([path, type]): Promise<boolean | null> =>
      fetcher(`${import.meta.env.BASE_URL}${path}`, { method: 'HEAD' }).then(
        (r) => (r.ok ? type.test(r.headers.get('content-type') ?? '') : r.status === 404 || r.status === 410 ? false : null),
        () => null,
      ),
    ),
  ).then((ok) => (ok.includes(false) ? false : ok.includes(null) ? null : true));
  probe = p;
  void p.then((v) => {
    if (v === null && probe === p) probe = null;
  });
  return p;
}

/** Forgets the probe's answer (tests). */
export function resetWasmProbe() {
  probe = null;
}

/** Said where the runtime is missing: the browser choices are off, with this as their reason. */
export const NO_WASM = 'This build of Studio does not include the in-browser runtime.';

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
  #phase = signal<Phase>('idle');
  #info = signal<DetectorInfo | null>(null);
  /** What the engine is doing for the latest request. */
  get phase(): Phase {
    return this.#phase.get();
  }
  set phase(v: Phase) {
    this.#phase.set(v);
  }
  /** The detector that ran the latest prediction. */
  get info(): DetectorInfo | null {
    return this.#info.get();
  }
  set info(v: DetectorInfo | null) {
    this.#info.set(v);
  }

  #detectors = new Map<string, Entry>();
  /** The last weights downloaded (by model and ETag): loading the same model on the other backend reuses them. */
  #lastDownload: { key: string; bytes: Promise<ArrayBuffer> } | null = null;
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
    /** Aborted when a newer request replaces this one: it stops before decoding and before detecting. */
    signal?: AbortSignal;
  }): Promise<Prediction> {
    const entry = this.#detector(req);
    entry.users++;
    const superseded = () => {
      if (req.signal?.aborted) throw new DOMException('Superseded', 'AbortError');
    };
    try {
      const det = await entry.detector;
      this.phase = 'ready';
      // Requests replaced while the detector loaded all wake here: only the latest decodes and runs.
      superseded();
      const pixels = await this.decode(await req.image());
      superseded();
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
    // Server weights without an ETag cannot be known to be unchanged: they load for this request only (not cached).
    const cacheable = !!req.file || !!req.tag;
    const key = `${req.modelId}|${req.file ? `file#${fileId(req.file)}` : `server:${req.tag ?? ''}`}|${req.backend}`;
    const hit = cacheable ? this.#detectors.get(key) : undefined;
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
      const bytes = new Uint8Array(await (req.file ? req.file.arrayBuffer() : this.#download(req)));
      this.phase = 'compile';
      return m.FoloDetector.load(bytes, { backend: req.backend, modelId: req.modelId });
    })();
    // Uncached: born evicted, so it is freed as soon as its request is done.
    const entry: Entry = { detector, users: 0, evicted: !cacheable };
    if (!cacheable) return entry;
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

  #download(req: { modelId: string; download: () => Promise<ArrayBuffer>; tag?: string | null }): Promise<ArrayBuffer> {
    const key = `${req.modelId}|${req.tag ?? ''}`;
    // Without an ETag the bytes may have changed: fetch again.
    if (req.tag && this.#lastDownload?.key === key) return this.#lastDownload.bytes;
    const bytes = req.download();
    this.#lastDownload = { key, bytes };
    bytes.catch(() => {
      if (this.#lastDownload?.bytes === bytes) this.#lastDownload = null;
    });
    return bytes;
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
