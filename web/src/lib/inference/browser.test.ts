import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Prediction } from '$lib/api/types';
import type { SystemInfo } from '$lib/api/types';
import {
  BrowserEngine,
  probeWasm,
  resetDecodeCanvas,
  resetWasmProbe,
  servedBy,
  toImageData,
  wasmFromSystem,
  type Backend,
  type Detector,
  type WasmModule,
} from './browser';

/** A fake folo-wasm: counts loads, records frees, and lets a test hold a detect call open. */
function fakeWasm() {
  const loads: string[] = [];
  const freed: string[] = [];
  let hold: Promise<void> | null = null;
  const mod: WasmModule = {
    default: async () => {},
    FoloDetector: {
      load: async (bytes, { backend, modelId }) => {
        const tag = `${modelId}:${bytes[0]}:${backend}`;
        loads.push(tag);
        let alive = true;
        const det: Detector = {
          backend,
          info: () => ({ backend, device: backend === 'webgpu' ? 'gpu:0' : 'cpu', adapter: null, fallback: null, arch: 'folo-n', imgsz: 64, classes: [] }),
          detect: async () => {
            if (hold) await hold;
            if (!alive) throw new Error('detector was freed');
            return { model_id: modelId, image: { width: 1, height: 1 }, boxes: [], timing: { preprocess_ms: 0, inference_ms: 0, postprocess_ms: 0, total_ms: 0 }, device: 'cpu', backend } as Prediction;
          },
          free: () => {
            alive = false;
            freed.push(tag);
          },
        };
        return det;
      },
    },
  };
  return { mod, loads, freed, holdUntil: (p: Promise<void> | null) => (hold = p) };
}

const pixels = async () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }) as ImageData;
const req = (modelId: string, file: File | null, backend: Backend = 'cpu', byte = 1) => ({
  modelId,
  file,
  download: async () => new Uint8Array([byte]).buffer,
  // A server that names its weights' version (an ETag), as current ones do.
  tag: '"v1"' as string | null,
  backend,
  image: async () => new Blob(),
  options: { conf: 0.01, iou: 1, maxDet: 1000 },
});

describe('BrowserEngine', () => {
  it('loads a model once per backend and reuses it', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    await e.predict(req('m1', null));
    await e.predict(req('m1', null));
    await e.predict(req('m1', null, 'webgpu'));
    expect(w.loads).toEqual(['m1:1:cpu', 'm1:1:webgpu']);
  });

  it('never answers for one model with a detector loaded for another, even from the same file', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    const f = new File([new Uint8Array([7])], 'best.folo', { lastModified: 1 });
    expect((await e.predict(req('m1', f))).model_id).toBe('m1');
    expect((await e.predict(req('m2', f))).model_id).toBe('m2');
    // Another file with the same name, size and time is another file.
    const g = new File([new Uint8Array([9])], 'best.folo', { lastModified: 1 });
    await e.predict(req('m1', g));
    expect(w.loads).toEqual(['m1:7:cpu', 'm2:7:cpu', 'm1:9:cpu']);
  });

  it('frees an evicted detector only after the requests using it are done', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    await e.predict(req('m1', null));
    let release!: () => void;
    w.holdUntil(new Promise<void>((r) => (release = r)));
    const busy = e.predict(req('m1', null));
    await new Promise((r) => setTimeout(r, 0));
    w.holdUntil(null);
    // Two more models evict m1 while its request is still detecting.
    await e.predict(req('m2', null));
    await e.predict(req('m3', null));
    expect(w.freed).toEqual([]);
    release();
    expect((await busy).model_id).toBe('m1');
    await new Promise((r) => setTimeout(r, 0));
    expect(w.freed).toEqual(['m1:1:cpu']);
  });

  it('frees a detector evicted while it was still loading, once its request is done', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    let release!: () => void;
    const slow = new Promise<void>((r) => (release = r));
    const first = e.predict({ ...req('m1', null), download: async () => (await slow, new Uint8Array([1]).buffer) });
    await e.predict(req('m2', null));
    await e.predict(req('m3', null));
    expect(w.freed).toEqual([]);
    release();
    expect((await first).model_id).toBe('m1');
    await new Promise((r) => setTimeout(r, 0));
    expect(w.freed).toEqual(['m1:1:cpu']);
  });

  it('skips requests superseded while the detector loaded: only the latest decodes and detects', async () => {
    const w = fakeWasm();
    let decoded = 0;
    const e = new BrowserEngine(async () => w.mod, async () => (decoded++, pixels()));
    let release!: () => void;
    const slow = new Promise<void>((r) => (release = r));
    const download = async () => (await slow, new Uint8Array([1]).buffer);
    // Three images picked while the weights download; each pick aborts the one before.
    const ctrls = [new AbortController(), new AbortController(), new AbortController()];
    const runs = ctrls.map((c) => e.predict({ ...req('m1', null), download, signal: c.signal }));
    ctrls[0]!.abort();
    ctrls[1]!.abort();
    release();
    await expect(runs[0]).rejects.toMatchObject({ name: 'AbortError' });
    await expect(runs[1]).rejects.toMatchObject({ name: 'AbortError' });
    expect((await runs[2]!).model_id).toBe('m1');
    expect(decoded).toBe(1);
    expect(w.loads).toEqual(['m1:1:cpu']);
    // Aborted while decoding: no detect.
    const late = new AbortController();
    const e2 = new BrowserEngine(async () => w.mod, async () => (late.abort(), pixels()));
    await expect(e2.predict({ ...req('m2', null), signal: late.signal })).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('releases a detector after a request that failed, and drops a load that failed', async () => {
    const w = fakeWasm();
    let failDecode = true;
    const e = new BrowserEngine(async () => w.mod, async (b) => {
      if (failDecode) throw new Error('not an image');
      void b;
      return pixels();
    });
    await expect(e.predict(req('m1', null))).rejects.toThrow('not an image');
    failDecode = false;
    // The failed request did not pin m1: evicting it frees it.
    await e.predict(req('m2', null));
    await e.predict(req('m3', null));
    await new Promise((r) => setTimeout(r, 0));
    expect(w.freed).toEqual(['m1:1:cpu']);
    // A failed download is not cached: the next request loads again.
    await expect(e.predict({ ...req('m4', null), download: () => Promise.reject(new Error('offline')) })).rejects.toThrow('offline');
    await e.predict(req('m4', null));
    expect(w.loads.filter((l) => l.startsWith('m4'))).toEqual(['m4:1:cpu']);
  });

  it("loads a model again when the server's weights change (a new ETag), and only then", async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    await e.predict({ ...req('m1', null), tag: '"a"' });
    await e.predict({ ...req('m1', null), tag: '"a"' });
    await e.predict({ ...req('m1', null, 'cpu', 2), tag: '"b"' });
    expect(w.loads).toEqual(['m1:1:cpu', 'm1:2:cpu']);
  });

  it('shares one download across concurrent backend loads and releases the weights buffer once loaded', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    let downloads = 0;
    let releaseDownload!: () => void;
    const holdDownload = new Promise<void>((r) => (releaseDownload = r));
    const r = (backend: 'cpu' | 'webgpu') => ({
      ...req('m1', null, backend),
      tag: '"a"',
      download: async () => {
        downloads++;
        if (downloads === 1) await holdDownload;
        return new Uint8Array([1]).buffer;
      },
    });
    const concurrent = Promise.all([e.predict(r('cpu')), e.predict(r('webgpu'))]);
    releaseDownload();
    await concurrent;
    expect(downloads).toBe(1);
    expect(w.loads).toEqual(['m1:1:cpu', 'm1:1:webgpu']);

    // Once loaded into the wasm detector, the temporary weights buffer is no longer pinned:
    // loading a model on a second backend later fetches fresh instead of retaining the old ArrayBuffer in RAM.
    const e2 = new BrowserEngine(async () => w.mod, pixels);
    let seqDownloads = 0;
    const r2 = (backend: 'cpu' | 'webgpu') => ({
      ...req('m2', null, backend),
      tag: '"b"',
      download: async () => (seqDownloads++, new Uint8Array([2]).buffer),
    });
    await e2.predict(r2('cpu'));
    await e2.predict(r2('webgpu'));
    expect(seqDownloads).toBe(2);
  });

  it('caches decoded ImageData for repeated runs on the same imageKey or Blob and invalidates on change or clear', async () => {
    const w = fakeWasm();
    let fetches = 0;
    let decodes = 0;
    const e = new BrowserEngine(
      async () => w.mod,
      async () => {
        decodes++;
        return { data: new Uint8ClampedArray([decodes, 0, 0, 255]), width: 1, height: 1 } as ImageData;
      },
    );
    const makeReq = (imageKey: string, conf = 0.25) => ({
      ...req('m1', null),
      imageKey,
      image: async () => {
        fetches++;
        return new Blob();
      },
      options: { conf, iou: 1, maxDet: 1000 },
    });
    await e.predict(makeReq('dataset:d1:img1', 0.25));
    await e.predict(makeReq('dataset:d1:img1', 0.05));
    expect(fetches).toBe(1);
    expect(decodes).toBe(1);

    // Switching to another image replaces the cache; clearing drops it too.
    await e.predict(makeReq('dataset:d1:img2', 0.25));
    expect(fetches).toBe(2);
    expect(decodes).toBe(2);
    e.clearImageCache();
    await e.predict(makeReq('dataset:d1:img2', 0.25));
    expect(fetches).toBe(3);
    expect(decodes).toBe(3);

    // Without imageKey, reusing the same Blob instance still skips decode.
    const sameBlob = new Blob([new Uint8Array([1, 2, 3])]);
    await e.predict({ ...req('m1', null), image: async () => sameBlob });
    await e.predict({ ...req('m1', null), image: async () => sameBlob });
    expect(decodes).toBe(4);
  });

  it('dispose frees cached detectors and clears the image cache', async () => {
    const w = fakeWasm();
    let decodes = 0;
    const e = new BrowserEngine(async () => w.mod, async () => (decodes++, pixels()));
    await e.predict({ ...req('m1', null), imageKey: 'img1' });
    e.dispose();
    await new Promise((r) => setTimeout(r, 0));
    expect(w.freed).toEqual(['m1:1:cpu']);
    await e.predict({ ...req('m1', null), imageKey: 'img1' });
    expect(decodes).toBe(2);
  });

  it('never reuses server weights without an ETag (they may have changed), and frees them after the request', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    let downloads = 0;
    const r = () => ({ ...req('m1', null), tag: null, download: async () => (downloads++, new Uint8Array([1]).buffer) });
    await e.predict(r());
    await e.predict(r());
    expect(downloads).toBe(2);
    expect(w.loads).toEqual(['m1:1:cpu', 'm1:1:cpu']);
    await new Promise((res) => setTimeout(res, 0));
    expect(w.freed).toEqual(['m1:1:cpu', 'm1:1:cpu']);
  });

  it('does not let a superseded model load clobber active phase or info, while same-model image switches still update phase', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    let releaseM1!: () => void;
    let releaseM2!: () => void;
    const holdM1 = new Promise<void>((r) => (releaseM1 = r));
    const holdM2 = new Promise<void>((r) => (releaseM2 = r));

    const c1 = new AbortController();
    const p1 = e.predict({
      ...req('m1', null, 'cpu', 1),
      signal: c1.signal,
      download: async () => (await holdM1, new Uint8Array([1]).buffer),
    });
    // Switch to m2 (webgpu) while m1 is still downloading; m1 is aborted.
    c1.abort();
    const p2 = e.predict({
      ...req('m2', null, 'webgpu', 2),
      download: async () => (await holdM2, new Uint8Array([2]).buffer),
    });
    await Promise.resolve();
    expect(e.phase).toBe('weights');

    // Old m1 finishes its download/load first: it must reject as AbortError and MUST NOT mark engine ready or write m1's info.
    releaseM1();
    await expect(p1).rejects.toMatchObject({ name: 'AbortError' });
    expect(e.phase).toBe('weights');
    expect(e.info).toBeNull();

    // Active m2 finishes: phase becomes ready and info reflects m2 (webgpu).
    releaseM2();
    expect((await p2).model_id).toBe('m2');
    expect(e.phase).toBe('ready');
    expect(e.info?.backend).toBe('webgpu');

    // Image switch on the same in-flight model m3: aborting first image must not block the second image from updating phase.
    let releaseM3!: () => void;
    const holdM3 = new Promise<void>((r) => (releaseM3 = r));
    const imgA = new AbortController();
    const runA = e.predict({
      ...req('m3', null, 'cpu', 3),
      signal: imgA.signal,
      download: async () => (await holdM3, new Uint8Array([3]).buffer),
    });
    imgA.abort();
    const runB = e.predict({
      ...req('m3', null, 'cpu', 3),
      download: async () => (await holdM3, new Uint8Array([3]).buffer),
    });
    releaseM3();
    await expect(runA).rejects.toMatchObject({ name: 'AbortError' });
    expect((await runB).model_id).toBe('m3');
    expect(e.phase).toBe('ready');
    expect(e.info?.backend).toBe('cpu');
  });
});

describe('is the in-browser runtime in this build (DOGFOOD-3 S5)', () => {
  const answer = (types: Record<string, string | null>) => {
    const calls: string[] = [];
    const fetcher = (async (url: string) => {
      calls.push(url);
      const type = Object.entries(types).find(([k]) => url.endsWith(k))?.[1];
      return type === null || type === undefined ? new Response(null, { status: 404 }) : new Response(null, { status: 200, headers: { 'content-type': type } });
    }) as typeof fetch;
    return { fetcher, calls };
  };

  it('is, when both files answer with their own types; asked once', async () => {
    resetWasmProbe();
    const { fetcher, calls } = answer({ '.js': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm' });
    expect(await probeWasm(fetcher)).toBe(true);
    expect(await probeWasm(fetcher)).toBe(true);
    expect(calls).toHaveLength(2);
  });

  it("is not, when the server answers with the app's index.html (the SPA fallback) or a 404", async () => {
    resetWasmProbe();
    expect(await probeWasm(answer({ '.js': 'text/html; charset=utf-8', '.wasm': 'text/html; charset=utf-8' }).fetcher)).toBe(false);
    resetWasmProbe();
    expect(await probeWasm(answer({ '.js': 'text/javascript', '.wasm': null }).fetcher)).toBe(false);
    resetWasmProbe();
  });

  it('does not know while offline or while the server errs, and asks again next time instead of keeping that', async () => {
    resetWasmProbe();
    const offline = (async () => Promise.reject(new TypeError('Failed to fetch'))) as typeof fetch;
    expect(await probeWasm(offline)).toBeNull();
    // Back online: the next visit asks again and gets the answer, which is then kept.
    const { fetcher, calls } = answer({ '.js': 'text/javascript', '.wasm': 'application/wasm' });
    expect(await probeWasm(fetcher)).toBe(true);
    expect(await probeWasm(offline)).toBe(true);
    expect(calls).toHaveLength(2);
    resetWasmProbe();
    const down = (async () => new Response(null, { status: 503 })) as typeof fetch;
    expect(await probeWasm(down)).toBeNull();
    expect(await probeWasm(answer({ '.js': 'text/javascript', '.wasm': null }).fetcher)).toBe(false);
    resetWasmProbe();
  });

  it("takes the word of the server that served this page; another server's, an older one's or a dev server's is left to the probe", () => {
    expect(wasmFromSystem({ features: ['web_wasm'] } as SystemInfo, true)).toBe(true);
    expect(wasmFromSystem(undefined, true)).toBeNull();
    // Review of round 25: Studio without its own runtime files, connected to another server that has them, offered the
    // browser path and failed when it loaded. Another server's `web_wasm` says nothing of this page.
    expect(wasmFromSystem({ features: ['run_cli', 'web_wasm'] } as SystemInfo, false)).toBeNull();
    expect(wasmFromSystem({ features: ['web_wasm'] } as SystemInfo)).toBeNull();
    // A missing web_wasm means no runtime only from a server with the final list (`run_cli`) that served this page.
    const final = { features: ['queue_control', 'run_cli'] } as SystemInfo;
    expect(wasmFromSystem(final, true)).toBe(false);
    expect(wasmFromSystem(final, false)).toBeNull();
    expect(wasmFromSystem({ features: ['queue_control'] } as SystemInfo, true)).toBeNull();
  });

  it('knows when the server serves this page', () => {
    expect(servedBy('', 'http://127.0.0.1:7878', false)).toBe(true);
    expect(servedBy('http://127.0.0.1:7878', 'http://127.0.0.1:7878', false)).toBe(true);
    expect(servedBy('http://gpu-box:7878', 'http://127.0.0.1:7878', false)).toBe(false);
    expect(servedBy('', 'http://localhost:5173', true)).toBe(false);
  });
});

describe('toImageData canvas reuse and fallback', () => {
  afterEach(() => {
    resetDecodeCanvas();
    vi.unstubAllGlobals();
  });

  it('reuses one OffscreenCanvas across decodes, clears on same size, and resizes on new dimensions', async () => {
    let created = 0;
    const clearCalls: number[][] = [];
    const drawCalls: number[][] = [];
    const closed: number[] = [];

    class FakeOffscreenCanvas {
      width: number;
      height: number;
      constructor(w: number, h: number) {
        created++;
        this.width = w;
        this.height = h;
      }
      getContext() {
        return {
          clearRect: (x: number, y: number, w: number, h: number) => clearCalls.push([x, y, w, h]),
          drawImage: () => drawCalls.push([this.width, this.height]),
          getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }) as ImageData,
        };
      }
    }

    const sizes = [
      [10, 20],
      [10, 20],
      [30, 40],
    ];
    let idx = 0;
    vi.stubGlobal('OffscreenCanvas', FakeOffscreenCanvas);
    vi.stubGlobal('createImageBitmap', async () => {
      const i = idx++;
      const [width, height] = sizes[i]!;
      return { width, height, close: () => closed.push(i) };
    });

    const a = await toImageData(new Blob());
    const b = await toImageData(new Blob());
    const c = await toImageData(new Blob());

    expect(created).toBe(1);
    expect(clearCalls).toEqual([[0, 0, 10, 20]]);
    expect(drawCalls).toEqual([
      [10, 20],
      [10, 20],
      [30, 40],
    ]);
    expect([a.width, b.width, c.width]).toEqual([10, 10, 30]);
    expect(closed).toEqual([0, 1, 2]);
  });

  it('falls back gracefully to HTMLCanvasElement when OffscreenCanvas is unavailable', async () => {
    let createdEl = 0;
    vi.stubGlobal('OffscreenCanvas', undefined);
    vi.stubGlobal('document', {
      createElement: (tag: string) => {
        expect(tag).toBe('canvas');
        createdEl++;
        return {
          width: 0,
          height: 0,
          getContext: () => ({
            clearRect: () => {},
            drawImage: () => {},
            getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }) as ImageData,
          }),
        };
      },
    });
    vi.stubGlobal('createImageBitmap', async () => ({ width: 8, height: 6, close: () => {} }));

    const first = await toImageData(new Blob());
    const second = await toImageData(new Blob());
    expect(createdEl).toBe(1);
    expect(first.width).toBe(8);
    expect(second.height).toBe(6);

    // Clearing the image cache or disposing the engine resets the backing canvas (width/height -> 0).
    const engine = new BrowserEngine(async () => fakeWasm().mod, toImageData);
    engine.clearImageCache();
    await toImageData(new Blob());
    expect(createdEl).toBe(2);
    engine.dispose();
    await toImageData(new Blob());
    expect(createdEl).toBe(3);
  });
});
