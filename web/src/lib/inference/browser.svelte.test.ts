import { describe, expect, it } from 'vitest';
import type { Prediction } from '$lib/api/types';
import { BrowserEngine, type Backend, type Detector, type WasmModule } from './browser.svelte';

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

  it('downloads the weights once for both backends when the ETag is the same', async () => {
    const w = fakeWasm();
    const e = new BrowserEngine(async () => w.mod, pixels);
    let downloads = 0;
    const r = (backend: 'cpu' | 'webgpu') => ({ ...req('m1', null, backend), tag: '"a"', download: async () => (downloads++, new Uint8Array([1]).buffer) });
    await e.predict(r('cpu'));
    await e.predict(r('webgpu'));
    expect(downloads).toBe(1);
    expect(w.loads).toEqual(['m1:1:cpu', 'm1:1:webgpu']);
  });
});
