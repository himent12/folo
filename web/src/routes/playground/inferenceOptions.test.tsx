// @vitest-environment happy-dom
// Unit & integration tests for Playground Inference Options (SAHI tiled inference & TTA):
// honest availability across demo / older server / live server `predict_options` / older WASM bundles,
// request cancellation & re-execution on mode/option toggles, timing metadata, and single/batch JSON exports.
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { PredictOptions, Prediction } from '$lib/api/types';
import { BrowserEngine, resetWasmProbe, type WasmModule } from '$lib/inference/browser';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { isPredictionCutAt, needsCompletionAt } from './allCandidates';
import { buildBatchJson, batchPredKey, type BatchItem } from './batch';
import { ALL, FETCH, type RunOn } from './predict';
import { session } from './session';
import type { Source } from './source';
import { createCompareController } from './useCompare';
import { resolveInferenceAvailability, useInferenceOptions } from './useInferenceOptions';
import { predictionJson, whereLabel, type Det } from './viewState';

let target: HTMLElement | undefined;
let mounted: ReturnType<typeof render> | undefined;

async function until<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  target = undefined;
  session.set(null);
  hub.features = null;
  hub.serverMock = null;
  resetWasmProbe();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('resolveInferenceAvailability', () => {
  it('disables SAHI and TTA honestly in demo mode', () => {
    const a = resolveInferenceAvailability({
      runOn: 'server',
      isMock: true,
      browserOff: false,
      serverPredictOptions: true,
      wasmTiled: true,
      wasmTta: true,
    });
    expect(a.sahi).toBe(false);
    expect(a.tta).toBe(false);
    expect(a.note).toMatch(/Demo mode/i);
  });

  it('reports honest availability for live servers with and without predict_options', () => {
    const legacy = resolveInferenceAvailability({
      runOn: 'server',
      isMock: false,
      browserOff: false,
      serverPredictOptions: false,
      wasmTiled: null,
      wasmTta: null,
    });
    expect(legacy.sahi).toBe(false);
    expect(legacy.tta).toBe(false);
    expect(legacy.note).toContain('WebGPU or CPU');

    const modern = resolveInferenceAvailability({
      runOn: 'server',
      isMock: false,
      browserOff: true,
      serverPredictOptions: true,
      wasmTiled: null,
      wasmTta: null,
    });
    expect(modern).toEqual({ sahi: true, tta: true, sahiReason: null, ttaReason: null, note: null });
  });

  it('detects older browser WASM bundles missing detectTiled or detectTta', () => {
    const oldWasm = resolveInferenceAvailability({
      runOn: 'cpu',
      isMock: false,
      browserOff: false,
      serverPredictOptions: false,
      wasmTiled: false,
      wasmTta: false,
    });
    expect(oldWasm.sahi).toBe(false);
    expect(oldWasm.tta).toBe(false);
    expect(oldWasm.note).toContain('detectTiled');
    expect(oldWasm.note).toContain('detectTta');
  });
});

describe('Playground inference metadata in labels and JSON exports', () => {
  const samplePred: Prediction = {
    model_id: 'm_1',
    image: { width: 1280, height: 720 },
    boxes: [{ class: 0, name: 'bolt', score: 0.91, x1: 10, y1: 20, x2: 50, y2: 60 }],
    timing: { preprocess_ms: 4, inference_ms: 25, postprocess_ms: 1, total_ms: 30 },
    device: 'gpu:0',
    backend: 'webgpu',
    inference: { mode: 'sahi', tile_size: 512, overlap: 0.2, full_image: true },
  };

  it('reflects SAHI/TTA in whereLabel, single JSON export, batch cache keys, and batch JSON export', () => {
    expect(whereLabel(samplePred)).toBe('browser · WebGPU · SAHI 512px');
    expect(batchPredKey('m_1', 'webgpu')).toBe('m_1|webgpu');
    expect(batchPredKey('m_1', 'webgpu', 'sahi:512:0.2:1')).toBe('m_1|webgpu|sahi:512:0.2:1');

    const single = predictionJson(samplePred, true, samplePred.boxes, { conf: 0.25, iou: 0.7, maxDet: 300, o2o: false });
    expect(single?.inference).toEqual({ mode: 'sahi', tile_size: 512, overlap: 0.2, full_image: true });

    const item: BatchItem = {
      id: 'b1',
      file: new Blob(),
      name: 'frame01.jpg',
      url: 'blob:1',
      width: 1280,
      height: 720,
      pred: samplePred,
      predKey: 'm_1|webgpu|sahi:512:0.2:1',
      status: 'done',
      error: null,
    };
    const batch = buildBatchJson({
      items: [item],
      modelId: 'm_1',
      modelName: 'fasteners-n',
      runOn: 'webgpu',
      inference: samplePred.inference,
      filter: { conf: 0.25, iou: 0.7, maxDet: 300, o2o: false },
      hidden: new Set(),
      classes: ['bolt'],
    });
    expect(batch.inference).toEqual({ mode: 'sahi', tile_size: 512, overlap: 0.2, full_image: true });
  });
});

describe('Playground Inference Options UI & live server wiring', () => {
  function makeLiveApi(features: string[]) {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 404 })));
    const api = new MockApi('normal', 0);
    Object.defineProperty(api, 'mock', { value: false });
    api.imageUrl = async () => 'data:,';
    const origSystem = api.system.bind(api);
    api.system = async () => ({ ...(await origSystem()), features });
    return api;
  }

  it('sends SAHI tile and TTA payloads to live server with predict_options, updates timing strip and JSON, and aborts in-flight calls on toggle', async () => {
    const api = makeLiveApi(['run_cli', 'predict_options']);
    const seen: PredictOptions[] = [];
    const signals: (AbortSignal | undefined)[] = [];
    const origPredict = api.predict.bind(api);
    api.predict = vi.fn(async (input, opts, sig?: AbortSignal) => {
      seen.push(opts);
      signals.push(sig);
      return origPredict(input, opts);
    });
    hub.api = api;
    hub.features = ['run_cli', 'predict_options'];
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    mounted = render(() => <Playground />, target);

    (await until(() => target!.querySelector<HTMLButtonElement>('button.sample'), 'sample button')).click();
    await until(() => target!.querySelector('#pg-panel tbody tr'), 'initial detections');
    expect(seen).toHaveLength(1);
    expect(seen[0]?.tile).toBeUndefined();
    expect(seen[0]?.tta).toBeUndefined();

    // Open collapsible Inference Options panel in the rail.
    const toggle = target.querySelector<HTMLButtonElement>('[data-inference-options] button.inf-toggle')!;
    toggle.click();
    flushSync();

    // Switch to SAHI mode: sends real `tile` payload and updates header/timing/JSON metadata.
    const modeGroup = await until(() => target!.querySelector('[role="radiogroup"][aria-label="Inference mode"]'), 'mode selector');
    const btnOf = (group: Element, label: string) =>
      [...group.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label)!;

    btnOf(modeGroup, 'SAHI').click();
    await until(() => seen.length === 2, 'SAHI prediction call');
    expect(seen[1]?.tile).toEqual({
      tile_width: 640,
      tile_height: 640,
      overlap_x: 0.2,
      overlap_y: 0.2,
      full_image: true,
      max_det: 30_000,
    });
    expect(seen[1]?.max_det).toBe(30_000);
    await until(() => target!.querySelector('section[aria-labelledby="pg-timing"]')?.textContent?.includes('SAHI 640px'), 'SAHI timing mode');

    // Change tile size to 512 -> re-runs with 512px tiles.
    const sizeGroup = target.querySelector('[role="radiogroup"][aria-label="Tile size"]')!;
    btnOf(sizeGroup, '512').click();
    await until(() => seen.length === 3, '512px SAHI prediction call');
    expect(seen[2]?.tile?.tile_width).toBe(512);

    // Switch to TTA -> sends real `tta` payload with merge_iou clamped to 0.7 and full pre-merge budget (30,000).
    btnOf(modeGroup, 'TTA').click();
    await until(() => seen.length === 4, 'TTA prediction call');
    expect(seen[3]?.tta).toEqual({
      hflip: true,
      scales: [0.83, 1.0, 1.17],
      merge_iou: 0.7,
      max_det: 30_000,
    });
    await until(() => target!.querySelector('section[aria-labelledby="pg-timing"]')?.textContent?.includes('TTA 3× + flip'), 'TTA timing mode');

    // Open JSON tab and confirm active inference mode is exported.
    const jsonTab = [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((t) => t.textContent?.includes('JSON'))!;
    jsonTab.click();
    const pre = await until(() => target!.querySelector('#pg-panel pre'), 'JSON view');
    const parsed = JSON.parse((pre.textContent ?? '').replace(/\u00a0/g, ' ')) as { inference?: unknown };
    expect(parsed.inference).toEqual({ mode: 'tta', hflip: true, scales: [0.83, 1, 1.17] });

    // Switching mode while a prediction is in flight aborts the superseded request cleanly.
    let resolvePending!: (p: Prediction) => void;
    api.predict = vi.fn((_input, _opts, sig?: AbortSignal) => {
      signals.push(sig);
      return new Promise<Prediction>((r) => {
        resolvePending = r;
      });
    });
    btnOf(modeGroup, 'SAHI').click();
    await until(() => signals.length === 5, 'in-flight SAHI request');
    const inFlightSig = signals[4]!;
    expect(inFlightSig.aborted).toBe(false);
    btnOf(modeGroup, 'Standard').click();
    flushSync();
    expect(inFlightSig.aborted).toBe(true);
    void resolvePending;
  });

  it('disables SAHI and TTA honestly on an older server lacking predict_options', async () => {
    const api = makeLiveApi(['run_cli']);
    hub.api = api;
    hub.features = ['run_cli'];
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    mounted = render(() => <Playground />, target);

    await until(() => target!.querySelector('.model-pick'), 'model picker');
    const toggle = target.querySelector<HTMLButtonElement>('[data-inference-options] button.inf-toggle')!;
    toggle.click();
    flushSync();

    const modeGroup = await until(() => target!.querySelector('[role="radiogroup"][aria-label="Inference mode"]'), 'mode selector');
    const buttons = [...modeGroup.querySelectorAll<HTMLButtonElement>('button')];
    const sahi = buttons.find((b) => b.textContent?.trim() === 'SAHI')!;
    const tta = buttons.find((b) => b.textContent?.trim() === 'TTA')!;
    expect(sahi.disabled).toBe(true);
    expect(tta.disabled).toBe(true);
    const note = target.querySelector('.inf-note')!;
    expect(note.textContent).toMatch(/predict_options|does not support SAHI/i);
    expect(modeGroup.getAttribute('aria-describedby')).toContain(note.id);
  });

  it('disables SAHI and TTA on folo serve --mock (hub.serverMock) and treats hub.features as authoritative over stale system()', async () => {
    const api = makeLiveApi(['run_cli', 'predict_options']);
    hub.api = api;
    hub.features = ['run_cli', 'predict_options'];
    hub.serverMock = true;
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    mounted = render(() => <Playground />, target);

    await until(() => target!.querySelector('.model-pick'), 'model picker');
    target.querySelector<HTMLButtonElement>('[data-inference-options] button.inf-toggle')!.click();
    flushSync();

    const modeGroup = await until(() => target!.querySelector('[role="radiogroup"][aria-label="Inference mode"]'), 'mode selector');
    const getBtn = (label: string) => [...modeGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label)!;
    // HTTP mock server (`folo serve --mock`) must not advertise synthetic predictions as SAHI/TTA.
    expect(getBtn('SAHI').disabled).toBe(true);
    expect(getBtn('TTA').disabled).toBe(true);

    // Real server whose fresh `/api/health` (`hub.features`) omits `predict_options` must override cached `/api/system`.
    hub.serverMock = false;
    hub.features = ['run_cli'];
    flushSync();
    expect(getBtn('SAHI').disabled).toBe(true);
    expect(getBtn('TTA').disabled).toBe(true);
  });

  it('preserves SAHI mode across keepSession navigation without misattributing it to Standard', async () => {
    const api = makeLiveApi(['run_cli', 'predict_options']);
    const seen: PredictOptions[] = [];
    const origPredict = api.predict.bind(api);
    api.predict = vi.fn(async (input, opts) => {
      seen.push(opts);
      return origPredict(input, opts);
    });
    hub.api = api;
    hub.features = ['run_cli', 'predict_options'];
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    mounted = render(() => <Playground />, target);

    (await until(() => target!.querySelector<HTMLButtonElement>('button.sample'), 'sample')).click();
    await until(() => seen.length === 1, 'initial standard run');

    target.querySelector<HTMLButtonElement>('[data-inference-options] button.inf-toggle')!.click();
    flushSync();
    const modeGroup = await until(() => target!.querySelector('[role="radiogroup"][aria-label="Inference mode"]'), 'mode selector');
    [...modeGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'SAHI')!.click();
    await until(() => seen.length === 2, 'SAHI run');
    // Wait until the SAHI prediction actually resolves (Timing section reflects completed pred.inference).
    await until(() => target!.querySelector('section[aria-labelledby="pg-timing"]')?.textContent?.includes('SAHI 640px'), 'SAHI result settled');

    // Leave Playground and come back with the same model: session restores SAHI mode AND reuses the SAHI result (no extra fetch).
    mounted.unmount();
    target.remove();
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    mounted = render(() => <Playground />, target);

    await until(() => target!.querySelector('.model-pick select'), 'restored model picker');
    await until(() => target!.querySelector('#pg-panel tbody tr'), 'restored SAHI detections');
    expect(target.querySelector('[data-inference-options] .inf-badge')?.textContent).toBe('SAHI 640px');
    expect(seen).toHaveLength(2);

    // Switching back to Standard actually runs Standard inference (never wedged on the restored SAHI result).
    target.querySelector<HTMLButtonElement>('[data-inference-options] button.inf-toggle')!.click();
    flushSync();
    const modeGroup2 = await until(() => target!.querySelector('[role="radiogroup"][aria-label="Inference mode"]'), 'mode selector 2');
    [...modeGroup2.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Standard')!.click();
    await until(() => seen.length === 3, 'standard re-run after restore');
    expect(seen[2]?.tile).toBeUndefined();
    expect(seen[2]?.tta).toBeUndefined();
    await until(() => session.get()?.pred !== null, 'standard result saved to session');

    // If the server later drops `predict_options` between visits, returning must NOT reuse the cached SAHI result under Standard.
    const cur = session.get();
    expect(cur?.pred).not.toBeNull();
    mounted.unmount();
    target.remove();
    // Save a SAHI session, then reconnect to an older server without `predict_options`.
    session.set({
      ...cur!,
      inference: { mode: 'sahi', tile: { tileSize: 640, overlap: 0.2, fullImage: true }, tta: { hflip: true, scales: [0.83, 1, 1.17] } },
      pred: {
        ...cur!.pred!,
        inference: { mode: 'sahi', tile_size: 640, overlap: 0.2, full_image: true },
      },
    });
    hub.features = ['run_cli'];
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    mounted = render(() => <Playground />, target);
    await until(() => seen.length === 4, 're-run as standard when server lacks predict_options');
    expect(seen[3]?.tile).toBeUndefined();
    expect(seen[3]?.tta).toBeUndefined();
  });

  it('does not replay dense SAHI/TTA results (>1,000 boxes) and still reports a 30,000-box cap honestly', async () => {
    const makeBoxes = (n: number) =>
      Array.from({ length: n }, (_, i) => ({
        class: 0,
        name: 'bolt',
        score: Math.max(0.02, 0.99 - i * 0.00001),
        x1: 1,
        y1: 1,
        x2: 10,
        y2: 10,
      }));
    const denseSahi: Prediction = {
      model_id: 'm_hard_s',
      image: { width: 640, height: 640 },
      boxes: makeBoxes(FETCH.max_det + 200),
      timing: { preprocess_ms: 2, inference_ms: 40, postprocess_ms: 2, total_ms: 44 },
      device: 'gpu:0',
      inference: { mode: 'sahi', tile_size: 640, overlap: 0.2, full_image: true },
    };
    const cappedSahi: Prediction = { ...denseSahi, boxes: makeBoxes(ALL.max_det) };

    expect(isPredictionCutAt(denseSahi, 0.01)).toBe(false);
    expect(needsCompletionAt(denseSahi, 0.01)).toBe(false);
    expect(isPredictionCutAt(cappedSahi, 0.01)).toBe(true);
    expect(needsCompletionAt(cappedSahi, 0.01)).toBe(false);

    // End-to-end: a 1,200-box SAHI result at conf=0.01 must NOT trigger a duplicate second api.predict call.
    const api = makeLiveApi(['run_cli', 'predict_options']);
    const seen: PredictOptions[] = [];
    const origPredict = api.predict.bind(api);
    api.predict = vi.fn(async (input, opts) => {
      seen.push(opts);
      if (opts.tile) return denseSahi;
      return origPredict(input, opts);
    });
    hub.api = api;
    hub.features = ['run_cli', 'predict_options'];
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s&conf=0.01');
    mounted = render(() => <Playground />, target);

    (await until(() => target!.querySelector<HTMLButtonElement>('button.sample'), 'sample')).click();
    await until(() => seen.length === 1, 'initial standard run');

    target.querySelector<HTMLButtonElement>('[data-inference-options] button.inf-toggle')!.click();
    flushSync();
    const modeGroup = await until(() => target!.querySelector('[role="radiogroup"][aria-label="Inference mode"]'), 'mode selector');
    [...modeGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'SAHI')!.click();
    await until(() => seen.length === 2, 'single SAHI run');
    await until(() => target!.querySelector('section[aria-labelledby="pg-timing"]')?.textContent?.includes('SAHI 640px'), 'SAHI settled');
    await new Promise((r) => setTimeout(r, 60));
    flushSync();
    expect(seen).toHaveLength(2);
    expect(seen[1]?.max_det).toBe(ALL.max_det);
  });

  it('does not loop Compare Model B browser weights downloads when ETag is absent under SAHI', async () => {
    const api = makeLiveApi(['run_cli', 'predict_options']);
    let downloads = 0;
    let tiledRuns = 0;
    api.modelWeightsTag = vi.fn(async () => null);
    api.modelWeights = vi.fn(async () => {
      downloads++;
      return new ArrayBuffer(16);
    });
    hub.api = api;
    hub.features = ['run_cli', 'predict_options'];
    hub.mode = 'live';

    class FakeDetector {
      info() {
        return { model_id: 'm_b', backend: 'cpu' as const, device: 'cpu', classes: ['bolt'], input: [640, 640] as [number, number] };
      }
      detect() {
        return { model_id: 'm_b', image: { width: 10, height: 10 }, boxes: [], timing: { preprocess_ms: 0, inference_ms: 1, postprocess_ms: 0, total_ms: 1 }, device: 'cpu', backend: 'cpu' } as Prediction;
      }
      detectTiled() {
        tiledRuns++;
        return {
          model_id: 'm_b',
          image: { width: 10, height: 10 },
          boxes: [{ class: 0, name: 'bolt', score: 0.9, x1: 1, y1: 1, x2: 5, y2: 5 }],
          timing: { preprocess_ms: 0, inference_ms: 2, postprocess_ms: 0, total_ms: 2 },
          device: 'cpu',
          backend: 'cpu',
        } as Prediction;
      }
      detectTta() {
        return this.detect();
      }
      free() {}
      static async load() {
        return new FakeDetector();
      }
    }

    const engine = new BrowserEngine(
      async () => ({ FoloDetector: FakeDetector }) as unknown as WasmModule,
      async () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }) as ImageData,
    );
    const runOnA = signal<RunOn>('server');
    let cmp!: ReturnType<typeof createCompareController>;
    const catalog = await api.models();
    const modelA = { ...catalog[0]!, id: 'm_a', name: 'A', classes: ['bolt'] };
    const modelB = { ...catalog[0]!, id: 'm_b', name: 'B', classes: ['bolt'] };

    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(() => {
      const inf = useInferenceOptions({
        runOn: runOnA,
        compareRunOn: () => (cmp?.compare.get() ? cmp.runOnB.get() : null),
        engine,
        browserOff: () => null,
        serverPredictOptions: () => true,
      });
      inf.setMode('sahi');
      const source = signal<Source | null>({ kind: 'file', file: new File([new Uint8Array(4)], 'a.png', { type: 'image/png' }), name: 'a.png', url: 'blob:a', width: 10, height: 10 });
      cmp = createCompareController({
        ranked: () => [modelA, modelB],
        choices: () => [],
        modelA: () => modelA,
        modelAId: () => 'm_a',
        chooseModelA: () => {},
        runOnA,
        predA: signal<Prediction | null>(null),
        predSourceKeyA: signal<string | null>(null),
        hoverA: signal<Det | null>(null),
        pickedA: signal<Det | null>(null),
        resultA: () => null,
        rowsA: () => [],
        o2oA: () => false,
        source,
        conf: signal(0.25),
        iou: signal(0.7),
        maxDet: signal(300),
        hidden: signal(new Set<number>()),
        classesA: () => ['bolt'],
        browserOff: () => null,
        localWeights: signal(new Map<string, File>()),
        dropLocalB: () => {},
        badFileB: signal<{ name: string; why: string | null } | null>(null),
        engine,
        inferenceFor: inf.effectiveFor,
      });
      cmp.runOnB.set('cpu');
      cmp.compare.set(true);
      return <div />;
    }, target);

    await until(() => cmp.predB.get()?.boxes.length === 1, 'Compare Model B SAHI prediction');
    await new Promise((r) => setTimeout(r, 60));
    flushSync();
    expect(downloads).toBe(1);
    expect(tiledRuns).toBe(1);
    expect(cmp.predB.get()?.inference?.mode).toBe('sahi');
    engine.dispose();
  });
});
