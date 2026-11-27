// @vitest-environment happy-dom
// Unit & integration tests for Playground Inference Options (SAHI tiled inference & TTA):
// honest availability across demo / older server / live server `predict_options` / older WASM bundles,
// request cancellation & re-execution on mode/option toggles, timing metadata, and single/batch JSON exports.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { PredictOptions, Prediction } from '$lib/api/types';
import { resetWasmProbe } from '$lib/inference/browser';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { buildBatchJson, batchPredKey, type BatchItem } from './batch';
import { session } from './session';
import { resolveInferenceAvailability } from './useInferenceOptions';
import { predictionJson, whereLabel } from './viewState';

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
      max_det: 1000,
    });
    await until(() => target!.querySelector('section[aria-labelledby="pg-timing"]')?.textContent?.includes('SAHI 640px'), 'SAHI timing mode');

    // Change tile size to 512 -> re-runs with 512px tiles.
    const sizeGroup = target.querySelector('[role="radiogroup"][aria-label="Tile size"]')!;
    btnOf(sizeGroup, '512').click();
    await until(() => seen.length === 3, '512px SAHI prediction call');
    expect(seen[2]?.tile?.tile_width).toBe(512);

    // Switch to TTA -> sends real `tta` payload with merge_iou clamped to 0.7 instead of loose FETCH iou=1.
    btnOf(modeGroup, 'TTA').click();
    await until(() => seen.length === 4, 'TTA prediction call');
    expect(seen[3]?.tta).toEqual({
      hflip: true,
      scales: [0.83, 1.0, 1.17],
      merge_iou: 0.7,
      max_det: 1000,
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
    expect(target.querySelector('.inf-note')?.textContent).toMatch(/predict_options|does not support SAHI/i);
  });
});
