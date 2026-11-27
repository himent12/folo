// @vitest-environment happy-dom
// Unit & integration tests for Playground per-class confidence thresholds (JSON import/export, client-side
// filtering, overlays/counts, batch QA, compare summary, browser WASM wiring) and browser INT8 precision UI.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Prediction } from '$lib/api/types';
import { BrowserEngine, resetWasmProbe, type DetectorInfo, type WasmDetectOptions, type WasmModule } from '$lib/inference/browser';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { batchQaCounts, buildBatchCsvRows, buildBatchJson, filterBatchBoxes, type BatchItem } from './batch';
import { comparePredictionJson, summarizeComparison } from './compare';
import {
  exportThresholdsJson,
  mapPerClassConfToClasses,
  parseThresholdsJson,
} from './perClassConf';
import { ask, FETCH } from './predict';
import PlaygroundTiming from './PlaygroundTiming';
import { session } from './session';
import { areDefaultThresholds } from './viewState';

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

describe('perClassConf JSON import/export & class remapping', () => {
  const classes = ['person', 'bicycle', 'car'];

  it('imports #802 ThresholdCalibration JSON and wrapped folo val JSON, and round-trips export', () => {
    const cal = JSON.stringify({
      global: { best_conf: 0.4, best_f1: 0.78, precision: 0.82, recall: 0.74 },
      per_class: [
        { class: 0, best_conf: 0.25, best_f1: 0.85, precision: 0.86, recall: 0.84, truths: 20, preds: 19 },
        { class: 2, best_conf: 0.65, best_f1: 0.91, precision: 0.92, recall: 0.9, truths: 15, preds: 15 },
      ],
      macro_f1: 0.88,
    });
    const parsed = parseThresholdsJson(cal, classes);
    expect(parsed).toEqual({
      ok: true,
      globalConf: 0.4,
      perClassConf: { 0: 0.25, 2: 0.65 },
      confInclusive: true,
      appliedCount: 2,
      warnings: [],
    });

    const wrapped = parseThresholdsJson(JSON.stringify({ map50: 0.82, thresholds: JSON.parse(cal) }), classes);
    expect(wrapped.ok && wrapped.perClassConf).toEqual({ 0: 0.25, 2: 0.65 });

    const exported = exportThresholdsJson(0.4, { 0: 0.25, 2: 0.65 }, classes);
    expect(exported).toEqual({
      schema: 'folo.thresholds.v1',
      global: { best_conf: 0.4 },
      per_class: [
        { class: 0, name: 'person', best_conf: 0.25 },
        { class: 2, name: 'car', best_conf: 0.65 },
      ],
    });
    const reimported = parseThresholdsJson(JSON.stringify(exported), classes);
    expect(reimported.ok && reimported.globalConf).toBe(0.4);
    expect(reimported.ok && reimported.perClassConf).toEqual({ 0: 0.25, 2: 0.65 });

    // Full [0, 1] global confidence values are preserved without clamping, and folo val --json metrics-only records warn cleanly.
    const extreme = parseThresholdsJson(
      JSON.stringify({ global: { best_conf: 0.005 }, per_class: [{ class: 0, best_conf: 0.985 }] }),
      classes,
    );
    expect(extreme.ok && extreme.globalConf).toBe(0.005);
    expect(extreme.ok && extreme.perClassConf).toEqual({ 0: 0.985 });

    const valMetricsOnly = parseThresholdsJson(
      JSON.stringify({
        best_f1_conf: 0.37,
        per_class: [{ class: 0, name: 'person', ap50: 0.88, ap50_95: 0.61 }],
      }),
      classes,
    );
    expect(valMetricsOnly.ok).toBe(true);
    if (valMetricsOnly.ok) {
      expect(valMetricsOnly.globalConf).toBe(0.37);
      expect(valMetricsOnly.perClassConf).toEqual({});
      expect(valMetricsOnly.warnings[0]).toContain('best_f1_conf');
    }
  });

  it('imports simple numeric ID and class-name maps, and rejects invalid inputs cleanly', () => {
    const byId = parseThresholdsJson('{"0": 0.4, "2": 0.72}', classes);
    expect(byId).toEqual({
      ok: true,
      globalConf: null,
      perClassConf: { 0: 0.4, 2: 0.72 },
      confInclusive: true,
      appliedCount: 2,
      warnings: [],
    });

    const byName = parseThresholdsJson('{"person": 0.35, "Car": 0.8}', classes);
    expect(byName).toEqual({
      ok: true,
      globalConf: null,
      perClassConf: { 0: 0.35, 2: 0.8 },
      confInclusive: true,
      appliedCount: 2,
      warnings: [],
    });

    expect(parseThresholdsJson('{"0": 1.2}', classes).ok).toBe(false);
    expect(parseThresholdsJson('{"-1": 0.5}', classes).ok).toBe(false);
    expect(parseThresholdsJson('{"unknown_cls": 0.5}', classes).ok).toBe(false);
    expect(parseThresholdsJson('{"0": 0.4, "person": 0.6}', classes).ok).toBe(false);
    expect(parseThresholdsJson('{"per_class":[{"class":0,"best_conf":0.4},{"class":0,"best_conf":0.5}]}', classes).ok).toBe(false);
  });

  it('remaps per-class thresholds across models with different class orders', () => {
    expect(mapPerClassConfToClasses({ 0: 0.4, 2: 0.8 }, ['circle', 'square', 'triangle'], ['triangle', 'circle', 'star'])).toEqual({
      0: 0.8,
      1: 0.4,
    });
    expect(areDefaultThresholds(0.25, 0.7, 300, false, { 0: 0.4 })).toBe(false);
    expect(areDefaultThresholds(0.25, 0.7, 300, false, {})).toBe(true);
  });
});

describe('Batch QA & Compare summary with per-class thresholds', () => {
  const pred: Prediction = {
    model_id: 'm1',
    image: { width: 640, height: 480 },
    boxes: [
      { class: 0, name: 'circle', score: 0.55, x1: 10, y1: 10, x2: 50, y2: 50 },
      { class: 1, name: 'square', score: 0.35, x1: 80, y1: 80, x2: 120, y2: 120 },
    ],
    timing: { preprocess_ms: 1, inference_ms: 4, postprocess_ms: 0.5, total_ms: 5.5 },
    device: 'gpu:0',
  };

  it('reflects per-class thresholds in batch QA counts, CSV rows, batch JSON, and compare JSON', () => {
    const item: BatchItem = {
      id: 'b1',
      file: new Blob(),
      name: 'frame1.jpg',
      url: 'blob:1',
      width: 640,
      height: 480,
      pred,
      predKey: 'm1|server',
      status: 'done',
      error: null,
    };
    const filter = { conf: 0.25, iou: 0.7, maxDet: 300, o2o: false, perClassConf: { 0: 0.7, 1: 0.3 } };
    expect(filterBatchBoxes(pred, filter, new Set()).map((b) => b.name)).toEqual(['square']);
    const qa = batchQaCounts([item], filter, new Set(), ['circle', 'square']);
    expect(qa.withDetections).toBe(1);
    expect(qa.byClass.map((c) => [c.name, c.boxCount])).toEqual([['square', 1]]);
    expect(buildBatchCsvRows({ items: [item], filter, hidden: new Set() })).toHaveLength(1);
    const json = buildBatchJson({
      items: [item],
      modelId: 'm1',
      modelName: 'shapes-n',
      runOn: 'server',
      filter,
      hidden: new Set(),
      classes: ['circle', 'square'],
    });
    expect(json.filter).toMatchObject({ conf: 0.25, per_class_conf: { '0': 0.7, '1': 0.3 } });
    expect(json.images[0]!.boxes.map((b) => b.name)).toEqual(['square']);

    const boxesA = filterBatchBoxes(pred, filter, new Set());
    const cmpSummary = summarizeComparison({ boxesA, boxesB: pred.boxes, predA: pred, predB: pred });
    const cmpJson = comparePredictionJson({
      view: 'side-by-side',
      predA: pred,
      predB: pred,
      hasResultA: true,
      hasResultB: true,
      boxesA,
      boxesB: pred.boxes,
      modelAName: 'A',
      modelBName: 'B',
      runOnA: 'server',
      runOnB: 'server',
      o2oA: false,
      o2oB: false,
      filter: { conf: 0.25, iou: 0.7, maxDet: 300, perClassConf: { 0: 0.7, 1: 0.3 } },
      summary: cmpSummary,
    });
    expect(cmpJson?.filter).toMatchObject({ per_class_conf: { '0': 0.7, '1': 0.3 } });
    expect(cmpJson?.comparison.class_diff?.only_b).toEqual(['circle']);
  });

  it('forwards perClassConf and confInclusive from Playground ask() straight into BrowserEngine / folo-wasm', async () => {
    const api = new MockApi('normal', 0);
    Object.defineProperty(api, 'mock', { value: false });
    api.modelWeightsTag = async () => '"etag1"';
    api.modelWeights = async () => new ArrayBuffer(16);
    const seen: (WasmDetectOptions | undefined)[] = [];
    const mod: WasmModule = {
      default: async () => {},
      FoloDetector: {
        load: async (_bytes, { backend, modelId }) => ({
          backend,
          info: () => ({ backend, device: 'cpu', adapter: null, fallback: null, precision: 'int8', precisionFallback: null, arch: 'folo2-n', imgsz: 640, classes: ['circle', 'square'], modelId }),
          detect: async (_img, _w, _h, opts) => {
            seen.push(opts);
            return { ...pred, model_id: modelId, backend };
          },
          free: () => {},
        }),
      },
    };
    const engine = new BrowserEngine(
      async () => mod,
      async () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }) as ImageData,
    );
    await ask({
      api,
      engine,
      model: 'm1',
      source: { kind: 'file', file: new Blob([new Uint8Array(4)]), name: 'a.png', url: 'blob:a', width: 1, height: 1 },
      on: 'cpu',
      file: null,
      options: FETCH,
      perClassConf: { 0: 0.6 },
      confInclusive: true,
      precision: 'int8',
      signal: new AbortController().signal,
    });
    expect(seen).toEqual([{ conf: 0.01, iou: 1, maxDet: 1000, perClassConf: { 0: 0.6 }, confInclusive: true }]);
    expect(engine.info?.precision).toBe('int8');
    engine.dispose();
  });
});

describe('Playground UI: per-class thresholds & browser INT8 precision', () => {
  it('filters detections live with zero extra server requests on manual override, JSON import, and reset', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    let predictCalls = 0;
    api.predict = async (_input, opts) => {
      predictCalls++;
      return {
        model_id: opts.model_id,
        image: { width: 640, height: 480 },
        boxes: [
          { class: 0, name: 'circle', score: 0.85, x1: 10, y1: 10, x2: 80, y2: 80 },
          { class: 0, name: 'circle', score: 0.4, x1: 100, y1: 10, x2: 160, y2: 80 },
          { class: 1, name: 'square', score: 0.5, x1: 200, y1: 100, x2: 280, y2: 180 },
        ],
        timing: { preprocess_ms: 1, inference_ms: 3, postprocess_ms: 0.5, total_ms: 4.5 },
        device: 'gpu:0',
      };
    };
    hub.api = api;
    hub.mode = 'mock';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    mounted = render(() => <Playground />, target);

    (await until(() => target!.querySelector<HTMLButtonElement>('button.sample'), 'sample button')).click();
    await until(() => target!.querySelectorAll('#pg-panel tbody tr').length === 3, '3 initial detections');
    expect(predictCalls).toBe(1);

    // Expand per-class thresholds and raise circle (class 0) to 0.70 -> drops 0.40 circle with 0 extra requests
    target.querySelector<HTMLButtonElement>('[data-per-class-thresholds] button.pc-toggle')!.click();
    flushSync();
    const circleInput = target.querySelector<HTMLInputElement>('[data-class-threshold-row="0"] input[type="number"]')!;
    circleInput.value = '0.7';
    circleInput.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelectorAll('#pg-panel tbody tr')).toHaveLength(2);
    expect(target.querySelector('[data-per-class-badge]')?.textContent).toBe('1 override');
    expect(predictCalls).toBe(1);

    // Import #802 ThresholdCalibration JSON: global 0.30, circle 0.90, square 0.45 -> only square (0.50) survives
    target.querySelector<HTMLButtonElement>('[data-per-class-import-btn]')!.click();
    flushSync();
    const ta = target.querySelector<HTMLTextAreaElement>('[data-per-class-import] textarea')!;
    ta.value = JSON.stringify({
      global: { best_conf: 0.3 },
      per_class: [
        { class: 0, best_conf: 0.9 },
        { class: 1, best_conf: 0.45 },
      ],
    });
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    target.querySelector<HTMLButtonElement>('[data-per-class-apply]')!.click();
    flushSync();
    expect(target.querySelectorAll('#pg-panel tbody tr')).toHaveLength(1);
    expect(target.querySelector('#pg-panel tbody tr')?.textContent).toContain('square');
    expect(target.querySelector('[data-per-class-badge]')?.textContent).toBe('2 overrides');
    expect(predictCalls).toBe(1);

    // Switch to Classes tab and verify custom override badge renders
    const classesTab = [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((t) => t.textContent?.includes('Classes'))!;
    classesTab.click();
    flushSync();
    expect(target.querySelector('[data-class-conf-override="1"]')?.textContent?.trim()).toBe('≥0.45');

    // Switch to JSON tab and confirm per_class_conf is exported
    const jsonTab = [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((t) => t.textContent?.includes('JSON'))!;
    jsonTab.click();
    const pre = await until(() => target!.querySelector('#pg-panel pre'), 'JSON view');
    const parsed = JSON.parse((pre.textContent ?? '').replace(/\u00a0/g, ' ')) as { filter: { per_class_conf?: Record<string, number> }; boxes: unknown[] };
    expect(parsed.filter.per_class_conf).toEqual({ '0': 0.9, '1': 0.45 });
    expect(parsed.boxes).toHaveLength(1);

    // Reset all restores all 3 detections (at global 0.30) with zero extra requests
    target.querySelector<HTMLButtonElement>('[data-per-class-reset-all]')!.click();
    flushSync();
    const detTab = [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((t) => t.textContent?.includes('Detections'))!;
    detTab.click();
    flushSync();
    expect(target.querySelectorAll('#pg-panel tbody tr')).toHaveLength(3);
    expect(target.querySelector('[data-per-class-badge]')?.textContent).toBe('Default');
    expect(predictCalls).toBe(1);
  });

  it('renders INT8 precision badge and precisionFallback notice cleanly in PlaygroundTiming', () => {
    const pred: Prediction = {
      model_id: 'm_q8',
      image: { width: 640, height: 640 },
      boxes: [],
      timing: { preprocess_ms: 2, inference_ms: 14, postprocess_ms: 1, total_ms: 17 },
      device: 'cpu',
      backend: 'cpu',
    };
    const int8Info: DetectorInfo = {
      backend: 'cpu',
      device: 'cpu',
      adapter: null,
      fallback: null,
      precision: 'int8',
      precisionFallback: null,
      family: 'folo2',
      arch: 'folo2-n',
      headMode: 'o2o',
      hasIouHead: true,
      imgsz: 640,
      classes: ['circle'],
      modelId: 'm_q8',
    };
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <PlaygroundTiming
          compare={false}
          narrow={false}
          modelA={null}
          modelB={null}
          predA={pred}
          predB={null}
          predictingA={false}
          predictingB={false}
          resultMsA={0.12}
          engineInfo={int8Info}
        />
      ),
      target,
    );
    flushSync();
    expect(target.querySelector('[data-browser-precision="int8"]')?.textContent).toBe('INT8');
    expect(target.querySelector('[data-browser-engine-info]')?.textContent).toContain('folo2-n · FOLO v2 · o2o · IoU head');
    expect(target.querySelector('[data-browser-engine-info]')?.textContent).not.toContain('· INT8');
    expect(target.querySelector('[data-precision-fallback]')).toBeNull();

    mounted.unmount();
    const fallbackInfo: DetectorInfo = {
      ...int8Info,
      backend: 'webgpu',
      device: 'gpu:0',
      precision: 'f32',
      precisionFallback: 'int8 inference runs on the CPU; running in f32 on WebGPU',
    };
    mounted = render(
      () => (
        <PlaygroundTiming
          compare={false}
          narrow={false}
          modelA={null}
          modelB={null}
          predA={{ ...pred, backend: 'webgpu', device: 'gpu:0' }}
          predB={null}
          predictingA={false}
          predictingB={false}
          resultMsA={0.08}
          engineInfo={fallbackInfo}
        />
      ),
      target,
    );
    flushSync();
    expect(target.querySelector('[data-browser-precision="int8"]')).toBeNull();
    expect(target.querySelector('[data-precision-fallback]')?.textContent).toBe(
      'int8 inference runs on the CPU; running in f32 on WebGPU',
    );
    expect(target.querySelector('[data-browser-engine-info]')?.textContent).not.toContain('· INT8');
  });
});
