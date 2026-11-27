// @vitest-environment happy-dom
// Integration & edge-case tests for Playground per-class candidate refresh (server & browser floors, batch Run All/QA),
// class-aware ByteTrack tracking, cross-model session restore, browser INT8 weights/fallback/local-file handling, and
// ResultsPanel per-class empty-state UX.
import { createRoot, effect, flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { PredictBox, Prediction } from '$lib/api/types';
import { BrowserEngine, resetWasmProbe, type FoloTrackConfig, type FoloTrackedBox, type FoloTrackInputBox, type FoloTrackOptions, type FoloTrackerInfo, type Tracker, type WasmModule } from '$lib/inference/browser';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import {
  askAllWhenCut,
  loosePerClassFloor,
  needsCandidateRefresh,
  needsPerClassFloorRefresh,
  recordFetchedFloors,
} from './allCandidates';
import type { PerClassConfMap } from './perClassConf';
import { ask, browserWeightsFormat, FETCH, serverFetchConf } from './predict';
import PlaygroundTiming from './PlaygroundTiming';
import ResultsPanel from './ResultsPanel';
import { session } from './session';
import type { Source } from './source';
import { useBatchSession } from './useBatchSession';
import { useInferenceOptions } from './useInferenceOptions';
import { useSinglePredict } from './useSinglePredict';
import { useTracking } from './useTracking';

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

describe('Playground per-class candidate floors, tracking, session restore & browser INT8 precision', () => {
  it('requests stored weights for CPU auto/int8 and f16 for WebGPU/f32/f16', async () => {
    expect(browserWeightsFormat('cpu', 'auto')).toBe('stored');
    expect(browserWeightsFormat('cpu', 'int8')).toBe('stored');
    expect(browserWeightsFormat('cpu', 'f32')).toBe('f16');
    expect(browserWeightsFormat('cpu', 'f16')).toBe('f16');
    expect(browserWeightsFormat('webgpu', 'auto')).toBe('f16');
    expect(browserWeightsFormat('webgpu', 'int8')).toBe('f16');

    const api = new MockApi('normal', 0);
    Object.defineProperty(api, 'mock', { value: false });
    const tagFormats: (string | undefined)[] = [];
    const weightFormats: (string | undefined)[] = [];
    api.modelWeightsTag = async (_id, _sig, fmt) => {
      tagFormats.push(fmt);
      return `"etag-${fmt}"`;
    };
    api.modelWeights = async (_id, _sig, fmt) => {
      weightFormats.push(fmt);
      return new ArrayBuffer(8);
    };
    const mod: WasmModule = {
      default: async () => {},
      FoloDetector: {
        load: async (_bytes, { backend, modelId }) => ({
          backend,
          info: () => ({
            backend,
            device: backend === 'webgpu' ? 'gpu:0' : 'cpu',
            adapter: null,
            fallback: null,
            precision: 'int8',
            precisionFallback: null,
            arch: 'folo2-n',
            imgsz: 640,
            classes: ['a'],
            modelId,
          }),
          detect: async () => ({
            model_id: modelId,
            image: { width: 1, height: 1 },
            boxes: [],
            timing: { preprocess_ms: 0, inference_ms: 1, postprocess_ms: 0, total_ms: 1 },
            device: 'cpu',
            backend,
          }),
          free: () => {},
        }),
      },
    };
    const engine = new BrowserEngine(
      async () => mod,
      async () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }) as ImageData,
    );
    const src = { kind: 'file' as const, file: new Blob([new Uint8Array(4)]), name: 'x.png', url: 'blob:x', width: 1, height: 1 };
    await ask({ api, engine, model: 'm1', source: src, on: 'cpu', file: null, options: FETCH, precision: 'auto', signal: new AbortController().signal });
    await ask({ api, engine, model: 'm1', source: src, on: 'webgpu', file: null, options: FETCH, precision: 'auto', signal: new AbortController().signal });
    expect(tagFormats).toEqual(['stored', undefined]);
    expect(weightFormats).toEqual(['stored', undefined]);
    engine.dispose();
  });

  it('keeps loose browser fetches lossless above 0.01, honors server overrides below 0.01, and avoids aborting or replaying failed completions', async () => {
    expect(loosePerClassFloor({ 0: 0.8, 1: 0.005, 2: 0.01 })).toEqual({ 1: 0.005 });
    expect(loosePerClassFloor({ 0: 0.8 })).toBeUndefined();
    expect(serverFetchConf(0.25)).toBe(0.01);
    expect(serverFetchConf(0.005)).toBeLessThan(0.005);
    expect(serverFetchConf(0)).toBe(0);

    const serverPred: Prediction = {
      model_id: 'm1',
      image: { width: 100, height: 100 },
      boxes: [{ class: 0, name: 'circle', score: 0.5, x1: 0, y1: 0, x2: 10, y2: 10 }],
      timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
      device: 'gpu:0',
    };
    recordFetchedFloors(serverPred, 0.01, undefined, false);
    expect(needsPerClassFloorRefresh(serverPred, 0.25, { 0: 0.5 })).toBe(false);
    expect(needsPerClassFloorRefresh(serverPred, 0.25, { 0: 0.005 })).toBe(true);
    expect(needsCandidateRefresh(serverPred, 0.25, { 0: 0.005 }, 'server')?.fetchConf).toBeLessThan(0.005);

    const refreshed: Prediction = {
      ...serverPred,
      boxes: [
        ...serverPred.boxes,
        { class: 0, name: 'circle', score: 0.005, x1: 20, y1: 20, x2: 30, y2: 30 },
      ],
    };
    recordFetchedFloors(refreshed, serverFetchConf(0.005), undefined, false);
    expect(needsPerClassFloorRefresh(refreshed, 0.25, { 0: 0.005 })).toBe(false);
    expect(needsPerClassFloorRefresh(refreshed, 0.25, { 0: 0.001 })).toBe(true);

    const api = new MockApi('normal', 0);
    let release!: () => void;
    let calls = 0;
    let failNext = false;
    api.predict = async (_input, opts) => {
      calls++;
      if (failNext) throw new Error('server down');
      await new Promise<void>((r) => (release = r));
      return {
        model_id: opts.model_id,
        image: { width: 100, height: 100 },
        boxes: [
          { class: 0, name: 'circle', score: 0.5, x1: 0, y1: 0, x2: 10, y2: 10 },
          { class: 0, name: 'circle', score: 0.005, x1: 20, y1: 20, x2: 30, y2: 30 },
        ],
        timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
        device: 'gpu:0',
      };
    };
    hub.api = api;
    hub.mode = 'live';
    const engine = new BrowserEngine(async () => {
      throw new Error('unused');
    });
    const predSig = signal<Prediction | null>(serverPred);
    const confSig = signal(0.25);
    const pccSig = signal<PerClassConfMap>({ 0: 0.005 });
    const src = { kind: 'file' as const, file: new Blob([new Uint8Array(4)]), name: 'a.png', url: 'blob:a', width: 100, height: 100 };
    const dispose = createRoot((d) => {
      askAllWhenCut({
        pred: predSig,
        conf: () => confSig.get(),
        perClassConf: () => pccSig.get(),
        model: () => 'm1',
        source: () => src,
        runOn: () => 'server',
        file: () => null,
        busy: () => false,
        engine,
        weightsMissing: () => {},
      });
      return d;
    });
    flushSync();
    expect(calls).toBe(1);
    confSig.set(0.35);
    flushSync();
    expect(calls).toBe(1);
    release();
    await vi.waitFor(() => expect(predSig.peek()?.boxes).toHaveLength(2));

    failNext = true;
    pccSig.set({ 0: 0.001 });
    flushSync();
    await vi.waitFor(() => expect(calls).toBe(2));
    confSig.set(0.4);
    flushSync();
    expect(calls).toBe(2);
    dispose();
    engine.dispose();
  });

  it('refreshes under-floored cached batch items before QA export and Run All', async () => {
    const api = new MockApi('normal', 0);
    const predictConfs: number[] = [];
    api.predict = async (_input, opts) => {
      predictConfs.push(opts.conf ?? 0.01);
      const lowBox =
        (opts.conf ?? 0.01) < 0.005
          ? [{ class: 0, name: 'circle', score: 0.005, x1: 10, y1: 10, x2: 40, y2: 40 }]
          : [];
      return {
        model_id: opts.model_id,
        image: { width: 100, height: 100 },
        boxes: lowBox,
        timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
        device: 'gpu:0',
      };
    };
    hub.api = api;
    hub.mode = 'live';
    const engine = new BrowserEngine(async () => {
      throw new Error('unused');
    });
    const runOn = signal<'server' | 'webgpu' | 'cpu'>('server');
    const localWeights = signal<Map<string, File>>(new Map());
    const compare = signal(false);
    const conf = signal(0.25);
    const iou = signal(0.7);
    const maxDet = signal(300);
    const perClassConf = signal<PerClassConfMap>({});
    const hidden = signal(new Set<number>());
    const source = signal<Source | null>(null);
    const pred = signal<Prediction | null>(null);
    const predSourceKey = signal<string | null>(null);
    const predicting = signal(false);
    const error = signal<unknown>(null);
    const weightsMissing = signal(false);
    const hover = signal<PredictBox | null>(null);
    const picked = signal<PredictBox | null>(null);

    let batch!: ReturnType<typeof useBatchSession>;
    const dispose = createRoot((d) => {
      batch = useBatchSession({
        engine,
        model: () => ({ id: 'm1', name: 'shapes-n', classes: ['circle'], arch: 'folo-n', task: 'detect', params: 1, flops_g: 1, bytes: 1, imgsz: 640, created: '', run_id: null, metrics: { map50: 0.9, map50_95: 0.7, precision: 0.9, recall: 0.9 } }),
        modelId: () => 'm1',
        runOn,
        localWeights,
        compare,
        o2o: () => false,
        conf,
        iou,
        maxDet,
        perClassConf,
        hidden,
        classes: () => ['circle'],
        source,
        pred,
        predSourceKey,
        predicting,
        error,
        weightsMissing,
        hover,
        picked,
        choose: () => () => true,
        cancelSinglePredict: () => {},
        onSingleFile: () => {},
        onActivateFrame: () => {},
        onEnterBatch: () => {},
        dropLocal: () => {},
      });
      return d;
    });

    const f1 = new File([new Uint8Array([1])], 'img1.png', { type: 'image/png' });
    const f2 = new File([new Uint8Array([2])], 'img2.png', { type: 'image/png' });
    batch.items.set([
      { id: '1', file: f1, name: 'img1.png', url: 'blob:1', width: 100, height: 100, pred: null, predKey: null, status: 'idle', error: null },
      { id: '2', file: f2, name: 'img2.png', url: 'blob:2', width: 100, height: 100, pred: null, predKey: null, status: 'idle', error: null },
    ]);
    await batch.runAll();
    expect(predictConfs).toEqual([0.01, 0.01]);
    expect(batch.items.peek().every((i) => i.pred?.boxes.length === 0)).toBe(true);

    perClassConf.set({ 0: 0.005 });
    flushSync();
    await batch.runAll();
    expect(predictConfs).toHaveLength(4);
    expect(predictConfs[2]).toBeLessThan(0.005);
    expect(batch.items.peek().every((i) => i.pred?.boxes.length === 1)).toBe(true);
    dispose();
    engine.dispose();
  });

  it('applies per-class thresholds in ByteTrack tracking while preserving low-confidence recovery', async () => {
    const trackerOpts: FoloTrackOptions[] = [];
    const seenUpdateBoxes: number[][] = [];
    const mod: WasmModule = {
      default: async () => {},
      FoloDetector: {
        load: async () => {
          throw new Error('unused');
        },
      },
      FoloTracker: class implements Tracker {
        constructor(opts?: FoloTrackOptions) {
          trackerOpts.push({ ...opts });
        }
        update(boxes: readonly FoloTrackInputBox[]): FoloTrackedBox[] {
          seenUpdateBoxes.push(boxes.map((b) => Math.round(b.score * 100)));
          return boxes.map((b, i) => ({
            ...b,
            name: b.name ?? String(b.class),
            track_id: i + 1,
            age: 1,
            hits: 1,
            state: 'tracked',
          }));
        }
        reset() {}
        config(): FoloTrackConfig {
          return { highThresh: 0.25, lowThresh: 0.1, newTrackThresh: 0.25, matchIou: 0.8, secondMatchIou: 0.5, tentativeMatchIou: 0.5, trackBuffer: 30, minHits: 1, classAware: true };
        }
        info(): FoloTrackerInfo {
          return { frameId: 0, activeTracks: 0, lostTracks: 0, config: this.config() };
        }
        free() {}
      } as unknown as WasmModule['FoloTracker'],
    };
    const engine = new BrowserEngine(async () => mod);
    const mediaMode = signal<'none' | 'webcam' | 'video'>('webcam');
    const predSig = signal<Prediction | null>(null);
    const frameSeq = signal(1);
    const confSig = signal(0.25);
    const pccSig = signal<PerClassConfMap>({ 0: 0.7 });

    let tracking!: ReturnType<typeof useTracking>;
    const dispose = createRoot((d) => {
      tracking = useTracking({
        engine,
        browserOff: () => null,
        mediaMode: () => mediaMode.get(),
        mediaKey: () => 'webcam:1',
        frameKey: () => `webcam:1:${frameSeq.get()}`,
        modelId: () => 'm1',
        classes: () => ['person', 'car'],
        pred: () => predSig.get(),
        conf: () => confSig.get(),
        perClassConf: () => pccSig.get(),
        iou: () => 0.7,
        maxDet: () => 300,
        o2o: () => false,
      });
      return d;
    });

    tracking.setEnabled(true);
    flushSync();
    await vi.waitFor(() => expect(trackerOpts).toHaveLength(1));
    expect(trackerOpts[0]?.highThresh).toBe(0.25);

    const f1: Prediction = {
      model_id: 'm1',
      image: { width: 640, height: 480 },
      boxes: [
        { class: 0, name: 'person', score: 0.5, x1: 0, y1: 0, x2: 50, y2: 50 },
        { class: 0, name: 'person', score: 0.85, x1: 100, y1: 0, x2: 150, y2: 50 },
        { class: 1, name: 'car', score: 0.4, x1: 200, y1: 0, x2: 260, y2: 50 },
      ],
      timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
      device: 'gpu:0',
    };
    predSig.set(f1);
    flushSync();
    expect(seenUpdateBoxes[0]).toEqual([85, 40]);

    frameSeq.set(2);
    predSig.set({
      ...f1,
      boxes: [{ class: 0, name: 'person', score: 0.15, x1: 102, y1: 0, x2: 152, y2: 50 }],
    });
    flushSync();
    expect(seenUpdateBoxes[1]).toEqual([15]);

    pccSig.set({ 0: 0.7, 1: 0.15 });
    flushSync();
    await vi.waitFor(() => expect(trackerOpts).toHaveLength(2));
    expect(trackerOpts[1]?.highThresh).toBe(0.15);
    frameSeq.set(3);
    predSig.set({
      ...f1,
      boxes: [
        { class: 0, name: 'person', score: 0.2, x1: 0, y1: 0, x2: 50, y2: 50 },
        { class: 1, name: 'car', score: 0.2, x1: 200, y1: 0, x2: 260, y2: 50 },
      ],
    });
    flushSync();
    expect(seenUpdateBoxes.at(-1)).toEqual([20]);
    dispose();
    engine.dispose();
  });

  it('retains a valid local .folo file when explicit INT8 load fails in useSinglePredict', async () => {
    const api = new MockApi('normal', 0);
    Object.defineProperty(api, 'mock', { value: false });
    hub.api = api;
    hub.mode = 'live';
    // Valid minimal safetensors (.folo) header: 8-byte LE length (2) followed by `{}`.
    const validFolo = new File([new Uint8Array([2, 0, 0, 0, 0, 0, 0, 0, 0x7b, 0x7d])], 'quant.folo');
    const mod: WasmModule = {
      default: async () => {},
      FoloDetector: {
        load: async () => {
          throw new Error('checkpoint has no int8 quantization metadata');
        },
      },
    };
    const engine = new BrowserEngine(async () => mod);
    const dropLocal = vi.fn();
    const runOn = signal<'server' | 'webgpu' | 'cpu'>('cpu');
    const localWeights = signal<Map<string, File>>(new Map([['m1', validFolo]]));
    const source = signal<Source | null>({ kind: 'file', file: new Blob([new Uint8Array(4)]), name: 'a.png', url: 'blob:a', width: 10, height: 10 });
    const pred = signal<Prediction | null>(null);
    const predSourceKey = signal<string | null>(null);
    const predicting = signal(false);
    const error = signal<unknown>(null);
    const weightsMissing = signal(false);
    const hidden = signal(new Set<number>());
    const hover = signal<PredictBox | null>(null);
    const picked = signal<PredictBox | null>(null);
    const conf = signal(0.25);
    const perClassConf = signal<PerClassConfMap>({});
    const retryTick = signal(0);

    const dispose = createRoot((d) => {
      useSinglePredict({
        modelId: () => 'm1',
        source,
        runOn,
        localWeights,
        retryTick,
        restored: () => false,
        consumeReuse: () => false,
        pred,
        predSourceKey,
        predicting,
        error,
        weightsMissing,
        hidden,
        hover,
        picked,
        conf,
        perClassConf,
        engine,
        precision: () => 'int8',
        dropLocal,
        isLiveStream: () => false,
        isBatchRunningAll: () => false,
        getCachedBatchPred: () => null,
        wasBatchStopped: () => false,
        onBatchResult: () => {},
        onBatchBusy: () => {},
      });
      return d;
    });
    flushSync();
    await vi.waitFor(() => expect(error.peek()).toBeTruthy());
    expect(dropLocal).not.toHaveBeenCalled();
    expect(String(error.peek())).toContain('int8');
    dispose();
    engine.dispose();
  });

  it('keeps Server effectiveFor stable on browser precision toggles and scopes activePrecisionFallback per prediction', async () => {
    const mod: WasmModule = {
      default: async () => {},
      FoloDetector: {
        load: async (_bytes, { backend, modelId, precision }) => ({
          backend,
          info: () => ({
            backend,
            device: backend === 'webgpu' ? 'gpu:0' : 'cpu',
            adapter: null,
            fallback: null,
            precision: backend === 'webgpu' && precision === 'int8' ? 'f32' : (precision ?? 'int8'),
            precisionFallback: backend === 'webgpu' && precision === 'int8' ? 'int8 runs on CPU; using f32 on WebGPU' : null,
            arch: 'folo2-n',
            imgsz: 640,
            classes: ['a'],
            modelId,
          }),
          detect: async () => ({
            model_id: modelId,
            image: { width: 1, height: 1 },
            boxes: [],
            timing: { preprocess_ms: 0, inference_ms: 1, postprocess_ms: 0, total_ms: 1 },
            device: backend === 'webgpu' ? 'gpu:0' : 'cpu',
            backend,
          }),
          free: () => {},
        }),
      },
    };
    const engine = new BrowserEngine(
      async () => mod,
      async () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }) as ImageData,
    );
    const pWebgpuFallback = await engine.predict({
      modelId: 'mA',
      file: null,
      download: async () => new Uint8Array([1]).buffer,
      tag: '"v1"',
      backend: 'webgpu',
      precision: 'int8',
      image: async () => new Blob(),
      options: { conf: 0.01, iou: 1, maxDet: 1000 },
    });
    const pCpuInt8 = await engine.predict({
      modelId: 'mB',
      file: null,
      download: async () => new Uint8Array([2]).buffer,
      tag: '"v2"',
      backend: 'cpu',
      precision: 'auto',
      image: async () => new Blob(),
      options: { conf: 0.01, iou: 1, maxDet: 1000 },
    });

    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <PlaygroundTiming
          compare={true}
          narrow={false}
          modelA={null}
          modelB={null}
          predA={pWebgpuFallback}
          predB={pCpuInt8}
          predictingA={false}
          predictingB={false}
          resultMsA={0.1}
          engine={engine}
        />
      ),
      target,
    );
    flushSync();
    expect(target.querySelector('[data-precision-fallback]')?.textContent).toContain('runs on CPU');
    expect(target.querySelector('[data-browser-precision="int8"]')?.textContent).toBe('INT8');

    const runOn = signal<'server' | 'webgpu' | 'cpu'>('server');
    const runOnB = signal<'server' | 'webgpu' | 'cpu'>('webgpu');
    const activePreds = signal<(Prediction | null)[]>([pWebgpuFallback]);
    let serverEffectRuns = 0;
    let inf!: ReturnType<typeof useInferenceOptions>;
    const dispose = createRoot((d) => {
      inf = useInferenceOptions({
        runOn,
        compareRunOn: () => runOnB.get(),
        activePredictions: () => activePreds.get(),
        engine,
        browserOff: () => null,
        serverPredictOptions: () => true,
      });
      effect(() => {
        void inf.effectiveFor('server');
        serverEffectRuns++;
      });
      return d;
    });
    flushSync();
    expect(serverEffectRuns).toBe(1);
    inf.precision.set('int8');
    flushSync();
    expect(serverEffectRuns).toBe(1);
    expect(inf.activePrecisionFallback.get()).toContain('runs on CPU');
    inf.precision.set('f32');
    flushSync();
    expect(serverEffectRuns).toBe(1);
    expect(inf.activePrecisionFallback.get()).toBeNull();
    dispose();
    engine.dispose();
  });

  it('clears incompatible restored per-class overrides when opening Playground on a different model', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';

    session.set({
      source: null,
      pred: null,
      runOn: 'server',
      conf: 0.25,
      iou: 0.7,
      maxDet: 300,
      perClassConf: { 0: 0.85 },
      perClassModelId: 'm_shapes_s',
      perClassClasses: ['circle', 'square', 'triangle'],
      labels: true,
      scores: true,
      truth: false,
      hidden: [],
    });

    const models = await api.models();
    const other = models.find((m) => m.id !== 'm_shapes_s' && m.classes[0] !== 'circle');
    expect(other).toBeTruthy();

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse(`#/playground?model=${other!.id}`);
    mounted = render(() => <Playground />, target);
    await until(() => target!.querySelector('[data-per-class-badge]')?.textContent === 'Default', 'cleared cross-model override');
    expect(session.get()?.perClassConf).toEqual({});
  });

  it('offers Reset per-class thresholds in ResultsPanel empty state when per-class overrides filter out all boxes', () => {
    const onReset = vi.fn();
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <ResultsPanel
          result={{ boxes: [], ms: 0.1 }}
          rows={[]}
          predicting={false}
          notice={null}
          json={null}
          pred={{
            model_id: 'm1',
            image: { width: 100, height: 100 },
            boxes: [{ class: 0, name: 'circle', score: 0.5, x1: 0, y1: 0, x2: 10, y2: 10 }],
            timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
            device: 'gpu:0',
          }}
          conf={0.25}
          hasPerClassOverrides={true}
          onResetPerClass={onReset}
          hidden={new Set()}
          tabRow={null}
          activeIdx={null}
          selectedIdx={null}
          onrowkey={() => {}}
        />
      ),
      target,
    );
    flushSync();
    expect(target.textContent).toContain('per-class threshold');
    const resetBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) =>
      b.textContent?.includes('Reset per-class thresholds'),
    );
    expect(resetBtn).toBeTruthy();
    resetBtn!.click();
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
