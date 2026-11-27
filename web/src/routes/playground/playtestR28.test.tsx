// @vitest-environment happy-dom
// Regression tests for Live Studio Playtest Round 28 findings + PR #828 R3 follow-up:
// 1. BenchCompare waits for all compared entry results (robustness conditions, calibration, self-labeling) and condition
//    names to settle before enabling Export comparison / Copy Markdown / Download CSV (never exporting partial files).
// 2. Playground live webcam `[data-webcam-fps]` waits for a measured inter-frame interval instead of flashing an
//    inflated startup FPS computed from single-frame inference latency (`1000 / latencyMs`).
// 3. Playground paused video seek with Auto off invalidates `lastFrameMeta` and blocks re-recording a pre-seek frame
//    if Track IDs is toggled off and back on before clicking Detect frame.
import { flush as flushSync, tick } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { BenchCatalog, BenchResult, PredictBox } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import BenchCompare from '../BenchCompare';
import Playground from '../Playground';
import { session } from './session';
import { useMediaSession } from './useMediaSession';
import { useTracking } from './useTracking';
import { BrowserEngine, type WasmModule } from '$lib/inference/browser';
import { signal } from 'fw';
import type { Prediction } from '$lib/api/types';

let target: HTMLElement;
let view: Rendered | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
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

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  session.set(null);
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  session.set(null);
  toast.items = [];
  vi.restoreAllMocks();
});

describe('Playtest R28 regressions', () => {
  it('disables BenchCompare exports while entry results or catalog are loading, on result error, and when adding a new entry', async () => {
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });
    const baseApi = new MockApi('normal', 0);
    const origResult = baseApi.benchResult.bind(baseApi);
    const origCatalog = baseApi.benchCatalog.bind(baseApi);

    const pendingResults = new Map<string, { resolve: (r: BenchResult) => void; reject: (e: unknown) => void }>();
    let failOnceId: string | null = null;
    vi.spyOn(baseApi, 'benchResult').mockImplementation((benchId, entryId) => {
      if (failOnceId === entryId) {
        failOnceId = null;
        return Promise.reject(new Error('transient result error'));
      }
      return new Promise<BenchResult>((resolve, reject) => {
        pendingResults.set(entryId, { resolve, reject });
        void origResult(benchId, entryId);
      });
    });

    let resolveCatalog!: (c: BenchCatalog) => void;
    vi.spyOn(baseApi, 'benchCatalog').mockImplementation(
      () => new Promise<BenchCatalog>((r) => (resolveCatalog = r)),
    );

    const blobs = new Map<string, Blob>();
    const downloads: Array<{ filename: string; text: string }> = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      const url = `blob:r28-cmp/${blobs.size}`;
      blobs.set(url, b as Blob);
      return url;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      const blob = blobs.get(this.getAttribute('href') ?? '');
      if (blob) void blob.text().then((text) => downloads.push({ filename: this.download, text }));
    });

    hub.api = baseApi;
    hub.mode = 'mock';
    const route = '#/benchmarks/bn_5e12/compare?entries=be_5a21,be_5a20';
    history.replaceState(history.state, '', route);
    router.route = parse(route);
    view = render(() => <BenchCompare />, target);

    const getMenu = () => target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"][aria-label="Export comparison"]');
    const getCopy = () => target.querySelector<HTMLButtonElement>('[data-compare-copy-md]');
    const getCsv = () => target.querySelector<HTMLButtonElement>('[data-compare-download-csv]');

    await until(() => !!getMenu() && !!getCopy() && !!getCsv(), 'Export comparison controls mounted');

    // While detailed results & catalog are still in flight, all export controls stay disabled so no partial CSV can download.
    expect(getMenu()!.disabled).toBe(true);
    expect(getCopy()!.disabled).toBe(true);
    expect(getCsv()!.disabled).toBe(true);

    // Wait for both result fetches to register, then resolve both results while catalog is still pending -> still disabled.
    await until(() => pendingResults.has('be_5a21') && pendingResults.has('be_5a20'), 'both benchResult requests');
    pendingResults.get('be_5a21')!.resolve(await origResult('bn_5e12', 'be_5a21'));
    pendingResults.get('be_5a20')!.resolve(await origResult('bn_5e12', 'be_5a20'));
    pendingResults.clear();
    await tick();
    flushSync();
    expect(getMenu()!.disabled).toBe(true);
    expect(getCopy()!.disabled).toBe(true);
    expect(getCsv()!.disabled).toBe(true);

    // Once catalog resolves, all export controls enable and produce the full CSV including all condition rows & labels.
    resolveCatalog(await origCatalog());
    await until(() => !getMenu()?.disabled && !getCsv()?.disabled && !getCopy()?.disabled, 'export controls enabled');

    getMenu()!.click();
    flushSync();
    await tick();
    const exportCsvItem = [...target.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((el) => el.textContent?.trim() === 'Export CSV')!;
    expect(exportCsvItem).toBeDefined();
    exportCsvItem.click();
    await until(() => downloads.length === 1, 'complete CSV download');
    expect(downloads[0]!.filename).toBe('benchmark-shapes-val-compare.csv');
    expect(downloads[0]!.text).toContain('Score,');
    expect(downloads[0]!.text).toContain('Condition · ');
    const lineCount = downloads[0]!.text.trim().split(/\r?\n/).length;
    expect(lineCount).toBeGreaterThan(10);

    // Adding a 3rd entry whose result fails keeps export disabled until Retry succeeds.
    const adder = target.querySelector<HTMLSelectElement>('select[aria-label="Add an entry to the comparison"]');
    expect(adder).not.toBeNull();
    expect(adder!.options.length).toBeGreaterThan(1);
    const thirdId = adder!.options[1]!.value;
    failOnceId = thirdId;
    adder!.value = thirdId;
    adder!.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    await until(() => getMenu()?.disabled === true && getCsv()?.disabled === true && getCopy()?.disabled === true, 'export disabled while 3rd entry result failed');
    const retryBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Try again') || b.textContent?.includes('Retry')),
      'Retry button on failed result',
    );
    retryBtn.click();
    flushSync();
    await until(() => pendingResults.has(thirdId), 'retried 3rd entry result request');
    pendingResults.get(thirdId)!.resolve(await origResult('bn_5e12', thirdId));
    await until(() => !getMenu()?.disabled && !getCsv()?.disabled && !getCopy()?.disabled, 'export re-enabled after 3rd entry result loads');
  });

  it('does not flash an inflated startup FPS on [data-webcam-fps] after the first fast webcam frame', async () => {
    const stopTrack = vi.fn();
    const fakeStream = {
      getTracks: () => [{ stop: stopTrack, addEventListener: vi.fn(), removeEventListener: vi.fn() }],
    } as unknown as MediaStream;
    const origMediaDevices = navigator.mediaDevices;
    const origSrcObject = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
    const origCreateUrl = URL.createObjectURL;
    const origRevokeUrl = URL.revokeObjectURL;
    let assigned: MediaProvider | null = null;
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      configurable: true,
      get: () => assigned,
      set: (v: MediaProvider | null) => {
        assigned = v;
      },
    });
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => fakeStream) },
    });
    let blobSeq = 0;
    URL.createObjectURL = vi.fn(() => `blob:r28-cam-${++blobSeq}`);
    URL.revokeObjectURL = vi.fn();
    const ctxStub = {
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      restore: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 40 })),
      roundRect: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
      cb(new Blob(['jpg'], { type: 'image/jpeg' }));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});

    try {
      const api = new MockApi('normal', 0);
      let resolveSecond!: () => void;
      let calls = 0;
      const fastPred = {
        model_id: 'm_hard_s',
        image: { width: 320, height: 240 },
        boxes: [{ class: 0, name: 'circle', score: 0.9, x1: 10, y1: 10, x2: 50, y2: 50 }],
        timing: { preprocess_ms: 1, inference_ms: 4, postprocess_ms: 1, total_ms: 6 },
        device: 'gpu:0',
      };
      vi.spyOn(api, 'predict').mockImplementation(async () => {
        calls++;
        if (calls === 1) return fastPred;
        if (calls === 2) {
          await new Promise<void>((r) => (resolveSecond = r));
          return fastPred;
        }
        return fastPred;
      });

      hub.api = api;
      hub.mode = 'mock';
      const route = '#/playground?model=m_hard_s';
      history.replaceState(history.state, '', route);
      router.route = parse(route);
      view = render(() => <Playground />, target);

      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'Live camera button',
      );
      camBtn.click();
      flushSync();

      const decoder = await until(() => target.querySelector<HTMLVideoElement>('video.media-decoder'), 'webcam decoder');
      Object.defineProperty(decoder, 'videoWidth', { configurable: true, value: 320 });
      Object.defineProperty(decoder, 'videoHeight', { configurable: true, value: 240 });
      Object.defineProperty(decoder, 'readyState', { configurable: true, value: 4 });

      // Wait until frame 1 has settled and frame 2 is in flight.
      await until(() => calls >= 2 && typeof resolveSecond === 'function', 'first frame settled and second frame started');
      // After frame 1 only (6 ms latency), [data-webcam-fps] must NOT display a bogus ~166.7 FPS spike!
      expect(target.querySelector('[data-webcam-fps]')).toBeNull();

      // Release frame 2 after a realistic ~50 ms interval -> [data-webcam-fps] now shows measured FPS (well below 100 FPS).
      await new Promise((r) => setTimeout(r, 40));
      resolveSecond();
      const fpsEl = await until(() => target.querySelector<HTMLElement>('[data-webcam-fps]'), 'measured webcam FPS readout');
      const m = fpsEl.textContent?.match(/([\d.]+)\s*FPS\s*·\s*(\d+)\s*ms/);
      expect(m).not.toBeNull();
      const measuredFps = Number(m![1]);
      expect(measuredFps).toBeGreaterThan(0);
      expect(measuredFps).toBeLessThan(60);
      expect(Number(m![2])).toBe(6);
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
      if (origSrcObject) Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', origSrcObject);
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
  });

  it('does not re-record a pre-seek video frame if Track IDs is toggled off and back on after a paused seek with Auto off', async () => {
    const origCreateUrl = URL.createObjectURL;
    const origRevokeUrl = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn(() => 'blob:r28-seek-vid');
    URL.revokeObjectURL = vi.fn();
    const ctxStub = {
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      restore: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 40 })),
      roundRect: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
      cb(new Blob(['jpg'], { type: 'image/jpeg' }));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    let nextTrackId = 1;
    class FakeTracker {
      update(boxes: PredictBox[]) {
        return boxes.map((b) => ({ ...b, track_id: nextTrackId }));
      }
      reset() {
        nextTrackId++;
      }
      free() {}
    }
    const fakeWasm = {
      default: vi.fn(async () => {}),
      FoloDetector: class {},
      FoloTracker: FakeTracker,
    } as unknown as WasmModule;
    const engine = new BrowserEngine(async () => fakeWasm);

    try {
      const pred = signal<Prediction | null>(null);
      const sourceKey = signal<string | null>(null);
      const predSourceKey = signal<string | null>(null);
      let frameSeqCounter = 0;
      let media!: ReturnType<typeof useMediaSession>;
      let tracking!: ReturnType<typeof useTracking>;
      const fakeVideo = document.createElement('video');
      Object.defineProperty(fakeVideo, 'videoWidth', { configurable: true, value: 320 });
      Object.defineProperty(fakeVideo, 'videoHeight', { configurable: true, value: 180 });
      Object.defineProperty(fakeVideo, 'duration', { configurable: true, value: 5 });
      Object.defineProperty(fakeVideo, 'readyState', { configurable: true, value: 4 });
      let t = 0;
      Object.defineProperty(fakeVideo, 'currentTime', {
        configurable: true,
        get: () => t,
        set: (v: number) => {
          t = v;
        },
      });

      function Harness() {
        media = useMediaSession({
          onFrame: (f) => {
            const key = `frame:${++frameSeqCounter}:${f.name}`;
            sourceKey.set(key);
            predSourceKey.set(key);
            pred.set({
              model_id: 'm1',
              image: { width: f.width, height: f.height },
              boxes: [{ class: 0, name: 'circle', score: 0.9, x1: 10, y1: 10, x2: 50, y2: 50 }],
              timing: { preprocess_ms: 1, inference_ms: 3, postprocess_ms: 1, total_ms: 5 },
              device: 'cpu',
            });
          },
          isPredicting: () => false,
          isInferenceBlocked: () => false,
          hasValidResult: () => !!pred.peek(),
          lastLatencyMs: () => 5,
          onTrackReset: () => tracking?.reset(),
        });
        media.videoEl.set(fakeVideo);
        const isUnsyncedVideo = () => media.mode.get() === 'video' && !media.frameSynced.get();
        tracking = useTracking({
          engine,
          browserOff: () => null,
          mediaMode: () => media.mode.get(),
          mediaKey: () => `${media.mode.get()}:${media.sessionSeq.get()}`,
          frameKey: () => (media.mode.get() === 'none' ? null : `${media.mode.get()}:${media.sessionSeq.get()}:${media.frameSeq.get()}`),
          frameMeta: () => media.lastFrameMeta.get(),
          predMatchesFrame: () => {
            const s = sourceKey.get();
            return !!s && !isUnsyncedVideo() && !media.seeking.get() && !!media.lastFrameMeta.get() && predSourceKey.get() === s;
          },
          modelId: () => 'm1',
          classes: () => ['circle'],
          pred: () => pred.get(),
          conf: () => 0.25,
          iou: () => 0.45,
          maxDet: () => 300,
          o2o: () => false,
        });
        return <div />;
      }

      view = render(() => <Harness />, target);
      flushSync();

      // Load video, turn Auto OFF, enable Track IDs, capture frame 1 (t=0s) and +1f frame 2 (t~0.035s).
      media.autoDetect.set(false);
      media.loadVideo(new File(['v'], 'clip.mp4', { type: 'video/mp4' }));
      media.videoEvents.onloadedmetadata({ currentTarget: fakeVideo } as unknown as Event & { currentTarget: HTMLVideoElement });
      media.videoEvents.onloadeddata({ currentTarget: fakeVideo } as unknown as Event & { currentTarget: HTMLVideoElement });
      tracking.setEnabled(true);
      await until(() => tracking.getObservations().length === 1, 'frame 1 tracked');

      media.stepFrame(1);
      media.videoEvents.onseeked({ currentTarget: fakeVideo } as unknown as Event & { currentTarget: HTMLVideoElement });
      await media.detectCurrentFrame();
      await until(() => tracking.getObservations().length === 2, 'frame 2 tracked');
      expect(tracking.getObservations().map((o) => o.frame)).toEqual([1, 2]);

      // Scrub to t = 2.0 s (non-forward seek with Auto OFF) -> resets tracker, clears lastFrameMeta, marks unsynced.
      media.seekTo(2.0, false);
      media.videoEvents.onseeked({ currentTarget: fakeVideo } as unknown as Event & { currentTarget: HTMLVideoElement });
      flushSync();
      expect(media.lastFrameMeta.get()).toBeNull();
      expect(media.frameSynced.get()).toBe(false);
      expect(tracking.getObservations()).toHaveLength(0);

      // Toggle Track IDs OFF and back ON BEFORE clicking Detect frame -> must NOT re-record the stale pre-seek frame!
      tracking.setEnabled(false);
      flushSync();
      tracking.setEnabled(true);
      await tick();
      flushSync();
      expect(tracking.getObservations()).toHaveLength(0);

      // Now click Detect frame at t = 2.0 s -> starts fresh cleanly at frame 1, timestamp 2000 ms.
      await media.detectCurrentFrame();
      await until(() => tracking.getObservations().length === 1, 'post-seek frame 1 tracked');
      const obs = tracking.getObservations();
      expect(obs).toHaveLength(1);
      expect(obs[0]!.frame).toBe(1);
      expect(obs[0]!.timestamp_ms).toBe(2000);
    } finally {
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
  });
});
