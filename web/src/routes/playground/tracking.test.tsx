// @vitest-environment happy-dom
// Playground live ByteTrack (`FoloTracker`) lifecycle, trail rendering, graceful fallback without `FoloTracker`,
// `track_id` in ResultsPanel & JSON output, and `folo2` Browser Engine status badge tests.
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import type { Prediction } from '$lib/api/types';
import {
  BrowserEngine,
  detectorInfoBadge,
  type DetectorInfo,
  type FoloTrackedBox,
  type FoloTrackInputBox,
  type FoloTrackOptions,
  type Tracker,
  type WasmModule,
} from '$lib/inference/browser';
import { drawAnnotations } from './export';
import MediaBar from './MediaBar';
import PlaygroundTiming from './PlaygroundTiming';
import ResultsPanel from './ResultsPanel';
import { formatBoxChipText, TrackTrailHistory, trackerOptionsForConf } from './tracking';
import type { MediaMode, MediaSession } from './useMediaSession';
import { useTracking } from './useTracking';
import { predictionJson } from './viewState';

interface FakeTrackerSpy {
  created: FoloTrackOptions[];
  updates: FoloTrackInputBox[][];
  resets: number;
  frees: number;
}

function makeFakeWasm(withTracker: boolean) {
  const spy: FakeTrackerSpy = { created: [], updates: [], resets: 0, frees: 0 };
  let nextId = 1;
  const mod: WasmModule = {
    default: async () => {},
    FoloDetector: {
      load: async () => {
        throw new Error('not used in tracker test');
      },
    },
    ...(withTracker
      ? {
          FoloTracker: class FakeFoloTracker implements Tracker {
            #opts: FoloTrackOptions;
            #ids = new Map<number, number>();
            #frameId = 0;
            constructor(opts: FoloTrackOptions = {}) {
              this.#opts = opts;
              spy.created.push(opts);
            }
            update(boxes: readonly FoloTrackInputBox[]): FoloTrackedBox[] {
              this.#frameId++;
              spy.updates.push([...boxes]);
              const high = this.#opts.highThresh ?? 0.5;
              const out: FoloTrackedBox[] = [];
              for (const b of boxes) {
                if (b.score < high) continue;
                let id = this.#ids.get(b.class);
                if (!id) {
                  id = nextId++;
                  this.#ids.set(b.class, id);
                }
                out.push({
                  track_id: id,
                  age: this.#frameId,
                  hits: this.#frameId,
                  state: 'tracked',
                  class: b.class,
                  name: b.name ?? `class${b.class}`,
                  score: b.score,
                  x1: b.x1,
                  y1: b.y1,
                  x2: b.x2,
                  y2: b.y2,
                });
              }
              return out.sort((a, b) => a.track_id - b.track_id);
            }
            reset(): void {
              spy.resets++;
              this.#ids.clear();
              this.#frameId = 0;
              nextId = 1;
            }
            config() {
              return {
                highThresh: this.#opts.highThresh ?? 0.5,
                lowThresh: this.#opts.lowThresh ?? 0.1,
                newTrackThresh: this.#opts.newTrackThresh ?? 0.6,
                matchIou: 0.2,
                secondMatchIou: 0.5,
                tentativeMatchIou: 0.3,
                trackBuffer: 30,
                minHits: 1,
                classAware: true,
              };
            }
            info() {
              return { frameId: this.#frameId, activeTracks: this.#ids.size, lostTracks: 0, config: this.config() };
            }
            free(): void {
              spy.frees++;
            }
          },
        }
      : {}),
  };
  return { mod, spy };
}

function predFrame(x: number, score = 0.9, modelId = 'm1'): Prediction {
  return {
    model_id: modelId,
    image: { width: 640, height: 360 },
    boxes: [{ class: 0, name: 'circle', score, x1: x, y1: 20, x2: x + 40, y2: 60 }],
    timing: { preprocess_ms: 1, inference_ms: 4, postprocess_ms: 1, total_ms: 6 },
    device: 'cpu',
    backend: 'cpu',
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('BrowserEngine FoloTracker bridge and folo2 detector badge', () => {
  it('creates FoloTracker when exported and falls back to null when missing on older WASM builds', async () => {
    const withT = makeFakeWasm(true);
    const e1 = new BrowserEngine(async () => withT.mod);
    const tr = await e1.createTracker(trackerOptionsForConf(0.25, ['circle']));
    expect(tr).not.toBeNull();
    expect(e1.capabilities.tracker).toBe(true);
    tr!.free();
    expect(withT.spy.frees).toBe(1);

    const withoutT = makeFakeWasm(false);
    const e2 = new BrowserEngine(async () => withoutT.mod);
    expect(await e2.createTracker()).toBeNull();
    expect(e2.capabilities.tracker).toBe(false);
  });

  it('formats concise folo2 browser engine info badges', () => {
    const info: DetectorInfo = {
      backend: 'webgpu',
      device: 'gpu:0',
      adapter: 'RTX 4080',
      fallback: null,
      arch: 'folo2-s',
      imgsz: 640,
      classes: ['circle'],
      family: 'folo2',
      headMode: 'o2o',
      hasO2o: true,
      hasIouHead: true,
      precision: 'f16',
    };
    expect(detectorInfoBadge(info)).toBe('folo2-s · FOLO v2 · o2o · IoU head · f16');
  });
});

describe('TrackTrailHistory & canvas annotation rendering', () => {
  it('accumulates bounded center points, prunes missing tracks, clears on reset, and draws trails + #id chips', () => {
    const hist = new TrackTrailHistory(3, 1);
    expect(hist.update([{ track_id: 1, class: 0, x1: 0, y1: 0, x2: 20, y2: 20 }])).toEqual([]);
    const t2 = hist.update([{ track_id: 1, class: 0, x1: 10, y1: 0, x2: 30, y2: 20 }]);
    expect(t2).toHaveLength(1);
    expect(t2[0]!.points).toEqual([
      { x: 10, y: 10 },
      { x: 20, y: 10 },
    ]);
    hist.update([{ track_id: 1, class: 0, x1: 20, y1: 0, x2: 40, y2: 20 }]);
    const t4 = hist.update([{ track_id: 1, class: 0, x1: 30, y1: 0, x2: 50, y2: 20 }]);
    expect(t4[0]!.points).toHaveLength(3);
    // Missing 2 frames (> staleAfter=1) prunes the trail.
    expect(hist.update([])).toHaveLength(1);
    expect(hist.update([])).toHaveLength(0);

    expect(formatBoxChipText({ class: 0, score: 0.91, track_id: 4 }, ['circle'], true)).toBe('#4 circle 0.91');

    const texts: string[] = [];
    const lines: number[][] = [];
    const ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn((x: number, y: number) => lines.push([x, y])),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      measureText: () => ({ width: 50 }),
      fillText: vi.fn((text: string) => texts.push(text)),
      setLineDash: vi.fn(),
      font: '',
      textBaseline: 'middle',
      lineWidth: 1,
      lineCap: 'butt',
      lineJoin: 'miter',
      strokeStyle: '',
      fillStyle: '',
      globalAlpha: 1,
    } as unknown as CanvasRenderingContext2D;

    drawAnnotations(
      ctx,
      {
        boxes: [{ class: 0, score: 0.91, x1: 30, y1: 0, x2: 50, y2: 20, track_id: 4 }],
        trails: t4,
        classes: ['circle'],
        labels: true,
        scores: true,
      },
      { ox: 0, oy: 0, width: 320, height: 180 },
    );
    expect(lines.length).toBeGreaterThanOrEqual(2);
    expect(texts).toContain('#4 circle 0.91');
  });
});

describe('useTracking lifecycle: update, reset on seek/model/source/toggle, free on unmount, fallback', () => {
  it('maintains one persistent FoloTracker across frames, resets on triggers, and frees on unmount', async () => {
    const { mod, spy } = makeFakeWasm(true);
    const engine = new BrowserEngine(async () => mod);
    const mediaMode = signal<MediaMode>('video');
    const mediaKey = signal<string | null>('video:1');
    const modelId = signal<string | null>('m1');
    const pred = signal<Prediction | null>(null);
    const conf = signal(0.25);
    const host = document.body.appendChild(document.createElement('div'));
    let ctrl!: ReturnType<typeof useTracking>;

    function Harness() {
      ctrl = useTracking({
        engine,
        browserOff: () => null,
        mediaMode: () => mediaMode.get(),
        mediaKey: () => mediaKey.get(),
        modelId: () => modelId.get(),
        classes: () => ['circle'],
        pred: () => pred.get(),
        conf: () => conf.get(),
        iou: () => 0.45,
        maxDet: () => 300,
        o2o: () => false,
      });
      return <div />;
    }

    const view = render(() => <Harness />, host);
    flushSync();

    // Enable Track IDs and feed 2 consecutive frames -> single persistent tracker, persistent track_id #1 + trail.
    ctrl.setEnabled(true);
    const f1 = predFrame(10);
    pred.set(f1);
    flushSync();
    await vi.waitFor(() => expect(ctrl.boxesFor(f1)?.[0]?.track_id).toBe(1));
    expect(spy.created).toHaveLength(1);
    expect(ctrl.activeCount.get()).toBe(1);

    const f2 = predFrame(24);
    pred.set(f2);
    flushSync();
    expect(ctrl.boxesFor(f2)?.[0]?.track_id).toBe(1);
    expect(ctrl.trails.get()).toHaveLength(1);
    expect(spy.updates).toHaveLength(2);

    // Seek reset -> calls tracker.reset() and clears trails & count without reallocating.
    ctrl.reset();
    flushSync();
    expect(spy.resets).toBe(1);
    expect(ctrl.trails.get()).toEqual([]);
    expect(ctrl.activeCount.get()).toBe(0);

    // Model switch -> calls tracker.reset().
    modelId.set('m2');
    flushSync();
    expect(spy.resets).toBe(2);

    // Source change (new clip or camera restart) -> calls tracker.reset().
    mediaKey.set('video:2');
    flushSync();
    expect(spy.resets).toBe(3);

    // Toggle off -> calls tracker.reset().
    ctrl.setEnabled(false);
    flushSync();
    expect(spy.resets).toBe(4);

    // Unmount -> calls tracker.free() exactly once (zero WASM memory leaks).
    view.unmount();
    host.remove();
    expect(spy.frees).toBe(1);
  });

  it('gracefully disables Track IDs in MediaBar with tooltip when FoloTracker is absent', async () => {
    const { mod } = makeFakeWasm(false);
    const engine = new BrowserEngine(async () => mod);
    await engine.createTracker(); // populates capabilities.tracker = false
    const host = document.body.appendChild(document.createElement('div'));

    const fakeMedia = {
      mode: signal<MediaMode>('webcam'),
      videoUrl: signal<string | null>(null),
      videoReady: signal(false),
      seeking: signal(false),
      camActive: signal(true),
      camStarting: signal(false),
      camPaused: signal(false),
      camErr: signal<string | null>(null),
      fps: signal(24),
      latencyMs: signal(12),
      toggleCamPause: vi.fn(),
      stopWebcam: vi.fn(),
    } as unknown as MediaSession;

    let ctrl!: ReturnType<typeof useTracking>;
    function Harness() {
      ctrl = useTracking({
        engine,
        browserOff: () => null,
        mediaMode: () => 'webcam',
        mediaKey: () => 'webcam:1',
        modelId: () => 'm1',
        classes: () => ['circle'],
        pred: () => null,
        conf: () => 0.25,
        iou: () => 0.45,
        maxDet: () => 300,
        o2o: () => false,
      });
      return <MediaBar media={fakeMedia} predicting={false} tracking={ctrl} />;
    }

    const view = render(() => <Harness />, host);
    flushSync();

    const toggle = host.querySelector<HTMLInputElement>('[data-track-toggle] input[role="switch"]')!;
    expect(toggle.disabled).toBe(true);
    expect(ctrl.disabledReason.get()).toContain('FoloTracker');
    const descId = toggle.getAttribute('aria-describedby');
    expect(descId).toBeTruthy();
    expect(host.querySelector(`#${descId}`)?.textContent).toContain('FoloTracker');

    view.unmount();
    host.remove();
  });
});

describe('ResultsPanel, predictionJson, and PlaygroundTiming integration', () => {
  it('renders #track_id in ResultsPanel, preserves track_id in JSON, and shows folo2 engine info badge', () => {
    const p = predFrame(20);
    const trackedBox = { ...p.boxes[0]!, track_id: 7 };
    const json = predictionJson(p, true, [trackedBox], { conf: 0.25, iou: 0.45, maxDet: 300, o2o: false });
    expect(json?.boxes[0]).toMatchObject({ track_id: 7, class: 0, name: 'circle' });

    const host = document.body.appendChild(document.createElement('div'));
    const info: DetectorInfo = {
      backend: 'webgpu',
      device: 'gpu:0',
      adapter: 'RTX 4080',
      fallback: null,
      arch: 'folo2-n',
      imgsz: 640,
      classes: ['circle'],
      family: 'folo2',
      headMode: 'o2m',
      hasIouHead: true,
    };
    const view = render(
      () => (
        <div>
          <ResultsPanel
            result={{ boxes: [trackedBox], ms: 0.1 }}
            rows={[{ b: trackedBox, i: 0 }]}
            predicting={false}
            notice={null}
            json={json}
            pred={p}
            hidden={new Set()}
            tabRow={0}
            activeIdx={null}
            selectedIdx={null}
            onrowkey={() => {}}
          />
          <PlaygroundTiming
            compare={false}
            narrow={false}
            modelA={null}
            modelB={null}
            predA={p}
            predB={null}
            predictingA={false}
            predictingB={false}
            resultMsA={0.1}
            adapter="RTX 4080"
            engineInfo={info}
          />
        </div>
      ),
      host,
    );
    flushSync();

    expect(host.querySelector('.track-id')?.textContent).toBe('#7');
    expect(host.querySelector('tbody tr')?.getAttribute('aria-label')).toContain('track #7');
    expect(host.querySelector('[data-browser-engine-info]')?.textContent).toContain('folo2-n · FOLO v2 · o2m · IoU head');

    view.unmount();
    host.remove();
  });
});
