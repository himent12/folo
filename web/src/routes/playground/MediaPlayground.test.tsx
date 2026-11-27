// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { session } from './session';

let target: HTMLElement;
let handle: { unmount(): void } | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

function mountPlayground(path = '#/playground?model=m_hard_s') {
  const api = new MockApi('normal', 0);
  api.imageUrl = async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
  const predictSpy = vi.spyOn(api, 'predict');
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  history.replaceState(history.state, '', path);
  router.route = parse(path);
  handle = render(() => <Playground />, target);
  return { api, predictSpy };
}

afterEach(() => {
  handle?.unmount();
  handle = undefined;
  target?.remove();
  session.set(null);
  vi.restoreAllMocks();
});

describe('Playground: Local Video Frame Scrubber & Live Webcam', () => {
  it('shows a graceful inline error when camera API is unavailable or permission is denied', async () => {
    const origMediaDevices = navigator.mediaDevices;
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: vi.fn(async () => {
          throw Object.assign(new Error('Permission denied'), { name: 'NotAllowedError' });
        }),
      },
    });
    try {
      mountPlayground('#/playground?model=m_hard_s');
      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'Live camera button',
      );
      camBtn.click();
      flushSync();

      const alert = await until(() => target.querySelector('[data-live-media="webcam"] [role="alert"]'), 'inline camera error');
      expect(alert.textContent).toContain('Camera permission was denied');

      // Dismiss returns cleanly to Playground DropZone.
      const backBtn = [...alert.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Back to Playground'))!;
      backBtn.click();
      flushSync();
      await until(() => target.querySelector('button.sample'), 'DropZone restored');
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
    }
  });

  it('streams live webcam detections and releases MediaStream tracks immediately on Stop and unmount', async () => {
    const stopTrack = vi.fn();
    const fakeStream = {
      getTracks: () => [{ stop: stopTrack, addEventListener: vi.fn(), removeEventListener: vi.fn() }],
    } as unknown as MediaStream;
    const origMediaDevices = navigator.mediaDevices;
    const origSrcObject = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
    let assignedSrcObject: MediaProvider | null = null;
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      configurable: true,
      get: () => assignedSrcObject,
      set: (v: MediaProvider | null) => {
        assignedSrcObject = v;
      },
    });
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => fakeStream) },
    });
    const playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    try {
      mountPlayground('#/playground?model=m_hard_s');
      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'Live camera button',
      );
      camBtn.click();
      flushSync();

      const hud = await until(() => target.querySelector('[data-live-media="webcam"] .live-hud'), 'live webcam HUD');
      expect(hud.textContent).toContain('LIVE');
      expect(playSpy).toHaveBeenCalled();

      // Pause updates status badge to PAUSED.
      const pauseBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-media-bar="webcam"] button')].find((b) => b.textContent?.trim() === 'Pause')!;
      pauseBtn.click();
      flushSync();
      expect(hud.textContent).toContain('PAUSED');
      expect(pauseSpy).toHaveBeenCalled();

      // Stop camera immediately stops all MediaStream tracks.
      const stopBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-media-bar="webcam"] button')].find((b) => b.textContent?.includes('Stop camera'))!;
      stopBtn.click();
      flushSync();
      expect(stopTrack).toHaveBeenCalledTimes(1);
      expect(assignedSrcObject).toBeNull();
      expect(target.querySelector('[data-media-bar="webcam"]')).toBeNull();
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
      if (origSrcObject) Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', origSrcObject);
    }
  });

  it('stops a pending MediaStream if Playground unmounts before getUserMedia resolves', async () => {
    const stopTrack = vi.fn();
    let resolveStream!: (s: MediaStream) => void;
    const pending = new Promise<MediaStream>((r) => (resolveStream = r));
    const origMediaDevices = navigator.mediaDevices;
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(() => pending) },
    });
    try {
      mountPlayground('#/playground?model=m_hard_s');
      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'Live camera button',
      );
      camBtn.click();
      flushSync();
      await until(() => target.querySelector('[data-media-bar="webcam"]'), 'webcam bar');

      // Navigate away / unmount before getUserMedia resolves.
      handle!.unmount();
      handle = undefined;

      resolveStream({ getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream);
      await new Promise((r) => setTimeout(r, 10));
      expect(stopTrack).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
    }
  });

  it('loads a local video file into the inline frame scrubber and runs inference on stepped frames', async () => {
    const origCreateUrl = URL.createObjectURL;
    const origRevokeUrl = URL.revokeObjectURL;
    const revokeSpy = vi.fn();
    let urlSeq = 0;
    URL.createObjectURL = vi.fn(() => `blob:mock-${++urlSeq}`);
    URL.revokeObjectURL = revokeSpy;
    const png1x1 = Uint8Array.from(
      atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='),
      (ch) => ch.charCodeAt(0),
    );
    const ctxStub = {
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      restore: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
      cb(new Blob([png1x1], { type: 'image/png' }));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    try {
      const { api } = mountPlayground('#/playground?model=m_hard_s');
      const predictSpy = vi.spyOn(api, 'predict').mockImplementation(async (_input, opts) => ({
        model_id: opts.model_id,
        image: { width: 640, height: 360 },
        boxes: [{ class: 0, name: 'circle', score: 0.91, x1: 20, y1: 20, x2: 120, y2: 120 }],
        timing: { preprocess_ms: 1, inference_ms: 8, postprocess_ms: 1, total_ms: 10 },
        device: 'gpu:0',
      }));
      await until(() => target.querySelector('input[accept*="video/mp4"]'), 'video file input');

      // Pick a local .mp4 file via the hidden video input.
      const videoInputs = [...target.querySelectorAll<HTMLInputElement>('input[type="file"]')];
      const videoInput = videoInputs.find((i) => i.accept.startsWith('video/mp4'))!;
      const clip = new File(['video-bytes'], 'warehouse.mp4', { type: 'video/mp4' });
      Object.defineProperty(videoInput, 'files', { configurable: true, value: [clip] });
      videoInput.dispatchEvent(new Event('change', { bubbles: true }));
      flushSync();

      const bar = await until(() => target.querySelector('[data-media-bar="video"]'), 'inline video scrubber bar');
      const decoder = target.querySelector<HTMLVideoElement>('video.media-decoder')!;
      Object.defineProperty(decoder, 'videoWidth', { configurable: true, value: 640 });
      Object.defineProperty(decoder, 'videoHeight', { configurable: true, value: 360 });
      Object.defineProperty(decoder, 'duration', { configurable: true, value: 10 });
      Object.defineProperty(decoder, 'readyState', { configurable: true, value: 4 });

      // Fire loadedmetadata + loadeddata -> frame 0 is extracted and sent to inference.
      decoder.dispatchEvent(new Event('loadedmetadata'));
      decoder.dispatchEvent(new Event('loadeddata'));
      await until(() => predictSpy.mock.calls.length === 1, 'initial video frame prediction');
      expect(bar.textContent).toContain('0:00.0 / 0:10.0');

      // Step forward +1 frame advances currentTime and triggers inference on seeked.
      const stepFwd = bar.querySelector<HTMLButtonElement>('button[aria-label="Step forward 1 frame"]')!;
      stepFwd.click();
      flushSync();
      decoder.dispatchEvent(new Event('seeked'));
      await until(() => predictSpy.mock.calls.length === 2, 'stepped video frame prediction');
      expect(decoder.currentTime).toBeGreaterThan(0);

      // Export buttons become active once the video frame prediction settles.
      await until(() => target.querySelector('button[aria-label="Export annotated PNG"]'), 'Export PNG button');
      expect(target.querySelector('button[aria-label="Export detections JSON"]')).not.toBeNull();

      // Clearing revokes object URLs cleanly.
      const clearBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Clear image"]')!;
      clearBtn.click();
      flushSync();
      expect(target.querySelector('[data-media-bar="video"]')).toBeNull();
      expect(revokeSpy).toHaveBeenCalled();
    } finally {
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
  });

  it('surfaces video.play() errors instead of swallowing them and isolates concurrent webcam restarts', async () => {
    const stopFirst = vi.fn();
    const stopSecond = vi.fn();
    const stream1 = { getTracks: () => [{ stop: stopFirst, addEventListener: vi.fn(), removeEventListener: vi.fn() }] } as unknown as MediaStream;
    const stream2 = { getTracks: () => [{ stop: stopSecond, addEventListener: vi.fn(), removeEventListener: vi.fn() }] } as unknown as MediaStream;
    const origMediaDevices = navigator.mediaDevices;
    const origSrcObject = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
    let assigned: MediaProvider | null = null;
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      configurable: true,
      get: () => assigned,
      set: (v: MediaProvider | null) => {
        assigned = v;
      },
    });
    let callIdx = 0;
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => (++callIdx === 1 ? stream1 : stream2)) },
    });
    let rejectPlay1!: (e: Error) => void;
    let playCall = 0;
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() => {
      playCall++;
      if (playCall === 1) return new Promise<void>((_, rej) => (rejectPlay1 = rej));
      return Promise.reject(new Error('Autoplay blocked'));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});

    try {
      mountPlayground('#/playground?model=m_hard_s');
      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'Live camera button',
      );
      camBtn.click();
      flushSync();
      await until(() => playCall === 1, 'first play pending');

      // Click Camera segment again to restart webcam while play #1 is still pending.
      const camRadio = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.includes('Camera') || b.getAttribute('aria-label') === 'Camera')!;
      camRadio.click();
      flushSync();

      // Play #2 rejects -> must surface inline error (never swallowed!).
      const alert = await until(() => target.querySelector('[data-live-media="webcam"] [role="alert"]'), 'play error surfaced');
      expect(alert.textContent).toContain('Autoplay blocked');
      expect(stopSecond).toHaveBeenCalledTimes(1);

      // Now reject stale play #1 -> must not crash or touch stream 2.
      rejectPlay1(new Error('Stale play failure'));
      await new Promise((r) => setTimeout(r, 10));
      expect(stopFirst).toHaveBeenCalled();
      expect(stopSecond).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
      if (origSrcObject) Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', origSrcObject);
    }
  });

  it('gates exports while prediction is in flight and lets Auto-off video scrub visually without hijacking Space', async () => {
    const origCreateUrl = URL.createObjectURL;
    const origRevokeUrl = URL.revokeObjectURL;
    let urlSeq = 0;
    URL.createObjectURL = vi.fn(() => `blob:mock-auto-${++urlSeq}`);
    URL.revokeObjectURL = vi.fn();
    const ctxStub = {
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      restore: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
      cb(new Blob(['img'], { type: 'image/jpeg' }));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    try {
      const { api } = mountPlayground('#/playground?model=m_hard_s');
      let resolvePred!: (p: Awaited<ReturnType<typeof api.predict>>) => void;
      const predictSpy = vi.spyOn(api, 'predict').mockImplementation(
        () => new Promise((r) => (resolvePred = r)),
      );
      await until(() => target.querySelector('input[accept*="video/mp4"]'), 'video file input');

      const videoInput = [...target.querySelectorAll<HTMLInputElement>('input[type="file"]')].find((i) => i.accept.startsWith('video/mp4'))!;
      const clip = new File(['video-bytes'], 'scrub.mp4', { type: 'video/mp4' });
      Object.defineProperty(videoInput, 'files', { configurable: true, value: [clip] });
      videoInput.dispatchEvent(new Event('change', { bubbles: true }));
      flushSync();

      const bar = await until(() => target.querySelector('[data-media-bar="video"]'), 'video bar');
      const decoder = target.querySelector<HTMLVideoElement>('video.media-decoder')!;
      Object.defineProperty(decoder, 'videoWidth', { configurable: true, value: 320 });
      Object.defineProperty(decoder, 'videoHeight', { configurable: true, value: 180 });
      Object.defineProperty(decoder, 'duration', { configurable: true, value: 5 });
      Object.defineProperty(decoder, 'readyState', { configurable: true, value: 4 });
      decoder.dispatchEvent(new Event('loadedmetadata'));
      decoder.dispatchEvent(new Event('loadeddata'));
      await until(() => predictSpy.mock.calls.length === 1, 'initial predict in flight');

      // While prediction is in flight, Export buttons stay disabled/hidden so stale exports cannot fire.
      expect(target.querySelector('button[aria-label="Export annotated PNG"]')).toBeNull();

      resolvePred({
        model_id: 'm_hard_s',
        image: { width: 320, height: 180 },
        boxes: [{ class: 0, name: 'circle', score: 0.9, x1: 10, y1: 10, x2: 50, y2: 50 }],
        timing: { preprocess_ms: 1, inference_ms: 4, postprocess_ms: 1, total_ms: 6 },
        device: 'gpu:0',
      });
      await until(() => target.querySelector('button[aria-label="Export annotated PNG"]'), 'export enabled after settle');

      // Turn Auto OFF via switch (Space key on switch must NOT be hijacked by video play/pause).
      const autoSwitch = bar.querySelector<HTMLInputElement>('input[role="switch"]')!;
      const spaceEv = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
      autoSwitch.dispatchEvent(spaceEv);
      expect(spaceEv.defaultPrevented).toBe(false);
      autoSwitch.click();
      flushSync();
      expect(autoSwitch.checked).toBe(false);

      // Step +1 frame with Auto OFF -> renders LiveMediaCanvas in PREVIEW mode without calling predict!
      bar.querySelector<HTMLButtonElement>('button[aria-label="Step forward 1 frame"]')!.click();
      flushSync();
      decoder.dispatchEvent(new Event('seeked'));
      flushSync();
      const hud = await until(() => target.querySelector('[data-live-media="video"] .live-hud'), 'video PREVIEW HUD');
      expect(hud.textContent).toContain('PREVIEW');
      expect(predictSpy).toHaveBeenCalledTimes(1);
    } finally {
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
  });

  it('auto-pauses live webcam on inference failure without retrying and starts streaming once models finish loading', async () => {
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
    URL.createObjectURL = vi.fn(() => 'blob:webcam-fail-test');
    URL.revokeObjectURL = vi.fn();
    const ctxStub = {
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      restore: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
      cb(new Blob(['jpg'], { type: 'image/jpeg' }));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});

    try {
      const api = new MockApi('normal', 0);
      const origModels = api.models.bind(api);
      let releaseModels!: (list: Awaited<ReturnType<typeof api.models>>) => void;
      api.models = () => new Promise((r) => (releaseModels = r));
      const predictSpy = vi.spyOn(api, 'predict').mockRejectedValue(new Error('server offline'));
      hub.api = api;
      hub.mode = 'mock';
      target = document.createElement('div');
      document.body.appendChild(target);
      history.replaceState(history.state, '', '#/playground?model=m_hard_s');
      router.route = parse('#/playground?model=m_hard_s');
      handle = render(() => <Playground />, target);

      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'Live camera button before models resolve',
      );
      camBtn.click();
      flushSync();
      const decoder = target.querySelector<HTMLVideoElement>('video.media-decoder')!;
      Object.defineProperty(decoder, 'videoWidth', { configurable: true, value: 320 });
      Object.defineProperty(decoder, 'videoHeight', { configurable: true, value: 240 });
      Object.defineProperty(decoder, 'readyState', { configurable: true, value: 4 });
      await until(() => target.querySelector('[data-live-media="webcam"] .live-hud'), 'LIVE HUD while models load');
      expect(predictSpy).toHaveBeenCalledTimes(0);

      // Model catalog resolves -> reactive effect automatically starts webcam detection without needing Pause/Resume!
      releaseModels(await origModels());
      await until(() => predictSpy.mock.calls.length === 1, 'first webcam prediction fires after models load');

      // Prediction failed -> live webcam auto-pauses and stops firing further requests.
      const hud = await until(() => {
        const el = target.querySelector('[data-live-media="webcam"] .live-hud');
        return el?.textContent?.includes('PAUSED') ? el : null;
      }, 'webcam auto-paused on inference failure');
      expect(hud.textContent).toContain('PAUSED');
      await new Promise((r) => setTimeout(r, 120));
      expect(predictSpy).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
      if (origSrcObject) Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', origSrcObject);
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
  });

  it('probes 10 fps WebM frame timing so +1f/-1f step across adjacent frames and display accurate frame counts', async () => {
    const origCreateUrl = URL.createObjectURL;
    const origRevokeUrl = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn(() => 'blob:mock-10fps');
    URL.revokeObjectURL = vi.fn();
    const ctxStub = {
      drawImage: vi.fn(),
      save: vi.fn(),
      beginPath: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      restore: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
      cb(new Blob(['jpg'], { type: 'image/jpeg' }));
    });
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    // Minimal valid EBML/WebM container with 10 fps video SimpleBlocks (0, 100, 200 ms) and no DefaultDuration.
    const webmBytes = new Uint8Array([
      0x1a, 0x45, 0xdf, 0xa3, 0x84, 0x42, 0x86, 0x81, 0x01, // EBML header
      0x18, 0x53, 0x80, 0x67, 0xff, // Segment (unknown size)
      0x15, 0x49, 0xa9, 0x66, 0x87, 0x2a, 0xd7, 0xb1, 0x83, 0x0f, 0x42, 0x40, // Info: TimestampScale = 1_000_000 ns
      0x16, 0x54, 0xae, 0x6b, 0x88, 0xae, 0x86, 0xd7, 0x81, 0x01, 0x83, 0x81, 0x01, // Tracks: track 1 video
      0x1f, 0x43, 0xb6, 0x75, 0x99, // Cluster (25 bytes)
      0xe7, 0x81, 0x00, // Cluster Timestamp = 0
      0xa3, 0x85, 0x81, 0x00, 0x00, 0x80, 0x00, // SimpleBlock track 1, t = 0 ms
      0xa3, 0x85, 0x81, 0x00, 0x64, 0x80, 0x00, // SimpleBlock track 1, t = 100 ms
      0xa3, 0x85, 0x81, 0x00, 0xc8, 0x80, 0x00, // SimpleBlock track 1, t = 200 ms
    ]);

    try {
      const { api } = mountPlayground('#/playground?model=m_hard_s');
      vi.spyOn(api, 'predict').mockResolvedValue({
        model_id: 'm_hard_s',
        image: { width: 320, height: 180 },
        boxes: [],
        timing: { preprocess_ms: 1, inference_ms: 4, postprocess_ms: 1, total_ms: 6 },
        device: 'cpu',
      });
      await until(() => target.querySelector('input[accept*="video/mp4"]'), 'video file input');

      const videoInput = [...target.querySelectorAll<HTMLInputElement>('input[type="file"]')].find((i) => i.accept.startsWith('video/mp4'))!;
      const clip = new File([webmBytes], 'ten-fps.webm', { type: 'video/webm' });
      Object.defineProperty(videoInput, 'files', { configurable: true, value: [clip] });
      videoInput.dispatchEvent(new Event('change', { bubbles: true }));
      flushSync();

      const bar = await until(() => target.querySelector('[data-media-bar="video"]'), 'video bar');
      const decoder = target.querySelector<HTMLVideoElement>('video.media-decoder')!;
      Object.defineProperty(decoder, 'videoWidth', { configurable: true, value: 320 });
      Object.defineProperty(decoder, 'videoHeight', { configurable: true, value: 180 });
      Object.defineProperty(decoder, 'duration', { configurable: true, value: 1.1 });
      Object.defineProperty(decoder, 'readyState', { configurable: true, value: 4 });
      decoder.dispatchEvent(new Event('loadedmetadata'));
      decoder.dispatchEvent(new Event('loadeddata'));

      // Async container probe sets 10 fps -> 1.1 s clip reports f 1 / 11 (never f 1 / 33).
      await until(() => bar.textContent?.includes('f 1 / 11'), '10 fps frame count f 1 / 11');

      // Turn Auto off, then step +1f -> jumps directly past 100 ms (frame 2) on the very first press.
      bar.querySelector<HTMLInputElement>('input[role="switch"]')!.click();
      flushSync();
      bar.querySelector<HTMLButtonElement>('button[aria-label="Step forward 1 frame"]')!.click();
      flushSync();
      expect(decoder.currentTime).toBeCloseTo(0.102, 3);
      decoder.dispatchEvent(new Event('seeked'));
      flushSync();
      expect(bar.textContent).toContain('f 2 / 11');

      // Step -1f returns cleanly to frame 1 (0 s).
      bar.querySelector<HTMLButtonElement>('button[aria-label="Step back 1 frame"]')!.click();
      flushSync();
      expect(decoder.currentTime).toBe(0);
      decoder.dispatchEvent(new Event('seeked'));
      flushSync();
      expect(bar.textContent).toContain('f 1 / 11');

      // Playtest R20 P3-3: focusing +1f or Detect frame and activating restores focus once transient disabled clears.
      const stepFwdBtn = bar.querySelector<HTMLButtonElement>('button[aria-label="Step forward 1 frame"]')!;
      stepFwdBtn.focus();
      expect(document.activeElement).toBe(stepFwdBtn);
      stepFwdBtn.click();
      flushSync();
      // Simulate browser blur-to-body when button becomes disabled while seeking
      stepFwdBtn.blur();
      decoder.dispatchEvent(new Event('seeked'));
      flushSync();
      await until(() => document.activeElement === stepFwdBtn, 'focus restored to +1f button after seek');

      let resolveDetect!: (p: { model_id: string; image: { width: number; height: number }; boxes: []; timing: { preprocess_ms: number; inference_ms: number; postprocess_ms: number; total_ms: number }; device: string }) => void;
      vi.spyOn(api, 'predict').mockImplementation(
        () =>
          new Promise((r) => {
            resolveDetect = r;
          }),
      );
      const detectBtn = bar.querySelector<HTMLButtonElement>('button[data-video-action="detect"]')!;
      detectBtn.focus();
      detectBtn.click();
      flushSync();
      detectBtn.blur();
      await until(() => typeof resolveDetect === 'function', 'detect prediction in flight');
      resolveDetect({
        model_id: 'm_hard_s',
        image: { width: 320, height: 180 },
        boxes: [],
        timing: { preprocess_ms: 1, inference_ms: 4, postprocess_ms: 1, total_ms: 6 },
        device: 'cpu',
      });
      await until(() => document.activeElement === detectBtn, 'focus restored to Detect frame button after predict');
    } finally {
      URL.createObjectURL = origCreateUrl;
      URL.revokeObjectURL = origRevokeUrl;
    }
  });
});
