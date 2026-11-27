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
    const ctxStub = { drawImage: vi.fn() } as unknown as CanvasRenderingContext2D;
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
});
