// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { session } from './session';
import ZoomStage, { clampPan, zoomToward } from './ZoomStage';

let target: HTMLElement;
let mounted: { unmount(): void } | undefined;

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

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  session.set(null);
  vi.restoreAllMocks();
});

describe('ZoomStage', () => {
  it('clamps pan offsets and anchors zoom around cursor', () => {
    expect(clampPan(50, 400, 1)).toBe(0);
    expect(clampPan(500, 400, 2)).toBe(200);
    expect(clampPan(-500, 400, 2)).toBe(-200);

    const z2 = zoomToward({ z: 1, tx: 0, ty: 0 }, 2, 300, 150, 400, 300);
    expect(z2.z).toBe(2);
    expect(z2.tx).toBe(-100);
    expect(z2.ty).toBe(0);

    const back = zoomToward(z2, 0.5, 200, 150, 400, 300);
    expect(back).toEqual({ z: 1, tx: 0, ty: 0 });
  });

  it('zooms in, pans by drag without triggering child clicks, resets on double-click, and resets on source change', () => {
    const key = signal<string | null>('img-1');
    const clicked = vi.fn();
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <ZoomStage resetKey={key.get()}>
          <div class="inner-box" onclick={clicked}>
            Box
          </div>
        </ZoomStage>
      ),
      target,
    );

    const stage = target.querySelector<HTMLElement>('.zoom-stage')!;
    const canvas = target.querySelector<HTMLElement>('.zoom-canvas')!;
    const box = target.querySelector<HTMLElement>('.inner-box')!;
    vi.spyOn(stage, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 400,
      height: 300,
      right: 400,
      bottom: 300,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    const zoomIn = stage.querySelector<HTMLButtonElement>('button[aria-label="Zoom in"]')!;
    const resetBtn = stage.querySelector<HTMLButtonElement>('button[aria-label="Reset zoom"]')!;
    const pctBtn = stage.querySelector<HTMLButtonElement>('.zoom-pct')!;
    expect(pctBtn.textContent?.trim()).toBe('Fit');
    expect(resetBtn.disabled).toBe(true);

    // Zoom in via toolbar button
    zoomIn.click();
    flushSync();
    expect(pctBtn.textContent?.trim()).toBe('150%');
    expect(resetBtn.disabled).toBe(false);
    expect(stage.classList.contains('zoomed')).toBe(true);
    expect(canvas.style.transform).toContain('scale(1.5)');

    // Plain click on child still fires while zoomed
    box.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 1, clientX: 100, clientY: 100 }));
    box.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 1, clientX: 101, clientY: 100 }));
    box.click();
    expect(clicked).toHaveBeenCalledTimes(1);

    // Drag beyond CLICK_SLOP pans and suppresses the trailing click
    box.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 2, clientX: 100, clientY: 100 }));
    box.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, button: 0, buttons: 1, pointerId: 2, clientX: 140, clientY: 120 }));
    flushSync();
    expect(stage.classList.contains('panning')).toBe(true);
    expect(canvas.style.transform).toContain('translate(40px, 20px)');
    box.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 2, clientX: 140, clientY: 120 }));
    box.click();
    flushSync();
    expect(clicked).toHaveBeenCalledTimes(1);

    // Releasing pointer outside the stage never leaves hover-panning stuck
    stage.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 3, clientX: 100, clientY: 100 }));
    window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 3, clientX: 500, clientY: 500 }));
    stage.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, button: 0, buttons: 0, pointerId: 3, clientX: 160, clientY: 160 }));
    flushSync();
    expect(stage.classList.contains('panning')).toBe(false);

    // Double-click resets zoom and pan back to Fit while keeping clip active until transitionend
    stage.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, clientX: 200, clientY: 150 }));
    flushSync();
    expect(pctBtn.textContent?.trim()).toBe('Fit');
    expect(stage.classList.contains('zoomed')).toBe(false);
    expect(stage.classList.contains('clip')).toBe(true);
    expect(canvas.style.transform).toBe('');
    const te = new Event('transitionend', { bubbles: true }) as TransitionEvent;
    Object.defineProperty(te, 'propertyName', { value: 'transform' });
    canvas.dispatchEvent(te);
    flushSync();
    expect(stage.classList.contains('clip')).toBe(false);

    // Double-click at Fit zooms in to 200%, wheel zooms further, then source change resets back to Fit
    stage.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, clientX: 200, clientY: 150 }));
    flushSync();
    expect(pctBtn.textContent?.trim()).toBe('200%');
    stage.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: -120, clientX: 200, clientY: 150 }));
    flushSync();
    expect(Number.parseInt(pctBtn.textContent ?? '0', 10)).toBeGreaterThan(200);
    key.set('img-2');
    flushSync();
    expect(pctBtn.textContent?.trim()).toBe('Fit');
    expect(stage.classList.contains('zoomed')).toBe(false);
    expect(stage.classList.contains('clip')).toBe(false);
  });

  it('clamps stale pan offset before anchoring subsequent zoom after resize', () => {
    const next = zoomToward({ z: 2, tx: 400, ty: 0 }, 1.5, 200, 150, 400, 300);
    // Clamped curTx at (width=400, z=2) is 200, so center-anchor zoom scales 200 * 1.5 = 300 (<= limit 400).
    expect(next.z).toBe(3);
    expect(next.tx).toBe(300);
  });
});

describe('Playground: confidence summary and quick JSON copy', () => {
  async function openWithPredictions() {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    api.predict = async (_input, opts) => ({
      model_id: opts.model_id,
      image: { width: 640, height: 480 },
      boxes: [
        { class: 0, name: 'circle', score: 0.92, x1: 10, y1: 10, x2: 80, y2: 80 },
        { class: 0, name: 'circle', score: 0.64, x1: 100, y1: 20, x2: 180, y2: 90 },
        { class: 1, name: 'square', score: 0.35, x1: 200, y1: 30, x2: 260, y2: 120 },
      ],
      timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
      device: 'gpu:0',
    });
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    mounted = render(() => <Playground />, target);
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample image')).click();
    await until(() => target.querySelectorAll('#pg-panel tbody tr').length === 3, 'three detections');
  }

  it('shows live confidence distribution badge and copies detections JSON in one click', async () => {
    const writeText = vi.fn(async (_text: string) => {});
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    await openWithPredictions();

    // Live confidence summary badge reflects all 3 detections
    const summary = target.querySelector('.conf-summary')!;
    expect(summary).not.toBeNull();
    expect(summary.textContent).toContain('3 detections');
    expect(summary.textContent).toContain('0.64');
    expect(summary.textContent).toContain('0.35–0.92');

    // Zoom stage HUD is mounted around the image canvas and preserves selected row on zoom click
    expect(target.querySelector('.zoom-stage .zoom-hud')).not.toBeNull();
    const firstRow = target.querySelector<HTMLTableRowElement>('#pg-panel tbody tr')!;
    firstRow.click();
    flushSync();
    expect(firstRow.getAttribute('aria-selected')).toBe('true');
    const zoomIn = target.querySelector<HTMLButtonElement>('.zoom-hud button[aria-label="Zoom in"]')!;
    zoomIn.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0 }));
    zoomIn.click();
    flushSync();
    expect(firstRow.getAttribute('aria-selected')).toBe('true');

    // One-click Copy detections JSON works directly from the Detections tab (square icon button in rail)
    const copyBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Copy detections JSON"]')!;
    expect(copyBtn).not.toBeNull();
    expect(copyBtn.classList.contains('square')).toBe(true);
    expect(copyBtn.querySelector('.label')).toBeNull();
    copyBtn.click();
    await vi.waitFor(() => {
      flushSync();
      expect(writeText).toHaveBeenCalledTimes(1);
    });
    const copiedText = writeText.mock.calls[0]?.[0] as string;
    const parsed = JSON.parse(copiedText) as { model_id: string; boxes: { score: number }[] };
    expect(parsed.model_id).toBe('m_shapes_s');
    expect(parsed.boxes.map((b) => b.score)).toEqual([0.92, 0.64, 0.35]);

    const status = await until(
      () => [...target.querySelectorAll('[role="status"]')].find((el) => el.textContent?.includes('Copied 3 detections as JSON')),
      'aria-live copy status',
    );
    expect(status.getAttribute('aria-live')).toBe('polite');
    expect(target.querySelector('button[aria-label="Copied detections JSON"]')).not.toBeNull();
  });

  it('announces clipboard failure cleanly without crashing', async () => {
    const writeText = vi.fn(async () => {
      throw new Error('blocked');
    });
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    await openWithPredictions();
    const copyBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Copy detections JSON"]')!;
    copyBtn.click();

    await until(
      () => [...target.querySelectorAll('[role="status"]')].find((el) => el.textContent?.includes('Could not copy detections JSON')),
      'aria-live failure status',
    );
  });

  it('excludes class-filtered detections from both the JSON tab and Copy JSON', async () => {
    const writeText = vi.fn(async (_text: string) => {});
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    await openWithPredictions();

    // Open Classes tab (rail layout) and hide `circle` (2 of the 3 boxes).
    const classesTab = target.querySelector<HTMLButtonElement>('#pg-classes')!;
    expect(classesTab).not.toBeNull();
    classesTab.click();
    flushSync();

    const circleToggle = [...target.querySelectorAll<HTMLButtonElement>('#pg-panel button')].find((b) => b.textContent?.includes('circle'))!;
    expect(circleToggle).not.toBeNull();
    circleToggle.click();
    flushSync();

    // Switch to JSON tab and verify only the visible `square` detection is rendered, with filter.hidden_classes.
    const jsonTab = target.querySelector<HTMLButtonElement>('#pg-json')!;
    jsonTab.click();
    flushSync();
    const jsonTextContent = target.querySelector('#pg-panel pre, #pg-panel code')?.textContent ?? '';
    expect(jsonTextContent).toContain('"hidden_classes"');
    expect(jsonTextContent).toContain('"square"');
    expect(jsonTextContent).not.toContain('"name": "circle"');

    // Copy JSON exports the same filtered payload and announces the visible count.
    const copyBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Copy detections JSON"]')!;
    copyBtn.click();
    await vi.waitFor(() => {
      flushSync();
      expect(writeText).toHaveBeenCalledTimes(1);
    });
    const parsed = JSON.parse(writeText.mock.calls[0]?.[0] as string) as {
      boxes: { name: string }[];
      filter: { hidden_classes?: string[] };
    };
    expect(parsed.boxes.map((b) => b.name)).toEqual(['square']);
    expect(parsed.filter.hidden_classes).toEqual(['circle']);

    await until(
      () => [...target.querySelectorAll('[role="status"]')].find((el) => el.textContent?.includes('Copied 1 detection as JSON')),
      'filtered aria-live copy status',
    );
  });

  it('shows Image unavailable with Retry when imageUrl fails or img errors, hides boxes, and recovers on Retry', async () => {
    const api = new MockApi('normal', 0);
    let urlAttempt = 0;
    api.imageUrl = async (_ds, _img, thumb) => {
      if (thumb !== undefined) return 'data:,';
      urlAttempt++;
      if (urlAttempt === 1) throw new Error('404 not found');
      return 'https://example.test/img.jpg';
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    mounted = render(() => <Playground />, target);

    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample button')).click();
    const errOverlay = await until(() => target.querySelector<HTMLElement>('.image-failed'), 'image unavailable overlay');
    expect(errOverlay.textContent).toContain('Image unavailable');
    expect(target.querySelectorAll('[data-box]').length).toBe(0);
    // ZoomStage and its focusable HUD buttons must not remain mounted behind the opaque error overlay.
    expect(target.querySelector('.zoom-stage, .zoom-hud')).toBeNull();

    // Retry succeeds and mounts the image with a cache-busting retry query param.
    errOverlay.querySelector<HTMLButtonElement>('button')!.click();
    const img = await until(() => target.querySelector<HTMLImageElement>('.stage img'), 'recovered img');
    expect(img.getAttribute('src')).toBe('https://example.test/img.jpg?retry=1');
    expect(target.querySelector('.image-failed')).toBeNull();

    // If the browser later fails decoding/loading the image, Image unavailable reappears and Retry recovers again.
    img.dispatchEvent(new Event('error'));
    const errOverlay2 = await until(() => target.querySelector<HTMLElement>('.image-failed'), 'img error overlay');
    errOverlay2.querySelector<HTMLButtonElement>('button')!.click();
    const img2 = await until(() => target.querySelector<HTMLImageElement>('.stage img'), 'second retry img');
    expect(img2.getAttribute('src')).toBe('https://example.test/img.jpg?retry=2');
  });

  it('preserves a deep-linked dataset image in the URL when a vanished model falls back before findImage settles', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    const dss = await api.datasets();
    const firstDs = dss[0]!;
    const page = await api.datasetImages(firstDs.id, { split: 'val', limit: 1 });
    const firstImg = page.images[0]!;
    let releaseDs!: () => void;
    const gate = new Promise<void>((r) => (releaseDs = r));
    const origDataset = api.dataset.bind(api);
    api.dataset = async (id: string) => {
      await gate;
      return origDataset(id);
    };
    hub.api = api;
    hub.mode = 'mock';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse(`#/playground?model=m_vanished&dataset=${firstDs.id}&image=${firstImg.id}&split=val&at=0`);
    mounted = render(() => <Playground />, target);

    // Wait for followVanished to rewrite the vanished model to a real model while findImage is still blocked.
    await until(() => {
      const m = router.route.query.get('model');
      return m && m !== 'm_vanished' ? m : null;
    }, 'vanished model fallback');
    expect(router.route.query.get('dataset')).toBe(firstDs.id);
    expect(router.route.query.get('image')).toBe(firstImg.id);

    releaseDs();
    await until(() => target.querySelector('.stage img'), 'linked image loaded');

    // Now navigate from the loaded image A to a link for image B with a vanished model + custom conf while B is blocked:
    // model fallback must keep B and conf=0.6 instead of reverting the URL to A or dropping thresholds.
    const page2 = await api.datasetImages(firstDs.id, { split: 'val', limit: 2 });
    const secondImg = page2.images[1]!;
    let releaseSecond!: () => void;
    const gate2 = new Promise<void>((r) => (releaseSecond = r));
    api.dataset = async (id: string) => {
      await gate2;
      return origDataset(id);
    };
    router.go(`/playground?model=m_vanished_2&dataset=${firstDs.id}&image=${secondImg.id}&split=val&at=1&conf=0.6`, { replace: true });
    await until(() => {
      const m = router.route.query.get('model');
      return m && m !== 'm_vanished_2' ? m : null;
    }, 'second vanished model fallback');
    expect(router.route.query.get('dataset')).toBe(firstDs.id);
    expect(router.route.query.get('image')).toBe(secondImg.id);
    expect(router.route.query.get('conf')).toBe('0.6');
    releaseSecond();
    await until(() => target.querySelector('.stage img')?.getAttribute('alt')?.includes(secondImg.file), 'second linked image shown');
  });
});
