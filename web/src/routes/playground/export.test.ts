// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { pendingBlobRevokeCount, revokePendingBlobUrls } from '$lib/tableExport';
import { compareBadgeText, drawAnnotations, exportComparePng, exportDetectionsJson, exportFilename } from './export';
import { visibleTruthBoxes } from './exportActions';

afterEach(() => {
  revokePendingBlobUrls();
  vi.restoreAllMocks();
});

describe('playground/export', () => {
  it('normalizes safe png and json filenames', () => {
    expect(exportFilename('Sample Image (01) vs Model B', 'png')).toBe('sample-image-01-vs-model-b.png');
    expect(exportFilename('Sample Image (01)', 'json')).toBe('sample-image-01.json');
    expect(exportFilename('   ', 'png')).toBe('export.png');
  });

  it('formats compare PNG badge captions with count, unavailable state, and latency', () => {
    expect(compareBadgeText('A', 'shapes-s', 1, 14.2)).toBe('A · shapes-s · 1 det · 14.2 ms');
    expect(compareBadgeText('B', 'shapes-n', 4, null)).toBe('B · shapes-n · 4 dets');
    expect(compareBadgeText('B', 'shapes-n', null, null)).toBe('B · shapes-n · unavailable');
  });

  it('filters exported ground-truth boxes by each pane visibility set (hidden classes)', () => {
    const truth = [
      { class: 0, x1: 0, y1: 0, x2: 10, y2: 10 },
      { class: 1, x1: 20, y1: 20, x2: 30, y2: 30 },
      { class: 2, x1: 40, y1: 40, x2: 50, y2: 50 },
    ];
    // No hidden classes -> keep all
    expect(visibleTruthBoxes(truth, null).map((b) => b.class)).toEqual([0, 1, 2]);
    // Model A hides class 0 -> only 1, 2 kept
    expect(visibleTruthBoxes(truth, new Set([1, 2])).map((b) => b.class)).toEqual([1, 2]);
    // Model B with a different class order hides class 2 -> only 0, 1 kept
    expect(visibleTruthBoxes(truth, new Set([0, 1])).map((b) => b.class)).toEqual([0, 1]);
  });

  it('draws ground-truth boxes, predicted boxes, and label chips onto a 2D canvas context', () => {
    const calls: string[] = [];
    const ctx = {
      save: () => calls.push('save'),
      restore: () => calls.push('restore'),
      beginPath: () => calls.push('beginPath'),
      rect: () => calls.push('rect'),
      clip: () => calls.push('clip'),
      setLineDash: () => calls.push('setLineDash'),
      strokeRect: () => calls.push('strokeRect'),
      fillRect: () => calls.push('fillRect'),
      fillText: (text: string) => calls.push(`fillText:${text}`),
      measureText: (text: string) => ({ width: text.length * 7 }),
      lineWidth: 1,
      strokeStyle: '',
      fillStyle: '',
      font: '',
      textBaseline: 'alphabetic',
    } as unknown as CanvasRenderingContext2D;

    drawAnnotations(
      ctx,
      {
        boxes: [
          { class: 0, x1: 20, y1: 40, x2: 120, y2: 140, score: 0.91 },
          { class: 1, x1: 200, y1: 80, x2: 300, y2: 180, score: 0.77 },
        ],
        truth: [{ class: 0, x1: 18, y1: 38, x2: 118, y2: 138 }],
        classes: ['circle', 'square'],
        labels: true,
        scores: true,
      },
      { ox: 0, oy: 0, width: 640, height: 480 },
    );

    expect(calls).toContain('setLineDash');
    // 1 ground truth + 2 boxes * (halo + stroke) = 5 strokeRect calls.
    expect(calls.filter((c) => c === 'strokeRect').length).toBe(5);
    expect(calls).toContain('fillText:circle 0.91');
    expect(calls).toContain('fillText:square 0.77');
  });

  it('downloads structured JSON via Blob object URL and revokes it', () => {
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-json');
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    const ok = exportDetectionsJson('frame-01-shapes-s', { model_id: 'm_1', boxes: [] });
    expect(ok).toBe(true);
    expect(createUrl).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
    expect(pendingBlobRevokeCount()).toBe(1);
    revokePendingBlobUrls();
    expect(pendingBlobRevokeCount()).toBe(0);
    expect(revokeUrl).toHaveBeenCalledWith('blob:mock-json');
  });

  it('renders a side-by-side compare canvas with both panes, captions, and clean blob download', async () => {
    const drawImageCalls: Array<[number, number, number, number]> = [];
    const captions: string[] = [];
    let canvasW = 0;
    let canvasH = 0;
    const fakeCtx = {
      save: () => {},
      restore: () => {},
      beginPath: () => {},
      rect: () => {},
      clip: () => {},
      setLineDash: () => {},
      strokeRect: () => {},
      fillRect: () => {},
      drawImage: (_img: unknown, dx: number, dy: number, dw: number, dh: number) => drawImageCalls.push([dx, dy, dw, dh]),
      fillText: (text: string) => captions.push(text),
      measureText: (text: string) => ({ width: text.length * 7 }),
      lineWidth: 1,
      strokeStyle: '',
      fillStyle: '',
      font: '',
      textBaseline: 'alphabetic',
    } as unknown as CanvasRenderingContext2D;

    const origCreate = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation(((tag: string) => {
      const el = origCreate(tag);
      if (tag === 'canvas') {
        const c = el as HTMLCanvasElement;
        vi.spyOn(c, 'getContext').mockReturnValue(fakeCtx);
        vi.spyOn(c, 'toBlob').mockImplementation((cb) => {
          canvasW = c.width;
          canvasH = c.height;
          cb(new Blob(['png-bytes'], { type: 'image/png' }));
        });
      }
      return el;
    }) as typeof document.createElement);

    const origCreateBitmap = globalThis.createImageBitmap;
    (globalThis as unknown as { createImageBitmap: unknown }).createImageBitmap = vi.fn(async () => ({
      width: 320,
      height: 240,
      close: vi.fn(),
    }));
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:side-by-side-png');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    try {
      const ok = await exportComparePng({
        source: {
          kind: 'file',
          file: new File(['x'], 'frame.png', { type: 'image/png' }),
          name: 'frame.png',
          url: 'blob:frame',
          width: 320,
          height: 240,
        },
        src: null,
        width: 320,
        height: 240,
        view: 'side-by-side',
        splitPct: 50,
        layerA: {
          boxes: [{ class: 0, x1: 10, y1: 10, x2: 50, y2: 50, score: 0.9 }],
          classes: ['circle'],
          labels: true,
          scores: true,
          caption: compareBadgeText('A', 'shapes-s', 1, 12),
        },
        layerB: {
          boxes: [],
          classes: ['circle'],
          labels: true,
          scores: true,
          caption: compareBadgeText('B', 'shapes-n', null, null),
        },
        filenameBase: 'frame-shapes-s-vs-shapes-n',
      });
      expect(ok).toBe(true);
      expect(canvasW).toBe(644); // 320 * 2 + 4px gap
      expect(canvasH).toBe(272); // 240 + 32px header
      expect(drawImageCalls).toEqual([
        [0, 32, 320, 240],
        [324, 32, 320, 240],
      ]);
      expect(captions).toContain('A · shapes-s · 1 det · 12.0 ms');
      expect(captions).toContain('B · shapes-n · unavailable');
      expect(click).toHaveBeenCalledTimes(1);
      expect(pendingBlobRevokeCount()).toBe(1);
      revokePendingBlobUrls();
      expect(pendingBlobRevokeCount()).toBe(0);
    } finally {
      (globalThis as unknown as { createImageBitmap: unknown }).createImageBitmap = origCreateBitmap;
    }
  });
});
