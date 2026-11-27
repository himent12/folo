// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchAccuracy, BenchEntry, BenchEntryImage } from '$lib/api/types';
import ImagesSection from './ImagesSection';

let target: HTMLElement;
let mounted: Rendered | undefined;

const BENCH: Bench = {
  id: 'bn_test',
  name: 'Test bench',
  description: '',
  dataset_id: 'ds_1',
  split: 'val',
  images: 3,
  classes: ['bolt', 'nut'],
  plan: { conditions: [] },
  fingerprint: 'fp_1',
  created: '2026-01-01T00:00:00Z',
} as unknown as Bench;

const ENTRY: BenchEntry = {
  id: 'be_test',
  name: 'folo-s',
  status: 'done',
  scored: true,
  contestant: { kind: 'model', model_id: 'm_1', arch: 'folo-s', params: 1e6, imgsz: 640 },
} as unknown as BenchEntry;

const ACCURACY: BenchAccuracy = {
  precision: 0.8,
  recall: 0.75,
  f1: 0.77,
  best_conf: 0.5,
  map50: 0.8,
  map50_95: 0.6,
  per_class: [],
} as unknown as BenchAccuracy;

const SAMPLE_IMAGES: BenchEntryImage[] = [
  {
    image: {
      id: 'im_fn',
      file: 'missed_only.jpg',
      width: 200,
      height: 200,
      boxes: [
        { class: 0, x1: 10, y1: 10, x2: 50, y2: 50 },
        { class: 1, x1: 80, y1: 80, x2: 120, y2: 120 },
      ],
    },
    detections: [{ class: 0, score: 0.9, x1: 10, y1: 10, x2: 50, y2: 50 }],
    missed: 1,
    wrong: 0,
  },
  {
    image: {
      id: 'im_fp',
      file: 'mismatch_fp.jpg',
      width: 200,
      height: 200,
      boxes: [{ class: 1, x1: 20, y1: 20, x2: 60, y2: 60 }],
    },
    detections: [
      { class: 0, score: 0.82, x1: 20, y1: 20, x2: 60, y2: 60 }, // class mismatch
      { class: 0, score: 0.41, x1: 130, y1: 130, x2: 160, y2: 160 }, // background FP
    ],
    missed: 1,
    wrong: 2,
  },
  {
    image: {
      id: 'im_clean',
      file: 'clean.jpg',
      width: 200,
      height: 200,
      boxes: [{ class: 0, x1: 10, y1: 10, x2: 50, y2: 50 }],
    },
    detections: [{ class: 0, score: 0.95, x1: 10, y1: 10, x2: 50, y2: 50 }],
    missed: 0,
    wrong: 0,
  },
];

beforeEach(() => {
  const api = new MockApi('normal', 0);
  api.benchEntryImages = vi.fn(async () => ({ total: SAMPLE_IMAGES.length, conf: 0.5, images: SAMPLE_IMAGES }));
  api.benchImageUrl = vi.fn(async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==');
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
    this.open = false;
  };
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target.remove();
  vi.restoreAllMocks();
});

async function until<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

describe('ImagesSection: Per-Image Error-Analysis Explorer', () => {
  it('filters by FN/FP/errors, sorts diagnostically, toggles GT/Pred/Errors layers, and shows per-box IoU & class mismatches', async () => {
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(240);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(240);

    mounted = render(() => <ImagesSection bench={BENCH} entry={ENTRY} accuracy={ACCURACY} />, target);
    await until(() => target.querySelectorAll('#images .tile:not(.sk)').length === 3, 'all 3 tiles loaded');

    const tileFiles = () => [...target.querySelectorAll('#images .tile:not(.sk)')].map((t) => t.getAttribute('aria-label')!.split(':')[0]);
    // Default worst-first puts mismatch_fp (3 errors) first, then missed_only (1 error), then clean (0 errors).
    expect(tileFiles()).toEqual(['mismatch_fp.jpg', 'missed_only.jpg', 'clean.jpg']);

    // Filter to False pos (FP): only mismatch_fp.jpg matches.
    const fpRadio = [...target.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Filter images by error"] [role="radio"]')].find((b) =>
      b.textContent?.includes('False pos (FP)'),
    )!;
    fpRadio.click();
    flushSync();
    expect(tileFiles()).toEqual(['mismatch_fp.jpg']);
    expect(target.querySelector('#images .count')?.textContent).toContain('1 matching of 3 loaded');

    // Reset filter to All, then sort by Lowest score.
    const allRadio = [...target.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Filter images by error"] [role="radio"]')][0]!;
    allRadio.click();
    flushSync();
    const sortSelect = target.querySelector<HTMLSelectElement>('select[aria-label="Sort evaluated images"]')!;
    sortSelect.value = 'lowest-score';
    sortSelect.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(tileFiles()).toEqual(['mismatch_fp.jpg', 'missed_only.jpg', 'clean.jpg']);

    // Toggle Pred off: MatchView renders GT boxes as plain-truth and hides predicted boxes.
    const predToggle = [...target.querySelectorAll<HTMLButtonElement>('[aria-label="Canvas box layers"] button')].find((b) => b.textContent?.trim() === 'Pred')!;
    expect(predToggle.getAttribute('aria-pressed')).toBe('true');
    predToggle.click();
    flushSync();
    expect(predToggle.getAttribute('aria-pressed')).toBe('false');
    expect(target.querySelectorAll('#images .tile .box.wrong, #images .tile .box.right')).toHaveLength(0);
    expect(target.querySelectorAll('#images .tile .box.plain-truth, #images .tile .box.missed').length).toBeGreaterThan(0);
    predToggle.click();
    flushSync();

    // Open mismatch_fp.jpg in the large inspector modal and verify per-box class mismatch + background FP diagnostics.
    (target.querySelector('#images .tile') as HTMLButtonElement).click();
    flushSync();
    const dlg = await until(() => target.querySelector('dialog[open]'), 'image inspector modal');
    const listText = dlg.querySelector('.v-list')?.textContent ?? '';
    expect(listText).toContain('Class mismatch (GT: nut)');
    expect(listText).toContain('Background FP');
    expect(listText).toContain('Missed · pred bolt');

    // Toggle Errors off inside the modal switches boxes to plain-pred / plain-truth without error crosses.
    const errToggle = [...dlg.querySelectorAll<HTMLButtonElement>('[aria-label="Canvas box layers"] button')].find((b) => b.textContent?.trim() === 'Errors')!;
    errToggle.click();
    flushSync();
    expect(dlg.querySelectorAll('.box .x')).toHaveLength(0);
    expect(dlg.querySelectorAll('.box.plain-pred').length).toBe(2);
    expect(dlg.querySelectorAll('.box.plain-truth').length).toBe(1);
  });

  it('exports filtered FP/FN error-analysis CSV and JSON from ImagesSection and disables export when no errors exist', async () => {
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(240);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(240);
    const blobs: Blob[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      blobs.push(b as Blob);
      return `blob:bench-err-${blobs.length}`;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    mounted = render(() => <ImagesSection bench={BENCH} entry={ENTRY} accuracy={ACCURACY} />, target);
    await until(() => target.querySelectorAll('#images .tile:not(.sk)').length === 3, 'all 3 tiles loaded');

    const csvBtn = target.querySelector<HTMLButtonElement>('[data-bench-error-export="csv"]')!;
    const jsonBtn = target.querySelector<HTMLButtonElement>('[data-bench-error-export="json"]')!;
    expect(csvBtn.disabled).toBe(false);
    expect(jsonBtn.disabled).toBe(false);

    // Filter to False pos (FP) and export both CSV and JSON -> only FP rows from mismatch_fp.jpg are exported.
    const fpRadio = [...target.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Filter images by error"] [role="radio"]')].find((b) =>
      b.textContent?.includes('False pos (FP)'),
    )!;
    fpRadio.click();
    flushSync();
    csvBtn.click();
    jsonBtn.click();
    expect(click).toHaveBeenCalledTimes(2);
    const csvText = await blobs[0]!.text();
    expect(csvText).toContain('mismatch_fp.jpg');
    expect(csvText).toContain(',FP,class-mismatch,');
    expect(csvText).toContain(',FP,background,');
    expect(csvText).not.toContain(',FN,');

    const jsonData = JSON.parse(await blobs[1]!.text());
    expect(jsonData.filter).toBe('fp');
    expect(jsonData.summary.fp).toBe(2);
    expect(jsonData.summary.fn).toBe(0);

    // Clean-only entry disables both export buttons.
    mounted.unmount();
    const api = hub.api as MockApi;
    api.benchEntryImages = vi.fn(async () => ({ total: 1, conf: 0.5, images: [SAMPLE_IMAGES[2]!] }));
    mounted = render(() => <ImagesSection bench={BENCH} entry={ENTRY} accuracy={ACCURACY} />, target);
    await until(() => target.querySelectorAll('#images .tile:not(.sk)').length === 1, 'clean tile loaded');
    expect(target.querySelector<HTMLButtonElement>('[data-bench-error-export="csv"]')?.disabled).toBe(true);
    expect(target.querySelector<HTMLButtonElement>('[data-bench-error-export="json"]')?.disabled).toBe(true);
  });
});
