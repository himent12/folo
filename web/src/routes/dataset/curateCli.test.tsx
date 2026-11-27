// @vitest-environment happy-dom
// Shell-safe `folo dataset` curation CLI helper paired with DatasetInsights (`check --perceptual`, `split`, `filter`, `merge`).
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Dataset, DatasetStats } from '$lib/api/types';
import { render, type Rendered } from '../../../framework/testing';
import DatasetInsights from './DatasetInsights';
import { buildDatasetCurateCli } from './curateCli';

const BASE_DATASET: Dataset = {
  id: 'ds_fasteners',
  name: 'fasteners',
  path: 'D:/datasets/fasteners',
  format: 'yolo',
  classes: ['bolt', 'hex nut', '-washer'],
  created: '2026-11-01T00:00:00Z',
  splits: { train: 800, val: 160, test: 40 },
};

const BASE_STATS: DatasetStats = {
  boxes: 1450,
  empty_images: 140,
  broken_labels: 0,
  class_counts: [900, 400, 150],
  box_size_hist: { edges: [0, 0.05, 1], counts: [50, 1400] },
  boxes_per_image_hist: { edges: [0, 1, 2, 4], counts: [140, 400, 460] },
};

let target: HTMLElement;
let mounted: Rendered | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

describe('buildDatasetCurateCli', () => {
  it('generates exact docs/CLI.md commands for check, split, filter, and merge', () => {
    expect(buildDatasetCurateCli(BASE_DATASET, 'check').command).toBe(
      'folo dataset check D:/datasets/fasteners --perceptual --fix-dry-run',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'split').command).toBe(
      'folo dataset split D:/datasets/fasteners --out <out_dir>/fasteners-split --val 0.2 --seed 0',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'filter', { stats: BASE_STATS }).command).toBe(
      'folo dataset filter D:/datasets/fasteners --out <out_dir>/fasteners-filtered --drop-empty-images',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'merge').command).toBe(
      'folo dataset merge D:/datasets/fasteners <other_dataset> --out <out_dir>/fasteners-merged --splits stratified --val 0.2',
    );
  });

  it('quotes spaces, apostrophes, and leading-dash paths and class names safely', () => {
    const tricky: Dataset = {
      ...BASE_DATASET,
      path: "-O'Brien's data",
    };
    expect(buildDatasetCurateCli(tricky, 'check').command).toBe(
      "folo dataset check --perceptual --fix-dry-run -- '-O''Brien''s data'",
    );
    expect(buildDatasetCurateCli(tricky, 'filter', { stats: { ...BASE_STATS, empty_images: 0 }, activeClass: 1 }).command).toBe(
      "folo dataset filter --out <out_dir>/fasteners-filtered --include-class 'hex nut' -- '-O''Brien''s data'",
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'filter', { stats: { ...BASE_STATS, empty_images: 0 }, activeClass: 2 }).command).toBe(
      'folo dataset filter D:/datasets/fasteners --out <out_dir>/fasteners-filtered --include-class=-washer',
    );
  });
});

describe('DatasetInsights CLI curation integration', () => {
  it('keeps curation CLI closed by default, opens on click, switches modes, and adapts to activeClass and dataset switch', async () => {
    const ds = signal(BASE_DATASET);
    const activeClass = signal<number | undefined>(undefined);
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <DatasetInsights
          dataset={ds.get()}
          stats={BASE_STATS}
          activeClass={activeClass.get()}
          activeSplit="all"
          onSelectClass={() => {}}
          onSelectSplit={() => {}}
          onOpenStats={() => {}}
        />
      ),
      target,
    );
    flushSync();

    const toggle = target.querySelector<HTMLButtonElement>('[data-curate-toggle]')!;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelector('[data-dataset-curate-cli]')).toBeNull();

    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(target.querySelector('[data-dataset-curate-cli] pre.line')?.textContent).toBe(
      'folo dataset check D:/datasets/fasteners --perceptual --fix-dry-run',
    );

    // Switch to Filter and activate class filter 1 ('hex nut')
    const filterBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-dataset-curate-cli] [role="radio"]')].find(
      (b) => b.textContent?.trim() === 'Filter',
    )!;
    filterBtn.click();
    activeClass.set(1);
    flushSync();
    const expectedFilter = "folo dataset filter D:/datasets/fasteners --out <out_dir>/fasteners-filtered --include-class 'hex nut' --drop-empty-images";
    expect(target.querySelector('[data-dataset-curate-cli] pre.line')?.textContent).toBe(expectedFilter);

    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    target.querySelector<HTMLButtonElement>('[data-dataset-curate-cli] button[aria-label="Copy command"]')!.click();
    await Promise.resolve();
    expect(write).toHaveBeenCalledWith(expectedFilter);

    // Switching dataset collapses the helper cleanly
    ds.set({ ...BASE_DATASET, id: 'ds_other', name: 'other', splits: { train: 50, val: 0, test: 0 } });
    flushSync();
    expect(target.querySelector('[data-dataset-curate-cli]')).toBeNull();
  });
});
