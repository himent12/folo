// @vitest-environment happy-dom
// Shell-safe `folo dataset` curation CLI helper paired with DatasetInsights (`check --perceptual`, `split`, `filter`, `merge`).
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Dataset, DatasetStats } from '$lib/api/types';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../framework/testing';
import DatasetInsights from './DatasetInsights';
import { buildDatasetCurateCli, curateCliCopiedMessage } from './curateCli';

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
  for (const t of toast.items) toast.dismiss(t.id);
  vi.restoreAllMocks();
});

describe('buildDatasetCurateCli', () => {
  it('generates exact docs/CLI.md commands for check, split, filter (with split), merge, and convert', () => {
    expect(buildDatasetCurateCli(BASE_DATASET, 'check').command).toBe(
      'folo dataset check D:/datasets/fasteners --perceptual --fix-dry-run',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'split').command).toBe(
      'folo dataset split D:/datasets/fasteners --out <out_dir>/fasteners-split --val 0.2 --seed 0',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'filter', { stats: BASE_STATS }).command).toBe(
      'folo dataset filter D:/datasets/fasteners --out <out_dir>/fasteners-filtered --drop-empty-images',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'filter', { stats: { ...BASE_STATS, empty_images: 0 }, activeSplit: 'val' }).command).toBe(
      'folo dataset filter D:/datasets/fasteners --out <out_dir>/fasteners-filtered --split val',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'merge').command).toBe(
      'folo dataset merge D:/datasets/fasteners <other_dataset> --out <out_dir>/fasteners-merged --splits stratified --val 0.2',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'convert').command).toBe(
      'folo dataset convert D:/datasets/fasteners <out_dir>/fasteners-coco --to coco',
    );
    expect(buildDatasetCurateCli({ ...BASE_DATASET, format: 'coco' }, 'convert').command).toBe(
      'folo dataset convert D:/datasets/fasteners <out_dir>/fasteners-yolo --to yolo',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'convert', { targetFormat: 'voc' }).command).toBe(
      'folo dataset convert D:/datasets/fasteners <out_dir>/fasteners-voc --to voc',
    );
    expect(buildDatasetCurateCli({ ...BASE_DATASET, path: '-leading-dash' }, 'convert', { targetFormat: 'coco' }).command).toBe(
      'folo dataset convert --to coco -- -leading-dash <out_dir>/fasteners-coco',
    );
    expect(buildDatasetCurateCli(BASE_DATASET, 'export').command).toBe(
      'folo dataset export D:/datasets/fasteners --out <out_dir>/fasteners-coco-export --format coco --images copy',
    );
    expect(
      buildDatasetCurateCli(BASE_DATASET, 'export', {
        activeSplit: 'val',
        cocoImageMode: 'reference',
        cocoPretty: true,
      }).command,
    ).toBe(
      'folo dataset export D:/datasets/fasteners --out <out_dir>/fasteners-coco-export --format coco --split val --images reference --pretty',
    );
    expect(
      buildDatasetCurateCli({ ...BASE_DATASET, path: '-leading-dash' }, 'export', { cocoImageMode: 'link' }).command,
    ).toBe(
      'folo dataset export --out <out_dir>/fasteners-coco-export --format coco --images link -- -leading-dash',
    );
    expect(curateCliCopiedMessage(buildDatasetCurateCli(BASE_DATASET, 'export'))).toBe(
      'Replace <out_dir> before running this export coco command.',
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

  it('omits --include-class with an explicit warning for comma or double-quote class names, and preserves backticks in descriptions', () => {
    const edgeDs: Dataset = {
      ...BASE_DATASET,
      classes: ['nut, washer', 'kid"s class', 'back`tick'],
    };

    // Comma in class name: clap splits --include-class on commas
    const comma = buildDatasetCurateCli(edgeDs, 'filter', { stats: { ...BASE_STATS, empty_images: 0 }, activeClass: 0 });
    expect(comma.command).not.toContain('--include-class');
    expect(comma.warning).toContain('comma');

    // Double quote in class name: PowerShell 5.1 strips literal double quotes on native arguments
    const dquote = buildDatasetCurateCli(edgeDs, 'filter', { stats: { ...BASE_STATS, empty_images: 0 }, activeClass: 1 });
    expect(dquote.command).not.toContain('--include-class');
    expect(dquote.warning).toContain('double quote');

    // Backtick in class name: stays quoted safely and preserved verbatim in description
    const backtick = buildDatasetCurateCli(edgeDs, 'filter', { stats: { ...BASE_STATS, empty_images: 0 }, activeClass: 2 });
    expect(backtick.warning).toBeNull();
    expect(backtick.command).toContain("--include-class 'back`tick'");
    expect(backtick.description).toContain('back`tick');

    // Unsafe dataset path with literal double quote falls back to fixed placeholder + warning (never interpolating name)
    const badPath = buildDatasetCurateCli({ ...BASE_DATASET, name: 'bad; rm -rf /', path: '/data/bad"dir' }, 'convert');
    expect(badPath.command).toContain('<dataset_path>');
    expect(badPath.command).not.toContain('rm -rf');
    expect(badPath.warning).toContain('Windows PowerShell');
    // Copy guidance warns about angle-bracket placeholders so users replace them before running in PowerShell (Playtest R24 P3-3)
    expect(curateCliCopiedMessage(buildDatasetCurateCli(BASE_DATASET, 'check'))).toBe(
      'Run this audit & dedup command from your terminal.',
    );
    expect(curateCliCopiedMessage(buildDatasetCurateCli(BASE_DATASET, 'filter', { stats: BASE_STATS }))).toBe(
      'Replace <out_dir> before running this filter command.',
    );
    expect(curateCliCopiedMessage(buildDatasetCurateCli(BASE_DATASET, 'merge'))).toBe(
      'Replace <other_dataset> and <out_dir> before running this merge command.',
    );
    expect(curateCliCopiedMessage(badPath)).toBe(
      'Replace <dataset_path> and <out_dir> before running this convert command.',
    );
    expect(curateCliCopiedMessage(buildDatasetCurateCli({ ...BASE_DATASET, path: '/data/bad"dir' }, 'merge'))).toBe(
      'Replace <dataset_path>, <other_dataset> and <out_dir> before running this merge command.',
    );
    // Literal `<...>` inside quoted dataset paths or class names must NOT be reported as placeholders.
    const literalAngle: Dataset = {
      ...BASE_DATASET,
      path: '/data/road/<archive>',
      classes: ['<unknown>', 'bolt'],
    };
    expect(curateCliCopiedMessage(buildDatasetCurateCli(literalAngle, 'check'))).toBe(
      'Run this audit & dedup command from your terminal.',
    );
    expect(curateCliCopiedMessage(buildDatasetCurateCli(literalAngle, 'filter', { stats: BASE_STATS, activeClass: 0 }))).toBe(
      'Replace <out_dir> before running this filter command.',
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
    expect(toast.items.at(-1)?.detail).toBe('Replace <out_dir> before running this filter command.');

    // Switch to Export COCO and toggle image mode & Pretty JSON
    const exportBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-dataset-curate-cli] [role="radio"]')].find(
      (b) => b.textContent?.trim() === 'Export COCO',
    )!;
    exportBtn.click();
    flushSync();
    expect(target.querySelector('[data-dataset-curate-cli] pre.line')?.textContent).toBe(
      'folo dataset export D:/datasets/fasteners --out <out_dir>/fasteners-coco-export --format coco --images copy',
    );
    const refBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-coco-export-options] [role="radio"]')].find(
      (b) => b.textContent?.trim() === 'Reference',
    )!;
    refBtn.click();
    target.querySelector<HTMLButtonElement>('[data-coco-export-options] [role="switch"]')!.click();
    flushSync();
    expect(target.querySelector('[data-dataset-curate-cli] pre.line')?.textContent).toBe(
      'folo dataset export D:/datasets/fasteners --out <out_dir>/fasteners-coco-export --format coco --images reference --pretty',
    );

    // Switching dataset collapses the helper cleanly
    ds.set({ ...BASE_DATASET, id: 'ds_other', name: 'other', splits: { train: 50, val: 0, test: 0 } });
    flushSync();
    expect(target.querySelector('[data-dataset-curate-cli]')).toBeNull();
  });
});
