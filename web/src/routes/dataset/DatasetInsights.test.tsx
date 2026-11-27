// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import type { Dataset, DatasetStats, Split } from '$lib/api/types';
import DatasetInsights from './DatasetInsights';

let target: HTMLElement;
let mounted: { unmount(): void } | undefined;

const DATASET: Dataset = {
  id: 'ds_fasteners',
  name: 'fasteners',
  path: '/data/fasteners',
  format: 'yolo',
  classes: ['bolt', 'nut', 'washer', 'rivet'],
  created: '2026-11-01T00:00:00Z',
  splits: { train: 800, val: 160, test: 40 },
};

const IMBALANCED_STATS: DatasetStats = {
  boxes: 1450,
  empty_images: 140,
  broken_labels: 2,
  class_counts: [900, 400, 110, 40],
  box_size_hist: { edges: [0, 0.05, 1], counts: [220, 1230] },
  boxes_per_image_hist: { edges: [0, 1, 2, 4], counts: [140, 400, 460] },
};

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
});

describe('DatasetInsights', () => {
  it('renders class imbalance, quality indicators, split breakdown, and quick-jump controls', () => {
    const activeClass = signal<number | undefined>(undefined);
    const activeSplit = signal<'all' | Split>('all');
    const onSelectClass = vi.fn((c: number | undefined) => activeClass.set(c));
    const onSelectSplit = vi.fn((s: 'all' | Split) => activeSplit.set(s));
    const onOpenStats = vi.fn();

    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <DatasetInsights
          dataset={DATASET}
          stats={IMBALANCED_STATS}
          counts={{ error: 2, warning: 5, info: 140, by_kind: {} }}
          activeClass={activeClass.get()}
          activeSplit={activeSplit.get()}
          onSelectClass={onSelectClass}
          onSelectSplit={onSelectSplit}
          onOpenStats={onOpenStats}
        />
      ),
      target,
    );
    flushSync();

    const root = target.querySelector('[data-dataset-insights]')!;
    expect(root.textContent).toContain('22.5× class imbalance');
    expect(root.textContent).toContain('bolt');
    expect(root.textContent).toContain('rivet');
    expect(root.textContent).toContain('2 broken');
    expect(root.textContent).toContain('140 empty (14%)');
    expect(root.textContent).toContain('15% tiny');
    expect(root.textContent).toContain('5 warnings');

    // Quick jump to rarest underrepresented class ("rivet", cls = 3)
    const jumpBtn = target.querySelector<HTMLButtonElement>('[data-inspect-rarest]')!;
    expect(jumpBtn.textContent).toContain('Inspect rarest: rivet');
    jumpBtn.click();
    flushSync();
    expect(onSelectClass).toHaveBeenLastCalledWith(3);
    expect(jumpBtn.getAttribute('aria-pressed')).toBe('true');

    // Clicking again clears the class filter
    jumpBtn.click();
    flushSync();
    expect(onSelectClass).toHaveBeenLastCalledWith(undefined);

    // Clicking a split pill toggles the split filter
    const valPill = [...target.querySelectorAll<HTMLButtonElement>('.spill')].find((b) => b.textContent?.includes('val'))!;
    valPill.click();
    flushSync();
    expect(onSelectSplit).toHaveBeenLastCalledWith('val');
    valPill.click();
    flushSync();
    expect(onSelectSplit).toHaveBeenLastCalledWith('all');

    // Clicking Stats jumps to the Statistics tab
    target.querySelector<HTMLButtonElement>('.stats-link')!.click();
    expect(onOpenStats).toHaveBeenCalledTimes(1);
  });

  it('shows a clean summary when the dataset has balanced classes and zero issues', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <DatasetInsights
          dataset={DATASET}
          stats={{
            boxes: 1600,
            empty_images: 0,
            broken_labels: 0,
            class_counts: [420, 410, 390, 380],
            box_size_hist: { edges: [0, 0.05, 1], counts: [20, 1580] },
            boxes_per_image_hist: { edges: [0, 1, 2, 4], counts: [0, 400, 600] },
          }}
          activeClass={undefined}
          activeSplit="all"
          onSelectClass={() => {}}
          onSelectSplit={() => {}}
          onOpenStats={() => {}}
        />
      ),
      target,
    );
    flushSync();

    const root = target.querySelector('[data-dataset-insights]')!;
    expect(root.textContent).toContain('1.1× balanced');
    expect(root.textContent).toContain('Labels clean');
    expect(target.querySelector('[data-empty-indicator]')).toBeNull();
  });

  it('shows unreadable image errors alongside broken labels without double-counting, formats sub-1% empty share, and hides split bar when 0 images', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <DatasetInsights
          dataset={{ ...DATASET, splits: { train: 5000, val: 0, test: 0 } }}
          stats={{
            boxes: 10000,
            empty_images: 1,
            broken_labels: 3,
            class_counts: [2500, 2500, 2500, 2500],
            box_size_hist: { edges: [0, 0.05, 1], counts: [0, 10000] },
            boxes_per_image_hist: { edges: [0, 1, 2, 4], counts: [1, 0, 4999] },
          }}
          counts={{ error: 5, warning: 0, info: 1, by_kind: { broken_label: 3, corrupt_image: 1, missing_image: 1 } }}
          activeClass={undefined}
          activeSplit="all"
          onSelectClass={() => {}}
          onSelectSplit={() => {}}
          onOpenStats={() => {}}
        />
      ),
      target,
    );
    flushSync();

    const root = target.querySelector('[data-dataset-insights]')!;
    expect(root.textContent).toContain('3 broken');
    expect(root.textContent).toContain('2 unreadable');
    expect(root.textContent).toContain('1 empty (<1%)');
    expect(root.textContent).toContain('no val');

    // 0-image dataset hides the empty split bar cleanly
    mounted.unmount();
    mounted = render(
      () => (
        <DatasetInsights
          dataset={{ ...DATASET, splits: { train: 0, val: 0, test: 0 } }}
          stats={{
            boxes: 0,
            empty_images: 0,
            broken_labels: 0,
            class_counts: [0, 0, 0, 0],
            box_size_hist: { edges: [0, 1], counts: [0] },
            boxes_per_image_hist: { edges: [0, 1], counts: [0] },
          }}
          activeClass={undefined}
          activeSplit="all"
          onSelectClass={() => {}}
          onSelectSplit={() => {}}
          onOpenStats={() => {}}
        />
      ),
      target,
    );
    flushSync();
    expect(target.querySelector('.split-bar')).toBeNull();
  });
});
