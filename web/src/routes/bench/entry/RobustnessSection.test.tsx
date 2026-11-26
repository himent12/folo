// @vitest-environment happy-dom
// RobustnessSection: fallback selection and keyboard navigation safety when chosen cell disappears or rows update.
import { flush, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { Bench, BenchCatalog, BenchEntry, BenchResult } from '$lib/api/types';
import { render } from '../../../../framework/testing';
import RobustnessSection from './RobustnessSection';

const entry = { id: 'be_1', name: 'e1', status: 'done' } as unknown as BenchEntry;

const makeResult = (conditions: { condition: string; severities: number[] }[]): BenchResult => ({
  entry,
  accuracy: {
    map50_95: 0.75,
    map50: 0.9,
    map75: 0.8,
    precision: 0.85,
    recall: 0.82,
    f1: 0.83,
    map_small: null,
    map_medium: 0.74,
    map_large: 0.78,
    ci95: [0.7, 0.8],
    best_conf: 0.25,
    per_class: [],
  },
  localization: { mean_iou: 0.8, matches: 10, histogram: [0, 0, 0, 0, 0, 2, 3, 3, 2, 0] },
  robustness: {
    retention: 0.8,
    conditions: conditions.map((c) => ({
      condition: c.condition,
      weight: 1,
      retention: 0.8,
      points: c.severities.map((s) => ({ severity: s, map50_95: 0.6, map50: 0.75, retention: 0.8 })),
    })),
  },
  calibration: null,
  self_labeling: {
    target_precision: 0.9,
    threshold: null,
    precision: 0.8,
    recall: 0.5,
    mean_iou: 0.75,
    labels_per_image: 1,
    labels: 10,
    readiness: 'not_ready',
    curve: [],
  },
});

const bench = {
  id: 'bn_1',
  name: 'Test',
  description: '',
  source: { kind: 'dataset', dataset_id: 'ds_1' },
  source_name: 'ds',
  fingerprint: 'abc',
  images: 10,
  boxes: 20,
  classes: ['box'],
  entries: 1,
  active: false,
  best: null,
  created: '2026-01-01T00:00:00Z',
  plan: { conditions: [{ condition: 'haze', severities: [1, 3], weight: 1 }] },
} as unknown as Bench;

const catalog = undefined as BenchCatalog | undefined;

let host: HTMLElement;
beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
});
afterEach(() => {
  host.remove();
});

describe('RobustnessSection', () => {
  it('falls back to a valid cell and keeps keyboard navigation working when the previously chosen cell vanishes', () => {
    const res = signal<BenchResult>(
      makeResult([
        { condition: 'haze', severities: [1, 3] },
        { condition: 'blur', severities: [2, 4] },
      ]),
    );
    const view = render(() => <RobustnessSection result={res.get()} entry={entry} bench={bench} catalog={catalog} />, host);
    flush();

    // Select blur severity 4.
    const blur4 = [...host.querySelectorAll<HTMLButtonElement>('button.cell')].find((b) => b.getAttribute('aria-label')?.startsWith('blur, severity 4'))!;
    blur4.click();
    flush();
    expect(blur4.tabIndex).toBe(0);

    // Replace result so blur is no longer present: tabindex="0" must fall back to a valid haze cell and arrows must not crash.
    res.set(makeResult([{ condition: 'haze', severities: [1, 3] }]));
    flush();
    const activeCells = [...host.querySelectorAll<HTMLButtonElement>('button.cell[tabindex="0"]')];
    expect(activeCells).toHaveLength(1);
    expect(activeCells[0]!.getAttribute('aria-label')).toContain('haze');

    const table = host.querySelector('table.heat')!;
    expect(() => table.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }))).not.toThrow();
    flush();
    expect(host.querySelector<HTMLButtonElement>('button.cell[tabindex="0"]')?.getAttribute('aria-label')).toContain('haze, severity 3');

    view.unmount();
  });
});
