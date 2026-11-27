// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import type { Epoch } from '$lib/api/types';
import PerClass from './PerClass';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
});

function open(epochs: Epoch[], ended: boolean) {
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <PerClass classes={['circle']} epochs={epochs} version={0} loading={false} ended={ended} />, target);
  flushSync();
  return target.querySelector('.none')?.textContent?.trim();
}

const validated: Epoch = {
  epoch: 1,
  train_loss: { box: 1, cls: 1, dfl: 1, total: 3 },
  val: { map50: 0, map50_95: 0, precision: 0, recall: 0, per_class: [] },
  duration_s: 1,
};

describe('PerClass with no rows', () => {
  it('tells a run that never validated from a validation with no class to score', () => {
    expect(open([], true)).toBe('No per-class results: it ended before its first validation.');
    c?.unmount();
    target.remove();
    expect(open([], false)).toBe('Per-class metrics appear after the first validation pass.');
    c?.unmount();
    target.remove();
    // Validation ran, but its images have no boxes (a background-only split): it did not "end before" one.
    expect(open([validated], true)).toBe('No per-class results: no class has a box in the validation images.');
  });
});

describe('PerClass with a long class name', () => {
  it('cuts the name in its cell, whole in its tooltip, so the table keeps its column (studio-shell handover)', () => {
    const long = 'platelet_cluster_with_overlapping_red_blood_cells_partially';
    const e: Epoch = { ...validated, val: { ...validated.val, per_class: [{ class: 0, map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5 }] } };
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <PerClass classes={[long]} epochs={[e]} version={0} loading={false} />, target);
    flushSync();
    const name = target.querySelector<HTMLElement>('td.c-name .truncate')!;
    expect(name.textContent?.trim()).toBe(long);
    expect(name.getAttribute('title')).toBe(long);
  });
});

describe('PerClass with a tiny AP', () => {
  it('never writes a class that scored as 0.000 (issue #382)', () => {
    const e: Epoch = { ...validated, val: { ...validated.val, per_class: [{ class: 0, map50: 0.0009, map50_95: 0.00042, precision: 0.0002, recall: 0.4 }] } };
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <PerClass classes={['cell']} epochs={[e]} version={0} loading={false} />, target);
    flushSync();
    const cells = [...target.querySelectorAll('td.num')].map((td) => td.textContent?.trim());
    expect(cells).toEqual(['0.00042', '·', '0.001', '0.00020', '0.400']);
  });

  it('keeps a positive upper bound on the bar scale for near-zero AP instead of rounding to 0–0', () => {
    const e: Epoch = { ...validated, val: { ...validated.val, per_class: [{ class: 0, map50: 0.0001, map50_95: 0.00004, precision: 0.0001, recall: 0.01 }] } };
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <PerClass classes={['circle']} epochs={[e]} version={0} loading={false} />, target);
    flushSync();
    const meta = target.querySelector('#per-class .meta')?.textContent?.trim() ?? target.textContent ?? '';
    const barHeader = target.querySelector('th.bar-col .sr-only')?.textContent?.trim() ?? '';
    expect(meta).toBe('epoch 1 · bars 0–0.000042');
    expect(barHeader).toBe('AP50-95 bar, scaled to 0–0.000042');
  });
});
