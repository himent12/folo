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
