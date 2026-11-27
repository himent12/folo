// @vitest-environment happy-dom
// Validation helpers, multi-error WhyLine indicator, and Summary reviewRows warning flags for New run.
import { flush as flushSync, signal } from 'fw';
import { describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import type { CloudPick } from './cloudPick';
import { errorCount, firstError, jumpTarget, type FormErrors, type Jump } from './jump';
import { reviewRows } from './summary';
import WhyLine from './WhyLine';

const cleanErrors = (): FormErrors => ({
  name: null,
  epochs: null,
  lr0: null,
  seed: null,
  close: null,
  dataset: null,
  weights: null,
  device: null,
  amp: null,
  runon: null,
  offer: null,
  hours: null,
  preempt: null,
});

describe('New run validation summary & WhyLine', () => {
  it('counts errors and reports remaining errors in firstError', () => {
    expect(errorCount(cleanErrors())).toBe(0);
    expect(firstError(cleanErrors())).toBeNull();

    const one = { ...cleanErrors(), epochs: 'A whole number from 1 to 10,000.' };
    expect(errorCount(one)).toBe(1);
    expect(firstError(one)).toEqual({ text: 'Epochs: A whole number from 1 to 10,000.', key: 'epochs', more: 0 });

    const three = {
      ...one,
      name: 'Give the run a name.',
      seed: 'A whole number from 0 up.',
    };
    expect(errorCount(three)).toBe(3);
    expect(firstError(three)).toEqual({ text: 'Run name: Give the run a name.', key: 'name', more: 2 });
  });

  it('reactively updates (+N more) in WhyLine and jumps to the first error', () => {
    const host = document.body.appendChild(document.createElement('div'));
    const jump = vi.fn();
    const why = signal<{ text: string; key: Jump; more?: number } | null>({ text: 'Run name: empty', key: 'name', more: 2 });
    const view = render(() => <WhyLine id="start-why" why={why.get()} jump={jump} />, host);
    flushSync();
    expect(host.querySelector('#start-why')?.textContent).toBe('Run name: empty (+2 more)');
    host.querySelector<HTMLButtonElement>('#start-why button')!.click();
    expect(jump).toHaveBeenCalledWith('name');

    why.set({ text: 'Epochs: bad', key: 'epochs', more: 0 });
    flushSync();
    expect(host.querySelector('#start-why')?.textContent).toBe('Epochs: bad');

    why.set(null);
    flushSync();
    expect(host.querySelector('#start-why')).toBeNull();
    view.unmount();
    host.remove();
  });

  it('flags amp, cloud hours (as its own Stop after row), lr0, seed, and close_mosaic errors in reviewRows', () => {
    const errs: FormErrors = {
      ...cleanErrors(),
      amp: 'Not available on this server now.',
      hours: 'Pick a stop-after limit.',
      lr0: 'Greater than 0 and below 10.',
      seed: 'A whole number from 0 up.',
      close: 'A whole number up to the epoch count.',
    };
    const fakeCloud = { provider: 'RunPod', provisions: true, preemptRetries: 0, offer: { gpu: 'RTX 4090' } } as unknown as CloudPick;
    const rows = reviewRows(undefined, errs, 'folo-s', 50, 16, 640, true, fakeCloud, 'Auto', null, undefined, true, 'none', undefined, 'auto', false, true);
    const byKey = new Map(rows.map((r) => [r.key, r]));
    expect(byKey.get('runon')?.warn).toBe(false);
    expect(byKey.get('hours')).toEqual({ key: 'hours', label: 'Stop after', value: 'Invalid', warn: true });
    expect(byKey.get('preempt')).toEqual({ key: 'preempt', label: 'Spot auto-resume', value: 'Off', warn: false });
    expect(byKey.get('amp')?.warn).toBe(true);
    expect(byKey.get('lr0')).toEqual({ key: 'lr0', label: 'Learning rate', value: 'Invalid', warn: true });
    expect(byKey.get('seed')).toEqual({ key: 'seed', label: 'Seed', value: 'Invalid', warn: true });
    expect(byKey.get('close')).toEqual({ key: 'close', label: 'Close mosaic', value: 'Invalid', warn: true });
  });

  it('jumpTarget("data") falls back to the first enabled dataset radio when none is checked', () => {
    const host = document.body.appendChild(document.createElement('div'));
    host.innerHTML = '<input type="radio" name="dataset" disabled /><input type="radio" name="dataset" id="ds-first" />';
    try {
      expect(jumpTarget('data')?.id).toBe('ds-first');
    } finally {
      host.remove();
    }
  });
});
