// @vitest-environment happy-dom
// BenchCompare keyboard Escape coordination: G-chord cancellation and open popovers stay ahead of chip unpinning.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { gChord } from '$lib/shell/shortcuts';
import { keyPrefs } from '$lib/ui/keyPrefs';
import { render } from '../../framework/testing';
import BenchCompare from './BenchCompare';

let target: HTMLElement;
let c: ReturnType<typeof render> | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  keyPrefs.singleKeys = true;
  gChord.set(false);
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  gChord.set(false);
  vi.restoreAllMocks();
});

describe('BenchCompare Escape coordination', () => {
  it('does not unpin an entry when Escape cancels a G-chord or when a popover is open', async () => {
    const benches = await hub.api.benches();
    let benchId = '';
    let entries: Awaited<ReturnType<typeof hub.api.benchEntries>> = [];
    for (const candidate of benches) {
      const done = (await hub.api.benchEntries(candidate.id)).filter((e) => e.status === 'done');
      if (done.length >= 2) {
        benchId = candidate.id;
        entries = done;
        break;
      }
    }
    expect(entries.length).toBeGreaterThanOrEqual(2);
    const ids = `${entries[0]!.id},${entries[1]!.id}`;
    const url = `#/benchmarks/${benchId}/compare?entries=${ids}`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);

    c = render(() => <BenchCompare />, target);
    let firstChip!: HTMLButtonElement;
    await vi.waitFor(() => {
      flushSync();
      const btn = target.querySelector<HTMLButtonElement>('button.ck');
      if (!btn) throw new Error('loading chips');
      firstChip = btn;
    });

    // Pin the first entry.
    firstChip.click();
    flushSync();
    expect(firstChip.getAttribute('aria-pressed')).toBe('true');

    // Escape while a G-chord is active must NOT unpin the chip.
    gChord.set(true);
    const chordEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(chordEsc);
    flushSync();
    expect(chordEsc.defaultPrevented).toBe(false);
    expect(firstChip.getAttribute('aria-pressed')).toBe('true');
    gChord.set(false);

    // Repeating Escape must NOT unpin the chip.
    const repeatEsc = new KeyboardEvent('keydown', { key: 'Escape', repeat: true, bubbles: true, cancelable: true });
    document.dispatchEvent(repeatEsc);
    flushSync();
    expect(firstChip.getAttribute('aria-pressed')).toBe('true');

    // Escape while an open info popover is present must NOT unpin the chip.
    const pop = document.createElement('div');
    pop.className = 'info-pop open';
    document.body.appendChild(pop);
    try {
      const popEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
      document.dispatchEvent(popEsc);
      flushSync();
      expect(popEsc.defaultPrevented).toBe(false);
      expect(firstChip.getAttribute('aria-pressed')).toBe('true');
    } finally {
      pop.remove();
    }

    // Clean Escape unpins the chip.
    const esc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(esc);
    flushSync();
    expect(esc.defaultPrevented).toBe(true);
    expect(firstChip.getAttribute('aria-pressed')).toBe('false');
  });

  it('keeps the Add entry picker available when invalid entry IDs fill URL slots and prunes them on add', async () => {
    const benches = await hub.api.benches();
    let benchId = '';
    let entries: Awaited<ReturnType<typeof hub.api.benchEntries>> = [];
    for (const candidate of benches) {
      const done = (await hub.api.benchEntries(candidate.id)).filter((e) => e.status === 'done');
      if (done.length >= 3) {
        benchId = candidate.id;
        entries = done;
        break;
      }
    }
    expect(entries.length).toBeGreaterThanOrEqual(3);

    // One valid entry plus three invalid IDs (4 total URL tokens): picker must still show and adding must succeed.
    const url = `#/benchmarks/${benchId}/compare?entries=${entries[0]!.id},missing1,missing2,missing3`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    let adder!: HTMLSelectElement;
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('.chips .chip')).toHaveLength(1);
      const sel = target.querySelector<HTMLSelectElement>('.add select');
      if (!sel) throw new Error('expected Add entry picker');
      adder = sel;
    });

    adder.value = entries[1]!.id;
    adder.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(go).toHaveBeenCalledWith(`/benchmarks/${benchId}/compare?entries=${entries[0]!.id},${entries[1]!.id}`, { replace: true });
    expect(target.querySelectorAll('.chips .chip')).toHaveLength(2);
    expect(target.querySelector<HTMLSelectElement>('.add select')).not.toBeNull();

    // Two valid entries plus two invalid IDs also keep the Add entry picker available.
    router.route = parse(`#/benchmarks/${benchId}/compare?entries=${entries[0]!.id},${entries[1]!.id},missing1,missing2`);
    flushSync();
    expect(target.querySelectorAll('.chips .chip')).toHaveLength(2);
    expect(target.querySelector<HTMLSelectElement>('.add select')).not.toBeNull();
  });
});
