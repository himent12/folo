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

    // Clean Escape unpins the chip.
    const esc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(esc);
    flushSync();
    expect(esc.defaultPrevented).toBe(true);
    expect(firstChip.getAttribute('aria-pressed')).toBe('false');
  });
});
