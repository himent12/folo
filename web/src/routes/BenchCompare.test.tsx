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

  it('clears stale hovered highlight when a hovered chip is removed and later re-added', async () => {
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
    const url = `#/benchmarks/${benchId}/compare?entries=${entries[0]!.id},${entries[1]!.id},${entries[2]!.id}`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('.chips .chip')).toHaveLength(3);
    });

    // Hover chip 0 and remove it via its × button (which unmounts the chip without firing pointerleave).
    const firstChip = target.querySelector<HTMLElement>('.chips .chip')!;
    firstChip.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
    flushSync();
    const removeBtn = firstChip.querySelector<HTMLButtonElement>('button.rm')!;
    removeBtn.click();
    flushSync();
    expect(target.querySelectorAll('.chips .chip')).toHaveLength(2);

    // Re-add entry 0: no interval row should be dimmed by a stale hover on entry 0.
    const adder = target.querySelector<HTMLSelectElement>('.add select')!;
    adder.value = entries[0]!.id;
    adder.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelectorAll('.chips .chip')).toHaveLength(3);
    expect(target.querySelectorAll('.iv li.dim')).toHaveLength(0);
  });

  it('prunes unavailable entry IDs from ?entries= and shows a dismissible notice when 2+ valid entries remain', async () => {
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
    const url = `#/benchmarks/${benchId}/compare?entries=${entries[0]!.id},${entries[1]!.id},be_gone`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('.chips .chip')).toHaveLength(2);
      const notice = target.querySelector<HTMLElement>('[data-missing-entries]');
      if (!notice) throw new Error('waiting for missing-entries notice');
      expect(notice.textContent).toContain('be_gone');
      expect(notice.textContent).toContain('Omitted unavailable entry');
    });
    expect(go).toHaveBeenCalledWith(`/benchmarks/${benchId}/compare?entries=${entries[0]!.id},${entries[1]!.id}`, { replace: true });

    // Dismissing hides the notice and moves focus into the chips group instead of stranding on document.body.
    const dismiss = target.querySelector<HTMLButtonElement>('[data-missing-entries] button')!;
    dismiss.focus();
    dismiss.click();
    flushSync();
    await Promise.resolve();
    expect(target.querySelector('[data-missing-entries]')).toBeNull();
    expect(target.querySelector('.chips')?.contains(document.activeElement)).toBe(true);
  });

  it('clears the omission notice when navigating to a clean comparison on the same benchmark', async () => {
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
    const url = `#/benchmarks/${benchId}/compare?entries=${entries[0]!.id},${entries[1]!.id},be_gone,be_also_gone`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      const notice = target.querySelector<HTMLElement>('[data-missing-entries]');
      if (!notice) throw new Error('waiting for plural missing-entries notice');
      expect(notice.textContent).toContain('Omitted 2 unavailable entries');
      expect(notice.textContent).toContain('be_gone');
      expect(notice.textContent).toContain('be_also_gone');
    });
    // Mounted live region announced the omission.
    expect(target.querySelector('[role="status"]')?.textContent).toContain('Omitted 2 unavailable entries');

    // Navigating to a clean pair on the SAME benchmark must clear the old omission notice.
    router.route = parse(`#/benchmarks/${benchId}/compare?entries=${entries[1]!.id},${entries[2]!.id}`);
    flushSync();
    expect(target.querySelector('[data-missing-entries]')).toBeNull();
    expect(target.querySelectorAll('.chips .chip')).toHaveLength(2);
  });

  it('keeps an active running entry in ?entries= across baseline/add/remove edits so it joins when SSE finishes it', async () => {
    const benches = await hub.api.benches();
    let benchId = '';
    let doneEntries: Awaited<ReturnType<typeof hub.api.benchEntries>> = [];
    let runningEntry: Awaited<ReturnType<typeof hub.api.benchEntries>>[number] | undefined;
    for (const candidate of benches) {
      const all = await hub.api.benchEntries(candidate.id);
      const done = all.filter((e) => e.status === 'done');
      const running = all.find((e) => e.status === 'running' || e.status === 'queued');
      if (done.length >= 4 && running) {
        benchId = candidate.id;
        doneEntries = done;
        runningEntry = running;
        break;
      }
    }
    expect(doneEntries.length).toBeGreaterThanOrEqual(4);
    expect(runningEntry).toBeTruthy();

    let emitEntry: ((e: Awaited<ReturnType<typeof hub.api.benchEntries>>[number]) => void) | undefined;
    const realEvents = hub.api.benchEvents.bind(hub.api);
    vi.spyOn(hub.api, 'benchEvents').mockImplementation((id, handlers) => {
      emitEntry = handlers.entry;
      return realEvents(id, handlers);
    });

    const url = `#/benchmarks/${benchId}/compare?entries=${doneEntries[0]!.id},${doneEntries[1]!.id},${runningEntry!.id}`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    // Live region is already mounted during loading before any omission/status announcement.
    expect(target.querySelector('[role="status"]')).not.toBeNull();
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('.chips .chip')).toHaveLength(2);
      const pendingNote = target.querySelector<HTMLElement>('[data-pending-entries]');
      if (!pendingNote) throw new Error('waiting for pending entry note');
      expect(pendingNote.textContent).toContain(runningEntry!.name);
    });
    // Running entry must NOT be pruned from `?entries=` or reported as unavailable.
    expect(go).not.toHaveBeenCalled();
    expect(target.querySelector('[data-missing-entries]')).toBeNull();

    // Rotating baseline via `B` preserves the running entry in `?entries=` and keeps the waiting notice.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'b', bubbles: true, cancelable: true }));
    flushSync();
    expect(go).toHaveBeenLastCalledWith(`/benchmarks/${benchId}/compare?entries=${doneEntries[1]!.id},${doneEntries[0]!.id},${runningEntry!.id}`, { replace: true });
    expect(target.querySelector('[data-pending-entries]')).not.toBeNull();

    // Adding a 3rd finished entry fills all 4 slots (3 finished + 1 pending) and hides the adder while preserving runningEntry.
    const adder = target.querySelector<HTMLSelectElement>('.add select')!;
    adder.value = doneEntries[2]!.id;
    adder.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(go).toHaveBeenLastCalledWith(
      `/benchmarks/${benchId}/compare?entries=${doneEntries[1]!.id},${doneEntries[0]!.id},${doneEntries[2]!.id},${runningEntry!.id}`,
      { replace: true },
    );
    expect(target.querySelector('.add select')).toBeNull();

    // Removing one finished entry still preserves runningEntry in `?entries=`.
    target.querySelector<HTMLButtonElement>('.chips .chip button.rm')!.click();
    flushSync();
    expect(go).toHaveBeenLastCalledWith(`/benchmarks/${benchId}/compare?entries=${doneEntries[0]!.id},${doneEntries[2]!.id},${runningEntry!.id}`, { replace: true });

    // When SSE delivers the finished entry with a score, it joins the comparison automatically.
    const finishedCopy = { ...doneEntries[0]!, id: runningEntry!.id, name: runningEntry!.name };
    emitEntry?.(finishedCopy);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('.chips .chip')).toHaveLength(3);
      expect(target.querySelector('[data-pending-entries]')).toBeNull();
    });
  });

  it('reports combined missing count in EmptyState for mixed running + unavailable entries and clears omission notice on Back history entry change', async () => {
    const benches = await hub.api.benches();
    let benchId = '';
    let doneEntries: Awaited<ReturnType<typeof hub.api.benchEntries>> = [];
    let runningEntry: Awaited<ReturnType<typeof hub.api.benchEntries>>[number] | undefined;
    for (const candidate of benches) {
      const all = await hub.api.benchEntries(candidate.id);
      const done = all.filter((e) => e.status === 'done');
      const running = all.find((e) => e.status === 'running' || e.status === 'queued');
      if (done.length >= 2 && running) {
        benchId = candidate.id;
        doneEntries = done;
        runningEntry = running;
        break;
      }
    }
    expect(doneEntries.length).toBeGreaterThanOrEqual(2);
    expect(runningEntry).toBeTruthy();

    // 1 finished + 1 running + 1 unavailable -> EmptyState must report 2 entries, plus show the pending notice.
    const url = `#/benchmarks/${benchId}/compare?entries=${doneEntries[0]!.id},${runningEntry!.id},be_gone`;
    history.replaceState(history.state, '', url);
    router.route = parse(url);
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('2 entries in this link are');
      expect(target.querySelector('[data-pending-entries]')?.textContent).toContain(runningEntry!.name);
    });
    c.unmount();
    c = undefined;
    vi.restoreAllMocks();

    // History entry move (Back to clean `?entries=A,B` after `?entries=A,B,be_gone` pruned to `?entries=A,B`):
    // capture the onEntry listener to simulate traversing back to a clean history entry with identical parsed IDs.
    let entryCb: ((e: { from: number; to: number; traverse: boolean }) => void) | undefined;
    const realOnEntry = router.onEntry.bind(router);
    vi.spyOn(router, 'onEntry').mockImplementation((fn) => {
      entryCb = fn;
      return realOnEntry(fn);
    });
    const url2 = `#/benchmarks/${benchId}/compare?entries=${doneEntries[0]!.id},${doneEntries[1]!.id},be_gone`;
    history.replaceState(history.state, '', url2);
    router.route = parse(url2);
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    c = render(() => <BenchCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelector('[data-missing-entries]')?.textContent).toContain('be_gone');
    });
    // Back to original clean history entry with the same `?entries=A,B`: onEntry clears the stale notice.
    entryCb?.({ from: 1, to: 0, traverse: true });
    flushSync();
    expect(target.querySelector('[data-missing-entries]')).toBeNull();
  });
});
