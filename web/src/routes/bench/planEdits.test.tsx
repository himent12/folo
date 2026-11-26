// @vitest-environment happy-dom
// Regression tests for benchmark plan editing edge cases found in Live Chromium Playtest Round 6:
// 1. Edits made while Save plan is in flight are preserved in the draft and stay unsaved.
// 2. Clearing a weight as the only edit still engages the unsaved-changes leave guard.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, type BenchEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { RESYNC_MS } from '$lib/api/liveBench';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import BenchmarkPage from '../Benchmark';
import { useAddQuery } from './addQuery';

let target: HTMLElement;
let c: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  if (c) c.unmount();
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

async function settle(rounds = 12) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

async function openPlan() {
  router.route = parse('#/benchmarks/bn_9f03?tab=conditions');
  c = render(() => <BenchmarkPage />, target);
  flushSync();
  await settle();
}

function typeInput(el: HTMLInputElement, v: string) {
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

const weight = (name: string) => document.querySelector<HTMLInputElement>(`input[aria-label="Weight of ${name}"]`)!;
const planPanel = () => document.querySelector<HTMLElement>('.plan')!;
const btn = (label: string, root: ParentNode = planPanel()) => [...root.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label);

describe('Benchmark plan concurrent save and empty-weight leave guard', () => {
  it('preserves edits made while Save plan is in flight and keeps them marked unsaved', async () => {
    await openPlan();
    const real = hub.api.updateBench.bind(hub.api);
    let releaseSave!: () => void;
    const held = new Promise<void>((r) => (releaseSave = r));
    vi.spyOn(hub.api, 'updateBench').mockImplementationOnce(async (id, req) => {
      await held;
      return real(id, req);
    });
    const success = vi.spyOn(toast, 'success');
    const info = vi.spyOn(toast, 'info');

    typeInput(weight('Low light'), '2');
    btn('Save plan')!.click();
    flushSync();
    expect(btn('Discard')!.disabled).toBe(true);

    // Edit another condition while the first save is still pending.
    typeInput(weight('Overexposure'), '3');
    expect(weight('Overexposure').value).toBe('3');

    releaseSave();
    await settle();

    // The second edit stays in the draft, marked unsaved, with no false "Changed elsewhere" warning.
    expect(weight('Low light').value).toBe('2');
    expect(weight('Overexposure').value).toBe('3');
    expect(btn('Save plan')).toBeDefined();
    expect(btn('Discard')!.disabled).toBe(false);
    expect(document.body.textContent).toContain('Unsaved changes');
    expect(document.body.textContent).not.toContain('Changed elsewhere');
    expect(success).not.toHaveBeenCalled();
    expect(info).toHaveBeenCalledWith('Earlier plan saved', expect.stringMatching(/still unsaved/));

    // Saving again persists the second edit cleanly.
    btn('Save plan')!.click();
    await settle();
    expect(success).toHaveBeenCalled();
    expect(btn('Save plan')).toBeUndefined();
    const saved = await hub.api.bench('bn_9f03');
    expect(saved.plan.conditions.find((x) => x.condition === 'low_light')?.weight).toBe(2);
    expect(saved.plan.conditions.find((x) => x.condition === 'overexposure')?.weight).toBe(3);
  });

  it('keeps an emptied weight and blocks leaving when cleared while Save plan is in flight', async () => {
    await openPlan();
    const real = hub.api.updateBench.bind(hub.api);
    let releaseSave!: () => void;
    const held = new Promise<void>((r) => (releaseSave = r));
    vi.spyOn(hub.api, 'updateBench').mockImplementationOnce(async (id, req) => {
      await held;
      return real(id, req);
    });

    typeInput(weight('Low light'), '2');
    btn('Save plan')!.click();
    flushSync();

    // Clear Overexposure while the save is pending.
    typeInput(weight('Overexposure'), '');
    releaseSave();
    await settle();

    expect(weight('Overexposure').value).toBe('');
    expect(weight('Overexposure').getAttribute('aria-invalid')).toBe('true');

    // Trying to navigate away opens the leave dialog rather than silently dropping the cleared field.
    router.go('/runs');
    flushSync();
    const dialog = document.querySelector<HTMLDialogElement>('dialog[open]');
    expect(dialog).not.toBeNull();
    expect(dialog!.textContent).toMatch(/Overexposure: the weight is empty/);
    expect(btn('Save plan', dialog!)!.disabled).toBe(true);
  });

  it('engages the unsaved-plan leave guard and offers Discard when clearing a weight is the only edit', async () => {
    await openPlan();
    typeInput(weight('Low light'), '');
    expect(weight('Low light').getAttribute('aria-invalid')).toBe('true');

    // Toolbar offers Discard to restore the cleared field (and keeps Design with an agent on offer).
    expect(btn('Discard')).toBeDefined();
    expect(btn('Save plan')).toBeUndefined();
    expect(btn('Design with an agent')).toBeDefined();
    btn('Discard')!.click();
    flushSync();
    expect(weight('Low light').value).toBe('1');
    expect(weight('Low light').getAttribute('aria-invalid')).toBeNull();
    expect(btn('Discard')).toBeUndefined();

    // Clearing again engages the leave guard rather than silently navigating away.
    typeInput(weight('Low light'), '');
    router.go('/runs');
    flushSync();
    const dialog = document.querySelector<HTMLDialogElement>('dialog[open]');
    expect(dialog).not.toBeNull();
    expect(dialog!.textContent).toMatch(/Low light: the weight is empty/);
    expect(btn('Save plan', dialog!)!.disabled).toBe(true);
    expect(router.route.path).toBe('/benchmarks/bn_9f03');

    // Discard changes clears the invalid input and proceeds.
    btn('Discard changes', dialog!)!.click();
    await settle();
    expect(document.querySelector('dialog[open]')).toBeNull();
    expect(router.route.path).toBe('/runs');
  });

  it('clears a stale elsewhere warning when our save replaces another tab’s plan while a concurrent edit is made', async () => {
    await openPlan();
    typeInput(weight('Low light'), '2');

    // Another tab saves Haze = 4, raising the conflict warning.
    const b = await hub.api.bench('bn_9f03');
    const real = hub.api.updateBench.bind(hub.api);
    await real('bn_9f03', {
      plan: { ...b.plan, conditions: b.plan.conditions.map((c) => (c.condition === 'haze' ? { ...c, weight: 4 } : c)) },
    });
    await settle();
    expect(document.body.textContent).toContain('Changed elsewhere while you edited');

    // Save our plan (replacing the other tab's), and edit Overexposure while saving.
    let releaseSave!: () => void;
    const held = new Promise<void>((r) => (releaseSave = r));
    vi.spyOn(hub.api, 'updateBench').mockImplementationOnce(async (id, req) => {
      await held;
      return real(id, req);
    });
    btn('Save plan')!.click();
    flushSync();
    typeInput(weight('Overexposure'), '3');
    releaseSave();
    await settle();

    // The server now holds our save (Low light = 2), so the stale conflict warning must clear while Overexposure = 3 stays unsaved.
    expect(weight('Overexposure').value).toBe('3');
    expect(document.body.textContent).toContain('Unsaved changes');
    expect(document.body.textContent).not.toContain('Changed elsewhere');
  });

  it('adopts an external plan update on a clean editor without false conflict, and does not show elsewhere when unreadable matches server', async () => {
    await openPlan();
    const b = await hub.api.bench('bn_9f03');
    // Clean editor: another tab saves Haze = 4 -> adopted silently, no conflict, no dirty bar.
    await hub.api.updateBench('bn_9f03', {
      plan: { ...b.plan, conditions: b.plan.conditions.map((c) => (c.condition === 'haze' ? { ...c, weight: 4 } : c)) },
    });
    await settle();
    expect(weight('Haze').value).toBe('4');
    expect(document.body.textContent).not.toContain('Changed elsewhere');
    expect(btn('Save plan')).toBeUndefined();

    // Edit Low light = 2 and clear Overexposure, then another tab saves the exact same Low light = 2 (dirty becomes false, hasUnreadable stays true).
    typeInput(weight('Low light'), '2');
    typeInput(weight('Overexposure'), '');
    const b2 = await hub.api.bench('bn_9f03');
    await hub.api.updateBench('bn_9f03', {
      plan: { ...b2.plan, conditions: b2.plan.conditions.map((c) => (c.condition === 'low_light' ? { ...c, weight: 2 } : c)) },
    });
    await settle();
    expect(weight('Overexposure').value).toBe('');
    expect(document.body.textContent).not.toContain('Changed elsewhere');
    expect(btn('Discard')).toBeDefined();
    // An unreadable weight also gates both the header and empty-leaderboard "Add entry" buttons and blocks submitting AddEntryDialog.
    expect(btn('Add entry', document)!.disabled).toBe(true);
    const startSpy = vi.spyOn(hub.api, 'startBenchEntry');
    const lbTab = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((b) => b.textContent?.includes('Leaderboard'))!;
    lbTab.click();
    await settle();
    const addButtons = [...document.querySelectorAll<HTMLButtonElement>('button')].filter((b) => !b.closest('dialog') && b.textContent?.trim() === 'Add entry');
    expect(addButtons.length).toBe(2);
    for (const b of addButtons) {
      expect(b.disabled).toBe(true);
      b.click();
    }
    flushSync();
    expect(document.querySelector('dialog[open]')).toBeNull();
    // Even opening via ?add=model while unsaved must gate submission and refuse startBenchEntry.
    router.route = parse('#/benchmarks/bn_9f03?add=model');
    await settle();
    const addDialog = document.querySelector<HTMLDialogElement>('dialog[open]');
    expect(addDialog).not.toBeNull();
    expect(addDialog!.textContent).toContain('Save or discard the plan');
    const submitBtn = addDialog!.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    expect(submitBtn.disabled).toBe(true);
    addDialog!.querySelector<HTMLFormElement>('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
    expect(startSpy).not.toHaveBeenCalled();
  });

  it('opens Add Entry reactively when ?add= is navigated to on an already-mounted benchmark and strips ?add= on close', async () => {
    router.route = parse('#/benchmarks/bn_9f03');
    c = render(() => <BenchmarkPage />, target);
    flushSync();
    await settle();
    expect(document.querySelector('dialog[open]')).toBeNull();

    // Following ?add=agent while already mounted opens Add Entry immediately on the Agent tab without reload.
    router.go('/benchmarks/bn_9f03?add=agent', { replace: true });
    await settle();
    const dlg = document.querySelector<HTMLDialogElement>('dialog[open]');
    expect(dlg).not.toBeNull();
    const agentRadio = [...dlg!.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => r.textContent?.trim() === 'Agent');
    expect(agentRadio?.getAttribute('aria-checked')).toBe('true');

    // Closing the dialog removes ?add= from the URL so reloading does not reopen it.
    btn('Cancel', dlg!)!.click();
    await settle();
    expect(document.querySelector('dialog[open]')).toBeNull();
    expect(router.route.query.get('add')).toBeNull();

    // Opening via ?add=model and then navigating Back (dropping ?add=) closes the URL-opened dialog.
    router.go('/benchmarks/bn_9f03?add=model', { replace: true });
    await settle();
    expect(document.querySelector('dialog[open]')).not.toBeNull();
    router.go('/benchmarks/bn_9f03', { replace: true });
    await settle();
    expect(document.querySelector('dialog[open]')).toBeNull();
  });

  it('respects close confirmation when Back drops ?add= during an active upload, and gates Add entry on post-load resync outage', async () => {
    // Part 1: useAddQuery consults canClose before closing a URL-opened dialog.
    router.route = parse('#/benchmarks/bn_9f03?add=external');
    let allowClose = false;
    let closeCalls = 0;
    let state!: ReturnType<typeof useAddQuery>;
    c = render(() => {
      state = useAddQuery(
        () => router.go('/benchmarks/bn_9f03', { replace: true }),
        () => {
          closeCalls++;
          return allowClose;
        },
      );
      return <div />;
    }, target);
    flushSync();
    expect(state.addOpen.get()).toBe(true);
    router.route = parse('#/benchmarks/bn_9f03');
    flushSync();
    expect(closeCalls).toBe(1);
    expect(state.addOpen.get()).toBe(true);
    // Once idle (allowClose = true), another ?add= removal closes cleanly.
    router.route = parse('#/benchmarks/bn_9f03?add=external');
    flushSync();
    allowClose = true;
    router.route = parse('#/benchmarks/bn_9f03');
    flushSync();
    expect(state.addOpen.get()).toBe(false);
    c.unmount();
    c = undefined;

    // Part 2: post-load SSE reconnect + network resync failure disables Add entry across header, leaderboard, and dialog.
    let sseHandlers: BenchEventHandlers = {};
    const realEvents = hub.api.benchEvents.bind(hub.api);
    vi.spyOn(hub.api, 'benchEvents').mockImplementation((id, h) => {
      sseHandlers = h;
      return realEvents(id, h);
    });
    router.route = parse('#/benchmarks/bn_9f03?add=model');
    c = render(() => <BenchmarkPage />, target);
    flushSync();
    await settle();
    const dlg = document.querySelector<HTMLDialogElement>('dialog[open]')!;
    expect(dlg.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled).toBe(false);

    const realEntries = hub.api.benchEntries.bind(hub.api);
    const entriesSpy = vi.spyOn(hub.api, 'benchEntries').mockRejectedValue(new ApiError(0, 'network', 'Failed to fetch'));
    sseHandlers.connection?.('reconnecting');
    await new Promise((r) => setTimeout(r, RESYNC_MS + 40));
    await settle(4);
    expect(dlg.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled).toBe(true);
    for (const b of [...document.querySelectorAll<HTMLButtonElement>('button')].filter((x) => !x.closest('dialog') && x.textContent?.trim() === 'Add entry')) {
      expect(b.disabled).toBe(true);
    }

    // Recovery re-enables Add entry.
    entriesSpy.mockImplementation(realEntries);
    sseHandlers.connection?.('open');
    await new Promise((r) => setTimeout(r, 1100));
    await settle(4);
    expect(dlg.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled).toBe(false);
  });
});
