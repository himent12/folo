// @vitest-environment happy-dom
// Regression tests for benchmark plan editing edge cases found in Live Chromium Playtest Round 6:
// 1. Edits made while Save plan is in flight are preserved in the draft and stay unsaved.
// 2. Clearing a weight as the only edit still engages the unsaved-changes leave guard.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import BenchmarkPage from '../Benchmark';

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
  });
});
