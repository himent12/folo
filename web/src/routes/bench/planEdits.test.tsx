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

  it('engages the unsaved-plan leave guard when clearing a weight is the only edit', async () => {
    await openPlan();
    typeInput(weight('Low light'), '');
    expect(weight('Low light').getAttribute('aria-invalid')).toBe('true');

    router.go('/runs');
    flushSync();
    const dialog = document.querySelector<HTMLDialogElement>('dialog[open]');
    expect(dialog).not.toBeNull();
    expect(dialog!.textContent).toMatch(/Low light: the weight is empty/);
    expect(btn('Save plan', dialog!)!.disabled).toBe(true);
    expect(router.route.path).toBe('/benchmarks/bn_9f03');

    // Dismissing returns the user to the invalid field; Discard clears it and proceeds.
    btn('Discard changes', dialog!)!.click();
    await settle();
    expect(document.querySelector('dialog[open]')).toBeNull();
    expect(router.route.path).toBe('/runs');
  });
});
