// @vitest-environment happy-dom
// New run › Run on (docs/CLOUD.md): this machine or a cloud connection; choosing one reveals its GPU and Stop after,
// the Summary says where and what it costs, and the config carries `compute` only then and only with the feature.
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, hub } from '$lib/api/index.svelte';
import { MockApi, type Scenario } from '$lib/api/mock';
import { RTX_4090 } from '$lib/api/mock/cloud';
import type { TrainConfig } from '$lib/api/types';
import { parse, router } from '$lib/router.svelte';
import RunNew from './RunNew.svelte';

let api: MockApi;
let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

// Svelte's select binding reads the chosen option with `:checked`, which happy-dom does not match on <option>.
const qs = HTMLSelectElement.prototype.querySelector;
HTMLSelectElement.prototype.querySelector = function (this: HTMLSelectElement, sel: string) {
  return sel === ':checked' ? (this.options[this.selectedIndex] ?? null) : qs.call(this, sel);
} as typeof qs;

async function open(scenario: Scenario, query = '') {
  api = new MockApi(scenario, 0);
  api.imageUrl = async () => 'data:,';
  hub.api = api;
  router.route = parse(`#/runs/new${query}`);
  c = mount(RunNew, { target });
  await until(() => (target.querySelector<HTMLInputElement>('input#name')?.value ? true : null), 'the run name');
}

function choose(id: string, value: string) {
  const sel = target.querySelector<HTMLSelectElement>(`select#${id}`)!;
  sel.querySelector<HTMLOptionElement>(`option[value="${CSS.escape(value)}"]`)!.selected = true;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

/** Chooses the demo's RunPod connection and waits for its GPU to be preselected and the cost to be estimated. */
async function onRunPod() {
  await until(() => target.querySelector('select#runon option[value="cc_4d2a"]'), 'the connection');
  choose('runon', 'cc_4d2a');
  await until(() => target.querySelector('#offer-hint .price'), 'the preselected GPU');
  await until(() => target.querySelector('.summary .cost-note'), 'the cost');
}

async function start(): Promise<TrainConfig> {
  const create = vi.spyOn(api, 'createRun');
  target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  await tick();
  await until(() => create.mock.calls.length, 'the request');
  return create.mock.calls[0]![0];
}

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
const summary = (label: string) => {
  const dt = [...target.querySelectorAll('.summary dt')].find((d) => text(d) === label);
  return text(dt?.nextElementSibling);
};

beforeEach(() => {
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  Element.prototype.scrollIntoView ??= () => {};
  vi.spyOn(router, 'go').mockImplementation(() => {});
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('New run › Run on', () => {
  it('has no Run on and sends no compute on a server without the cloud feature', async () => {
    await open('cloud-off');
    await until(() => target.querySelector('select#device'), 'the device');
    expect(target.querySelector('select#runon')).toBeNull();
    expect(await start()).not.toHaveProperty('compute');
  });

  it('offers only this machine without a connection, and says where to connect one', async () => {
    await open('cloud-none');
    const hint = await until(() => target.querySelector('#runon-hint a') && target.querySelector('#runon-hint'), 'the hint');
    expect(text(hint)).toBe('Connect a cloud in Settings to train on a rented GPU.');
    expect(hint.querySelector('a')!.getAttribute('href')).toBe('#/settings?section=cloud');
    expect([...target.querySelectorAll('select#runon option')].map(text)).toEqual(['This machine']);
    expect(target.querySelector('select#offer')).toBeNull();
    const config = await start();
    expect(config).not.toHaveProperty('compute');
    expect(config.device).toBe('auto');
  });

  it('lists a connection whose check failed as off, its error the reason', async () => {
    await open('cloud-trouble');
    const opt = await until(() => target.querySelector<HTMLOptionElement>('select#runon option[value="cc_4d2a"]'), 'the connection');
    expect(opt.disabled).toBe(true);
    expect(text(opt)).toContain('RunPod rejected this API key');
  });

  it('reveals the GPU and Stop after for a connection, the device a fact, and sends compute', async () => {
    await open('normal');
    expect(target.querySelector('select#offer')).toBeNull();
    await onRunPod();
    // The cheapest available offer with the memory these settings need; the one with none available is off.
    const offer = target.querySelector<HTMLSelectElement>('select#offer')!;
    expect(offer.value).toBe('NVIDIA RTX A4000@community');
    expect(text(offer.querySelector('option:checked, option[value="NVIDIA RTX A4000@community"]'))).toBe('RTX A4000 · 16.0 GiB · Community Cloud · $0.17/h');
    expect(offer.querySelector<HTMLOptionElement>('option[value="NVIDIA A100 80GB PCIe@secure"]')!.disabled).toBe(true);
    expect(text(offer.querySelector('option[value="NVIDIA GeForce RTX 4090@community"]'))).toContain('low stock');
    expect([...target.querySelectorAll('#offer-hint .ui-meta > span')].map(text)).toEqual(['$0.17/h', 'Community Cloud']);
    expect(target.querySelector('select#device')).toBeNull();
    expect(text(target.querySelector('.dev-fact'))).toBe('The instance’s GPU');
    const hours = target.querySelector<HTMLInputElement>('input#hours')!;
    expect(Number(hours.value)).toBeGreaterThanOrEqual(1);
    expect(text(target.querySelector('#hours-hint'))).toMatch(/^The run is stopped and the instance terminated after this long at the latest\. It cannot cost more than \$\d+\.\d\d\.$/);
    const runsOn = [...target.querySelectorAll('.summary .rv')].find((r) => text(r.querySelector('.rv-k')) === 'Runs on');
    expect(text(runsOn?.querySelector('.rv-v'))).toBe('RunPod · RTX A4000');
    expect(summary('Est. cost')).toMatch(/^\$\d+\.\d\d–\$\d+\.\d\d$/);
    expect(summary('Limit')).toBe(`$${(0.17 * Number(hours.value)).toFixed(2)} at ${hours.value} h`);
    const config = await start();
    expect(config.compute).toEqual({ connection_id: 'cc_4d2a', offer_id: 'NVIDIA RTX A4000@community', max_hours: Number(hours.value) });
    expect(config.device).toBe('auto');
  });

  it('keeps a time limit set by hand, and refuses one out of range at its field', async () => {
    await open('normal');
    await onRunPod();
    const hours = target.querySelector<HTMLInputElement>('input#hours')!;
    hours.value = '1000';
    hours.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(text(await until(() => target.querySelector('#hours-error'), 'the error'))).toBe('From 0.25 to 720 hours.');
    expect(target.querySelector<HTMLButtonElement>('.start button[type="submit"]')!.disabled).toBe(true);
  });

  it('says a refusal at the control it is about: no capacity at the GPU (the offers read again), a connection at Run on', async () => {
    await open('normal');
    await onRunPod();
    const offers = vi.spyOn(api, 'cloudOffers');
    vi.spyOn(api, 'createRun').mockRejectedValueOnce(new ApiError(409, 'cloud_refused', 'RunPod has no RTX A4000 available right now', 'Pick another GPU.'));
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(text(await until(() => target.querySelector('#offer-error'), 'the GPU error'))).toBe('RunPod has no RTX A4000 available right now.');
    await until(() => offers.mock.calls.length, 'the offers read again');
    // Another GPU answers the refusal.
    choose('offer', RTX_4090.id);
    await until(() => !target.querySelector('#offer-error'), 'the error gone');

    vi.spyOn(api, 'createRun').mockRejectedValueOnce(new ApiError(404, 'connection_not_found', "no cloud connection 'cc_4d2a'"));
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(text(await until(() => target.querySelector('#runon-error'), 'the Run on error'))).toBe("No cloud connection 'cc_4d2a'.");
  });

  it('says a time limit the server refuses at Stop after', async () => {
    await open('normal');
    await onRunPod();
    vi.spyOn(api, 'createRun').mockRejectedValueOnce(new ApiError(400, 'invalid_config', '`compute.max_hours` must be from 0.25 to 720 hours, not 3'));
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(text(await until(() => target.querySelector('#hours-error'), 'the error'))).toBe('Compute.max_hours must be from 0.25 to 720 hours, not 3.');
  });

  it('copies a cloud run with its connection, GPU and limit; one whose GPU is gone trains here, saying why', async () => {
    await open('normal', '?from=run_c3d5');
    await until(() => target.querySelector<HTMLSelectElement>('select#offer')?.value === RTX_4090.id, 'the copied GPU');
    expect(target.querySelector<HTMLSelectElement>('select#runon')!.value).toBe('cc_4d2a');
    expect(target.querySelector<HTMLInputElement>('input#hours')!.value).toBe('2');
    unmount(c!);
    c = undefined;

    api = new MockApi('normal', 0);
    const real = api.cloudOffers.bind(api);
    vi.spyOn(api, 'cloudOffers').mockImplementation(async (...a) => {
      const list = await real(...a);
      return { ...list, offers: list.offers.filter((o) => o.id !== RTX_4090.id) };
    });
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    router.route = parse('#/runs/new?from=run_c3d5');
    c = mount(RunNew, { target });
    const note = await until(() => target.querySelector('#runon-hint')?.textContent?.includes('no longer offers') && target.querySelector('#runon-hint'), 'the note');
    expect(text(note)).toBe('RunPod no longer offers the run’s RTX 4090, so this one trains on this machine.');
    expect(target.querySelector<HTMLSelectElement>('select#runon')!.value).toBe('local');
    expect(target.querySelector('select#offer')).toBeNull();
  });
});
