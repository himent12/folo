// @vitest-environment happy-dom
// Generated-scene benchmark source parity: NewBenchDialog creation, capability gating, Benchmarks list filter,
// and Benchmark detail page badge (never misreported as a deleted dataset).
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { BENCH_FEATURE, BENCH_GENERATED_FEATURE, isGeneratedBenchSource } from '$lib/bench/rules';
import { parse, router } from '$lib/router';
import { undescribedFields } from '$lib/ui/fieldA11y';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../framework/testing';
import BenchmarkPage from '../Benchmark';
import Benchmarks from '../Benchmarks';
import NewBenchDialog from './NewBenchDialog';

let target: HTMLElement;
let c: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  hub.features = null;
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  hub.features = null;
  vi.restoreAllMocks();
});

function open<P extends object>(comp: Component<P>, props: P) {
  const Comp = comp as Component<Record<string, unknown>>;
  c = render(() => <Comp {...(props as Record<string, unknown>)} />, target);
  flushSync();
}

async function settle(rounds = 12) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

function type(el: HTMLInputElement | HTMLTextAreaElement, v: string) {
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

describe('Bench generated-scene source parity', () => {
  it('creates a generated-scene benchmark via NewBenchDialog with exact API payload and navigates to it', async () => {
    const createSpy = vi.spyOn(hub.api, 'createBench');
    const goSpy = vi.spyOn(router, 'go').mockImplementation(() => {});

    open(NewBenchDialog, { open: true });
    await settle();

    const genRadio = [...document.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Images from"] [role="radio"]')].find((r) =>
      r.textContent?.includes('Generated scenes'),
    )!;
    expect(genRadio).toBeDefined();
    expect(genRadio.disabled).toBe(false);
    genRadio.click();
    flushSync();

    // Every generated field has accessible label/description wiring.
    expect(undescribedFields(document)).toEqual([]);

    type(document.querySelector('input[id$="-name"]')!, 'Synthetic shapes stress');
    type(document.querySelector('input[id$="-gimg"]')!, '32');
    type(document.querySelector('input[id$="-gseed"]')!, '42');
    [...document.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Image size"] [role="radio"]')].find((r) => r.textContent?.trim() === '480')!.click();
    const varyBox = document.querySelector<HTMLInputElement>('label.aspect-toggle input[type="checkbox"]')!;
    varyBox.click();
    flushSync();

    const form = document.querySelector<HTMLFormElement>('dialog[open] form')!;
    form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();

    expect(createSpy).toHaveBeenCalledTimes(1);
    const body = createSpy.mock.calls[0]![0];
    expect(body).toEqual({
      name: 'Synthetic shapes stress',
      description: '',
      source: {
        kind: 'generated',
        images: 32,
        classes: ['circle', 'square', 'triangle'],
        imgsz: 480,
        seed: 42,
        vary_aspect: true,
      },
    });
    expect('max_images' in body).toBe(false);

    expect(goSpy).toHaveBeenCalledWith(expect.stringMatching(/^\/benchmarks\/bn_[0-9a-f]{12}$/));
    const created = (await hub.api.benches()).find((b) => b.name === 'Synthetic shapes stress')!;
    expect(created).toBeDefined();
    expect(isGeneratedBenchSource(created.source)).toBe(true);
    expect(created.source).toEqual({ kind: 'dataset', dataset_id: 'generated', split: null });
    expect(created.source_name).toBe('generated (32×480px, varied aspect, seed 42)');
    expect(created.images).toBe(32);
  });

  it('disables Generated scenes when the server does not advertise bench_generated_source', async () => {
    hub.features = [BENCH_FEATURE];
    open(NewBenchDialog, { open: true });
    await settle();

    const findGenRadio = () =>
      [...document.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Images from"] [role="radio"]')].find((r) =>
        r.textContent?.includes('Generated scenes'),
      )!;
    expect(findGenRadio().disabled).toBe(true);

    // Advertised via health features -> enabled immediately.
    hub.features = [BENCH_FEATURE, BENCH_GENERATED_FEATURE];
    flushSync();
    expect(findGenRadio().disabled).toBe(false);
  });

  it('shows Generated badge on Benchmark detail (without deleted-since) and filters Generated on Benchmarks list', async () => {
    const created = await hub.api.createBench({
      name: 'Gen Smoke Bench',
      source: { kind: 'generated', images: 12, classes: ['circle', 'star'], imgsz: 320, seed: 7, vary_aspect: false },
    });

    // Benchmark detail page: shows Generated badge and generated scenes label, never "deleted since".
    router.route = parse(`#/benchmarks/${created.id}`);
    open(BenchmarkPage, {});
    await settle(20);
    const bodyText = document.body.textContent ?? '';
    expect(bodyText).toContain('Generated');
    expect(bodyText).toContain('generated (12×320px, seed 7)');
    expect(bodyText).toContain('generated scenes');
    expect(bodyText).not.toContain('deleted since');

    // Benchmarks list: shows Generated filter tab and isolates generated benchmarks cleanly.
    c!.unmount();
    c = undefined;
    router.route = parse('#/benchmarks');
    open(Benchmarks, {});
    await settle(20);
    const genFilter = [...document.querySelectorAll<HTMLButtonElement>('[role="radiogroup"] [role="radio"]')].find((r) =>
      r.textContent?.includes('Generated'),
    )!;
    expect(genFilter).toBeDefined();
    genFilter.click();
    flushSync();
    const names = [...document.querySelectorAll('table.benches tbody tr[data-row] td:first-child a.name')].map((a) => a.textContent?.trim());
    expect(names).toEqual(['Gen Smoke Bench']);
  });
});
