// @vitest-environment happy-dom
// Generated-scene benchmark source parity: NewBenchDialog creation, capability gating, Benchmarks list filter,
// and Benchmark detail page badge (never misreported as a deleted dataset).
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { renderImage } from '$lib/api/mock/render';
import { BENCH_FEATURE, BENCH_GENERATED_FEATURE, isGeneratedBenchSource, resolveGeneratedShapes } from '$lib/bench/rules';
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

  it('surfaces omitted class names when more than 8 classes are entered instead of silently dropping them on submit', async () => {
    const createSpy = vi.spyOn(hub.api, 'createBench');
    open(NewBenchDialog, { open: true });
    await settle();

    [...document.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Images from"] [role="radio"]')].find((r) =>
      r.textContent?.includes('Generated scenes'),
    )!.click();
    flushSync();

    type(document.querySelector('input[id$="-name"]')!, 'Too many classes');
    // Already has 3 default classes; pasting 6 more brings total to 9 -> 9th ('extra') must be surfaced and block submit.
    type(document.querySelector('input[id$="-gcls"]')!, 'star, ring, cross, hexagon, heart, extra');
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('extra');
    expect(document.querySelector('.chips')?.classList.contains('invalid')).toBe(true);

    document.querySelector<HTMLFormElement>('dialog[open] form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
    expect(createSpy).not.toHaveBeenCalled();

    // Pressing Enter commits the first 8 valid chips, surfaces the left-out name in the hint, and lets the user submit those 8 valid classes.
    const clsInput = document.querySelector<HTMLInputElement>('input[id$="-gcls"]')!;
    clsInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    flushSync();
    expect(document.body.textContent).toContain('extra left out');
    expect(document.querySelector('[role="alert"]')?.textContent?.trim()).toBe('');
    expect(document.querySelector('.chips')?.classList.contains('invalid')).toBe(false);
    const submitBtn = document.querySelector<HTMLButtonElement>('dialog[open] button[type="submit"]')!;
    expect(submitBtn.disabled).toBe(false);
    document.querySelector<HTMLFormElement>('dialog[open] form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
    expect(createSpy).toHaveBeenCalledTimes(1);
    const sent = createSpy.mock.calls[0]![0];
    expect(sent.source.kind === 'generated' ? sent.source.classes : []).toHaveLength(8);
  });

  it('matches server HTTP validation in mock mode (distinct custom case allowed, built-in shape collision and unknown fields rejected)', async () => {
    // Distinct custom names differing only in case are allowed by Rust `shapes_for`.
    const custom = await hub.api.createBench({
      name: 'Custom case bench',
      source: { kind: 'generated', images: 4, classes: ['Bolt', 'bolt'], imgsz: 64, seed: 1 },
    });
    expect(custom.classes).toEqual(['Bolt', 'bolt']);

    // Two classes claiming the same built-in shape (case-insensitive) are rejected.
    await expect(
      hub.api.createBench({
        name: 'Shape collision',
        source: { kind: 'generated', images: 4, classes: ['Circle', 'circle'], imgsz: 64, seed: 1 },
      }),
    ).rejects.toMatchObject({ status: 400, code: 'invalid_body' });

    // Unknown fields inside generated source are rejected with 422 invalid_body (matching #[serde(deny_unknown_fields)]).
    await expect(
      hub.api.createBench({
        name: 'Unknown field',
        source: { kind: 'generated', images: 4, extra_field: true } as never,
      }),
    ).rejects.toMatchObject({ status: 422, code: 'invalid_body' });
  });

  it('keeps Benchmarks desktop fixed column widths compact with bounded Images from truncation', async () => {
    await hub.api.createBench({
      name: 'Round 23 Live QA Generated',
      source: { kind: 'generated', images: 8, classes: ['circle', 'square'], imgsz: 128, seed: 42 },
    });
    open(Benchmarks, {});
    await settle();

    const fixedWidths = [...document.querySelectorAll<HTMLTableCellElement>('table.benches thead th')]
      .map((th) => Number.parseInt(th.style.width, 10))
      .filter((n) => Number.isFinite(n));
    const totalFixed = fixedWidths.reduce((a, b) => a + b, 0);
    // At 1280px desktop with expanded sidebar (~1012px table width), fixed columns leave >= 230px for the Benchmark name.
    expect(totalFixed).toBeLessThanOrEqual(780);

    const srcRow = document.querySelector<HTMLElement>('table.benches tbody tr[data-row] td.src .src-row');
    expect(srcRow?.getAttribute('title')).toContain('generated scenes');
    expect(srcRow?.querySelector('.src-name.truncate')).not.toBeNull();
  });

  it('resolves generated shapes per folo-data rules and draws distinct ring and heart paths in mock mode', async () => {
    expect(resolveGeneratedShapes(['ring', 'heart'])).toEqual(['ring', 'heart']);
    expect(resolveGeneratedShapes(['Circle', 'Bolt', 'bolt'])).toEqual(['circle', 'square', 'triangle']);

    const ellipseCalls: unknown[][] = [];
    const bezierCalls: unknown[][] = [];
    const fillCalls: unknown[][] = [];
    const ctxStub = {
      scale: vi.fn(),
      createLinearGradient: () => ({ addColorStop: vi.fn() }),
      createRadialGradient: () => ({ addColorStop: vi.fn() }),
      fillRect: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      rect: vi.fn(),
      roundRect: vi.fn(),
      ellipse: (...args: unknown[]) => ellipseCalls.push(args),
      bezierCurveTo: (...args: unknown[]) => bezierCalls.push(args),
      fill: (...args: unknown[]) => fillCalls.push(args),
      clip: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    const origOffscreen = globalThis.OffscreenCanvas;
    vi.stubGlobal('OffscreenCanvas', undefined);
    const getCtxSpy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctxStub);
    const toBlobSpy = vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => cb(new Blob(['ok'], { type: 'image/jpeg' })));
    try {
      await renderImage(
        {
          id: 'gen_ring_heart',
          name: 'gen',
          path: 'generated',
          format: 'yolo',
          classes: ['ring', 'heart'],
          classWeights: [1, 1],
          classShape: ['round', 'square'],
          renderShapes: resolveGeneratedShapes(['ring', 'heart']),
          splits: { train: 0, val: 1, test: 0 },
          width: 128,
          height: 128,
          meanBoxes: 2,
          emptyRate: 0,
          sizeRange: [0.2, 0.3],
          brokenLabels: 0,
          createdAgoS: 0,
          style: 'shapes',
          seed: 1,
        },
        {
          id: '1',
          file: '000000.jpg',
          split: 'val',
          width: 128,
          height: 128,
          boxes: [
            { class: 0, x1: 10, y1: 10, x2: 50, y2: 50 },
            { class: 1, x1: 60, y1: 60, x2: 100, y2: 100 },
          ],
        },
        0,
      );
      expect(ellipseCalls.length).toBeGreaterThanOrEqual(2);
      expect(fillCalls.some((c) => c[0] === 'evenodd')).toBe(true);
      expect(bezierCalls.length).toBeGreaterThanOrEqual(2);
    } finally {
      getCtxSpy.mockRestore();
      toBlobSpy.mockRestore();
      vi.stubGlobal('OffscreenCanvas', origOffscreen);
    }
  });
});
