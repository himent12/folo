// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi, type Scenario } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import Playground from './Playground.svelte';

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
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

/** Opens the Playground on `model` and runs it on the first sample image. */
async function run(scenario: Scenario, model: string) {
  const api = new MockApi(scenario, 0);
  api.imageUrl = async () => 'data:,';
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse(`#/playground?model=${model}`);
  c = mount(Playground, { target });
  (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
}

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

describe('Playground: a failed prediction', () => {
  it('shows the error once, above the image, and moves focus to it', async () => {
    await run('oom', 'm_shapes_s');
    const notice = await until(() => target.querySelector('.notice'), 'the notice');
    // Above the image, where the result would be, not over it: the stage holds only the image. The same at every width.
    expect(target.querySelector('.stage .notice, .stage [role="alert"]')).toBeNull();
    expect(notice.nextElementSibling?.classList.contains('stage')).toBe(true);
    expect(notice.previousElementSibling?.classList.contains('vbar')).toBe(true);
    const title = notice.querySelector<HTMLElement>('.title')!;
    expect(title.textContent).toBe('Prediction failed');
    expect(document.activeElement).toBe(title);
    // One authoritative block: the caption under it adds only which model ran where.
    const foot = target.querySelector('.vfoot')?.textContent?.replace(/\s+/g, ' ').trim();
    expect(foot).not.toContain('Prediction failed');
    expect(foot).not.toContain('out of GPU memory');
    expect(foot).toContain('shapes-s-best on the server');
    expect(foot).not.toContain('Waiting');
    // The detections panel says why it is empty.
    expect(target.querySelector('#pg-panel .placeholder')?.textContent?.trim()).toBe('No detections: the last prediction failed.');
    // The ways out: this browser first (primary), then the smaller model, then Retry.
    expect([...notice.querySelectorAll('button')].map((b) => b.textContent?.trim()).filter((t) => t !== 'Technical details')).toEqual([
      'Run in this browser',
      'Use shapes-n-best',
      'Retry',
    ]);
    // One typeface in the label; the architecture is in its title.
    const smaller = [...notice.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Use shapes-n-best')!;
    expect(smaller.querySelector('.mono')).toBeNull();
    expect(smaller.getAttribute('title')).toBe('shapes-n-best is folo-n: a smaller model of the same classes');
  });

  it('moves focus to the notice without scrolling, and keeps it there when the notice re-renders', async () => {
    const focus = vi.spyOn(HTMLElement.prototype, 'focus');
    await run('oom', 'm_shapes_s');
    const notice = await until(() => target.querySelector<HTMLElement>('.notice'), 'the notice');
    const title = notice.querySelector<HTMLElement>('.title')!;
    expect(document.activeElement).toBe(title);
    const call = focus.mock.calls.find((_, i) => focus.mock.contexts[i] === title);
    expect(call?.[0]).toEqual({ preventScroll: true });
    focus.mockRestore();
    // Whatever re-renders the heading (here, by hand), focus comes back to it instead of falling to the page start.
    const again = title.cloneNode(true) as HTMLElement;
    title.replaceWith(again);
    await until(() => document.activeElement === again, 'focus back on the heading');
  });
});

describe('Playground on a phone', () => {
  const width = window.innerWidth;
  afterEach(() => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width });
  });

  it('puts a failed prediction above the image, so it is in the first screen', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 390 });
    await run('oom', 'm_shapes_s');
    const notice = await until(() => target.querySelector('.notice'), 'the notice');
    expect(notice.nextElementSibling?.classList.contains('stage')).toBe(true);
    expect(notice.previousElementSibling?.classList.contains('vbar')).toBe(true);
  });

  it('opens Thresholds with Confidence as the first row of the sheet', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 390 });
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const btn = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('Thresholds'))!;
    btn.click();
    const sheet = await until(() => target.querySelector('#pg-more'), 'the sheet');
    expect([...sheet.querySelectorAll('.slider label')].map((l) => l.textContent)).toEqual(['Confidence', 'Overlap (IoU)', 'Max detections']);
    // The pinned Confidence under the image steps out while the sheet holds it.
    expect(target.querySelector('.qconf')?.classList.contains('covered')).toBe(true);
  });
});

describe('Playground on a tablet', () => {
  const width = window.innerWidth;
  afterEach(() => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width });
  });

  // Bottom sheets are for phones (DESIGN_DIRECTION 9): round 24's tablet sheet covered Classes and Timing, clipped at
  // the viewport. More settings is a popover anchored under its button; focus moves in, and Escape brings it back.
  it('opens More settings as an anchored popover, not a sheet, and Escape returns focus to the button', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 820 });
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const btn = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('More settings'))!;
    btn.click();
    const pop = await until(() => target.querySelector<HTMLElement>('#pg-more'), 'the popover');
    expect(pop.classList.contains('pop')).toBe(true);
    expect(pop.querySelector('.sheet-handle')).toBeNull();
    await until(() => document.activeElement === pop, 'focus in the popover');
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    flushSync();
    expect(target.querySelector('#pg-more')).toBeNull();
    expect(document.activeElement).toBe(btn);
  });
});

describe('Playground: detections', () => {
  it('shows scores with one precision in the table and the picked line', async () => {
    await run('normal', 'm_hard_s');
    const row = await until(() => target.querySelector<HTMLTableRowElement>('#pg-panel tbody tr'), 'the detections');
    const cell = row.querySelector('.score .num')!.textContent!;
    expect(cell).toMatch(/^0\.\d\d$/);
    row.click();
    flushSync();
    expect(row.getAttribute('aria-label')).toContain(`score ${cell},`);
    expect(target.querySelector('.pick-line')?.textContent).toContain(cell);
  });

  it('draws score bars in neutral ink, with the class hue on the swatch only', async () => {
    await run('normal', 'm_hard_s');
    const row = await until(() => target.querySelector<HTMLTableRowElement>('#pg-panel tbody tr'), 'the detections');
    expect(row.querySelector<HTMLElement>('.s-track span')!.style.background).toBe('');
    expect(row.querySelector<HTMLElement>('.sw')!.style.background).not.toBe('');
  });

  it('describes IoU as suppression, under the one name the phone sheet uses too', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    expect(target.querySelector('.thresholds')?.textContent).toContain('Suppress same-class overlaps above this.');
    expect(target.querySelector('.thresholds')?.textContent).not.toContain('Merge');
    expect([...target.querySelectorAll('.thresholds .slider label')].map((l) => l.textContent)).toEqual(['Confidence', 'Overlap (IoU)', 'Max detections']);
  });

  it('marks the selected row with aria-selected', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelectorAll('#pg-panel tbody tr').length > 1, 'the detections');
    const rows = () => [...target.querySelectorAll<HTMLTableRowElement>('#pg-panel tbody tr')];
    expect(rows().some((r) => r.hasAttribute('aria-selected'))).toBe(false);
    rows()[1]!.click();
    flushSync();
    expect(rows().map((r) => r.getAttribute('aria-selected'))).toEqual(rows().map((_, i) => (i === 1 ? 'true' : null)));
    expect(rows()[1]!.classList.contains('on')).toBe(true);
  });
});

describe('Playground without the in-browser runtime (DOGFOOD-3 S5)', () => {
  async function open(scenario: Scenario) {
    hub.api = new MockApi(scenario, 0);
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    c = mount(Playground, { target });
    return until(() => target.querySelector('[role="radiogroup"][aria-label="Run the model on"]'), 'Run on');
  }
  const option = (g: Element, name: string) => [...g.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === name)!;

  it('offers WebGPU and CPU off, with the reason, where the build has none', async () => {
    const g = await open('no-wasm');
    await until(() => option(g, 'CPU').disabled, 'the CPU choice to turn off');
    expect(option(g, 'Server').disabled).toBe(false);
    expect(option(g, 'WebGPU').disabled).toBe(true);
    expect(target.querySelector('#runon-note')?.textContent).toBe('No in-browser runtime');
    expect(target.querySelector('.runon')?.getAttribute('aria-describedby')).toBe('runon-note');
  });

  it('offers the CPU where the build has it', async () => {
    const g = await open('normal');
    await until(() => target.querySelector('select, .model-pick'), 'the pickers');
    await new Promise((r) => setTimeout(r, 50));
    flushSync();
    expect(option(g, 'CPU').disabled).toBe(false);
    expect(target.querySelector('#runon-note')?.textContent ?? '').not.toContain('runtime');
  });
});
