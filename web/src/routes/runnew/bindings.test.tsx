// @vitest-environment happy-dom
// New run's two-way bindings, end to end: each control writes the page's state (what Start sends) and shows what the
// page sets, by typing, picking, sliding, switching and the keyboard. The Svelte suite covers most of them through
// their consequences; these cover the ones it does not (the fw port's `bind:` / `bindable` wiring of every section).
import { flush, tick } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { TrainConfig } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../framework/testing';
import RunNew from '../RunNew';

let api: MockApi;
let target: HTMLElement;
let c: Rendered | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flush();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

/** Mounts New run and waits for its devices and its dataset. */
async function open(query = '') {
  router.route = parse(`#/runs/new${query}`);
  c = render(() => <RunNew />, target);
  await until(() => target.querySelector('select#device option[value="gpu:0"]'), 'the device list');
  await until(() => target.querySelector('input[name="dataset"]:checked'), 'the dataset');
}

const $ = <E extends Element = HTMLElement>(sel: string) => target.querySelector<E>(sel)!;

function type(sel: string, v: string) {
  const el = $<HTMLInputElement>(sel);
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flush();
}

function choose(sel: string, value: string) {
  const el = $<HTMLSelectElement>(sel);
  el.querySelector<HTMLOptionElement>(`option[value="${value}"]`)!.selected = true;
  el.dispatchEvent(new Event('change', { bubbles: true }));
  flush();
}

function key(sel: string, k: string) {
  $(sel).dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
  flush();
}

const checked = (group: string) => target.querySelector(`#${group} [aria-checked="true"]`)?.textContent?.trim();
const row = (k: string) => [...target.querySelectorAll('.summary .rv')].find((r) => r.querySelector('.rv-k')?.textContent === k)?.querySelector('.rv-v')?.textContent;

async function start(): Promise<TrainConfig> {
  const create = vi.spyOn(api, 'createRun');
  target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  await tick();
  await until(() => create.mock.calls.length, 'the request');
  return create.mock.calls[0]![0];
}

beforeEach(() => {
  // New run keeps a draft for the tab (runnew/draft): every test starts from a fresh tab.
  sessionStorage.clear();
  api = new MockApi('normal', 0);
  api.imageUrl = async () => 'data:,';
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  Element.prototype.scrollIntoView ??= () => {};
  vi.spyOn(router, 'go').mockImplementation(() => {});
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('New run: bindings', () => {
  it('keeps a typed run name through dataset and size changes, and sends it', async () => {
    await open();
    type('input#name', 'my-run');
    $<HTMLInputElement>('input[name="dataset"]:not(:checked)').click();
    $<HTMLInputElement>('input[name="arch"][value="folo-m"]').click();
    flush();
    expect($<HTMLInputElement>('input#name').value).toBe('my-run');
    expect((await start()).name).toBe('my-run');
  });

  it('is one dataset in the phone select and the cards, whichever is used', async () => {
    await open();
    const other = $<HTMLInputElement>('input[name="dataset"]:not(:checked)').value;
    choose('select#dataset-pick', other);
    expect($<HTMLInputElement>(`input[name="dataset"][value="${other}"]`).checked).toBe(true);
    expect($(`input[name="dataset"][value="${other}"]`).closest('.card')!.classList.contains('on')).toBe(true);
    const back = $<HTMLInputElement>('input[name="dataset"]:not(:checked)');
    back.click();
    flush();
    expect($<HTMLSelectElement>('select#dataset-pick').value).toBe(back.value);
    expect((await start()).dataset_id).toBe(back.value);
  });

  it('moves the size with the arrow keys on the phone pick, the table rows following', async () => {
    const width = window.innerWidth;
    const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });
    setWidth(390);
    try {
      await open();
      expect(checked('arch-pick')).toBe('S');
      key('#arch-pick', 'ArrowRight');
      expect(checked('arch-pick')).toBe('M');
      expect($<HTMLInputElement>('input[name="arch"]:checked').value).toBe('folo-m');
      key('#arch-pick', 'Home');
      expect(checked('arch-pick')).toBe('N');
      expect(row('Model')).toBe('folo-n');
      expect((await start()).arch).toBe('folo-n');
    } finally {
      setWidth(width);
    }
  });

  it('moves batch and image size with the keyboard; a size picked by hand stays when the dataset changes', async () => {
    const real = api.datasetImages.bind(api);
    vi.spyOn(api, 'datasetImages').mockImplementation(async (id, q) => {
      const r = await real(id, q);
      return { ...r, images: r.images.map((i) => ({ ...i, width: 320, height: 320 })) };
    });
    await open('?dataset=ds_shapes');
    await until(() => checked('imgsz') === '320', 'the dataset’s own size');
    key('#batch', 'ArrowLeft');
    expect(checked('batch')).toBe('8');
    expect(row('Batch')).toBe('8');
    // By hand: the size is chosen, and no longer follows the dataset.
    [...target.querySelectorAll<HTMLElement>('#imgsz [role="radio"]')].find((b) => b.textContent?.trim() === '480')!.click();
    flush();
    expect(row('Image size')).toBe('480 px');
    $<HTMLInputElement>('input[name="dataset"]:not([value="ds_shapes"])').click();
    flush();
    await new Promise((r) => setTimeout(r, 50));
    flush();
    expect(checked('imgsz')).toBe('480');
    const config = await start();
    expect([config.batch, config.imgsz]).toEqual([8, 480]);
  });

  it('sends the optimizer, learning rate and seed as set in Advanced', async () => {
    await open();
    const adv = $<HTMLDetailsElement>('#sec-train details.advanced');
    adv.open = true;
    adv.dispatchEvent(new Event('toggle'));
    flush();
    choose('select#opt', 'sgd');
    type('input#lr0', '0.02');
    type('input#seed', '7');
    expect(row('Optimizer')).toBe('SGD');
    const config = await start();
    expect([config.optimizer, config.lr0, config.seed]).toEqual(['sgd', 0.02, 7]);
  });

  // Sweep 4 #330: a seed over 2^32 − 1 was refused, though the trainer takes any u64; past 2^53 − 1 a browser's number
  // is no longer the seed typed, so that is the limit, and the error names it.
  it('sends any seed a browser holds exactly, and names the limit past it', async () => {
    await open();
    const adv = $<HTMLDetailsElement>('#sec-train details.advanced');
    adv.open = true;
    adv.dispatchEvent(new Event('toggle'));
    flush();
    type('input#seed', '9007199254740992');
    expect($('#seed-error').textContent).toBe('A whole number from 0 to 9,007,199,254,740,991.');
    type('input#seed', '-1');
    expect($('#seed-error').textContent).toBe('A whole number from 0 to 9,007,199,254,740,991.');
    type('input#seed', '9007199254740991');
    expect(target.querySelector('#seed-error')).toBeNull();
    type('input#seed', '4294967296');
    expect(target.querySelector('#seed-error')).toBeNull();
    expect((await start()).seed).toBe(4294967296);
  });

  // #354 review: a number field rounds what is typed before it is checked, so 9007199254740990.5 went out as …990.
  // The seed is the text as typed (lib/seed, rule A): digits only.
  it('refuses a seed that is not digits as typed, and sends nothing', async () => {
    await open();
    const adv = $<HTMLDetailsElement>('#sec-train details.advanced');
    adv.open = true;
    adv.dispatchEvent(new Event('toggle'));
    flush();
    const create = vi.spyOn(api, 'createRun');
    for (const t of ['9007199254740990.5', '1e3', '5.0', '+5', '1,000', '']) {
      type('input#seed', t);
      expect($('#seed-error').textContent).toBe('A whole number from 0 to 9,007,199,254,740,991.');
    }
    type('input#seed', '9007199254740990.5');
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await tick();
    expect(create).not.toHaveBeenCalled();
    type('input#seed', ' 007 ');
    expect(target.querySelector('#seed-error')).toBeNull();
    expect((await start()).seed).toBe(7);
  });

  it('says the learning rate is not used while the optimizer is Auto, which picks its own', async () => {
    await open();
    const hint = () => {
      const id = $<HTMLInputElement>('input#lr0').getAttribute('aria-describedby');
      return id ? target.querySelector(`#${id}`)?.textContent?.trim() : undefined;
    };
    expect(hint()).toBe('Not used with Auto, which picks 0.01. Choose an optimizer to set it.');
    choose('select#opt', 'adamw');
    expect(hint()).toBe('Warmup 3 epochs, then linear to 1%.');
  });

  it('edits the augmentation from its sliders, and Defaults puts every one back', async () => {
    await open();
    $<HTMLButtonElement>('.aug-toggle').click();
    flush();
    const slider = (label: string) => [...target.querySelectorAll('#sec-aug .slider')].find((s) => s.querySelector('label')?.textContent === label)!;
    const slide = (label: string, v: string) => {
      const input = slider(label).querySelector<HTMLInputElement>('input[type="range"]')!;
      input.value = v;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flush();
    };
    slide('Mosaic', '0.5');
    slide('Colour jitter (saturation)', '0.3');
    expect(slider('Mosaic').querySelector('output')?.textContent).toBe('0.50');
    expect([...target.querySelectorAll('.aug-line.wide-only .changed')].map((s) => s.textContent)).toEqual(['mosaic 0.5', 'colour 0.3']);
    type('input#close', '3');
    const create = vi.spyOn(api, 'createRun');
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await until(() => create.mock.calls.length, 'the request');
    const sent = create.mock.calls[0]![0].augment;
    expect([sent.mosaic, sent.hsv[1], sent.close_mosaic]).toEqual([0.5, 0.3, 3]);
    // Defaults: the settings, the line and close mosaic (following the epochs again) back as they were.
    [...target.querySelectorAll<HTMLButtonElement>('#sec-aug .aug-actions button')].find((b) => b.textContent?.trim() === 'Defaults')!.click();
    flush();
    expect(slider('Mosaic').querySelector<HTMLInputElement>('input')!.value).toBe('1');
    expect(slider('Colour jitter (saturation)').querySelector<HTMLInputElement>('input')!.value).toBe('0.7');
    expect(target.querySelectorAll('.aug-line.wide-only .changed')).toHaveLength(0);
    type('input#epochs', '8');
    expect($<HTMLInputElement>('input#close').value).toBe('4');
  });

  it('opens the Summary sheet from the dock and closes it with Keep editing', async () => {
    await open();
    const dialog = $<HTMLDialogElement>('dialog');
    expect(dialog.open).toBe(false);
    $<HTMLButtonElement>('.dock-est').click();
    flush();
    expect(dialog.open).toBe(true);
    [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Keep editing')!.click();
    flush();
    expect(dialog.open).toBe(false);
    // A row of the sheet leads to its field, closing the sheet on the way.
    $<HTMLButtonElement>('.dock-est').click();
    flush();
    [...dialog.querySelectorAll<HTMLButtonElement>('.rv')].find((b) => b.querySelector('.rv-k')?.textContent === 'Epochs')!.click();
    flush();
    expect(dialog.open).toBe(false);
  });

  it('keeps the steps line open in the rail and the sheet alike (one state for both)', async () => {
    await open();
    const rail = $<HTMLButtonElement>('.summary button.steps');
    rail.click();
    flush();
    expect(document.getElementById('rail-steps-why')).not.toBeNull();
    expect(document.getElementById('sheet-steps-why')).not.toBeNull();
    $<HTMLButtonElement>('dialog button.steps').click();
    flush();
    expect(document.getElementById('rail-steps-why')).toBeNull();
    expect(rail.getAttribute('aria-expanded')).toBe('false');
  });
});

describe('New run: Start while the user moves on (round s2-runs)', () => {
  /** Starts with the server's answer held back; `answer()` lets it through. */
  async function startHeld() {
    let answer!: () => Promise<void>;
    const real = api.createRun.bind(api);
    vi.spyOn(api, 'createRun').mockImplementation((c) => new Promise((ok, fail) => (answer = () => real(c).then(ok, fail))));
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await tick();
    await until(() => answer, 'the request');
    return answer;
  }
  const wentToRun = () => vi.mocked(router.go).mock.calls.some(([to]) => /^\/runs\/(?!new)/.test(String(to)));
  /**
   * Lets the held answer through and waits for the form to have handled it: its success toast, which the form shows
   * in the same continuation just before it decides whether to open the run.
   */
  async function answered(answer: () => Promise<void>) {
    const shown = vi.spyOn(toast, 'success');
    await answer();
    await until(() => shown.mock.calls.length, 'the start’s toast');
    flush();
  }

  it('opens the new run when the form is still showing', async () => {
    await open();
    await answered(await startHeld());
    expect(wentToRun()).toBe(true);
  });

  it('leaves the user where they went when they left before the server answered (s2-runs-2)', async () => {
    await open();
    const answer = await startHeld();
    // Back, or a click on Datasets, while the request is out.
    router.route = parse('#/datasets');
    flush();
    await answered(answer);
    expect(wentToRun()).toBe(false);
  });

  it('leaves the user where they went while that screen’s chunk still loads (review of s2-runs-2)', async () => {
    await open();
    const answer = await startHeld();
    // App keeps the route on display (and the form) until the next screen's chunk is there (App `holdWhile`).
    const release = router.holdWhile((to) => to.name === 'datasets');
    try {
      router.route = parse('#/datasets');
      flush();
      expect(router.route.name).toBe('run-new');
      expect(target.querySelector('form')).not.toBeNull();
      await answered(answer);
      expect(wentToRun()).toBe(false);
    } finally {
      release();
    }
  });

  it('a start answered after the user left and opened New run again keeps the new form and its draft', async () => {
    await open();
    const answer = await startHeld();
    // Away and back to the same link before the answer: the form on screen is a new one, with edits of its own.
    c!.unmount();
    await open();
    type('input#name', 'second-form');
    await answered(answer);
    expect(wentToRun()).toBe(false);
    await until(() => {
      const draft = sessionStorage.getItem('folo.runNew.draft');
      return draft && JSON.parse(draft).name === 'second-form';
    }, 'the second form’s debounced draft');
  });

  it('clears its own draft once started', async () => {
    await open();
    type('input#name', 'first-form');
    expect(sessionStorage.getItem('folo.runNew.draft')).not.toBeNull();
    await answered(await startHeld());
    expect(sessionStorage.getItem('folo.runNew.draft')).toBeNull();
  });

  it('refuses a learning rate the server reads as 0 or 10 (32-bit), s2-runs-3', async () => {
    await open();
    for (const [v, bad] of [['1e-46', true], ['9.9999999999', true], ['1e-30', false], ['0.01', false]] as const) {
      type('input#lr0', v);
      expect($<HTMLInputElement>('input#lr0').getAttribute('aria-invalid'), v).toBe(bad ? 'true' : null);
    }
  });

  it('keeps a hand-typed 0 -> 0.25 cloud time limit without snapping back to the default when the estimate updates', async () => {
    await open();
    await until(() => target.querySelectorAll('select#runon option').length > 1, 'cloud connection options');
    const connId = target.querySelectorAll<HTMLOptionElement>('select#runon option')[1]!.value;
    choose('select#runon', connId);
    const hours = await until(() => target.querySelector<HTMLInputElement>('input#hours'), 'Stop after input');
    await until(() => Number(hours.value) >= 1, 'default stop-after limit');
    // Wait for the initial server estimate to settle so typing 0 triggers estimating false -> true.
    await new Promise((r) => setTimeout(r, 400));
    flush();

    // Simulate a real browser microtask checkpoint between bindValue's direct input listener and delegated oninput:
    // stopPropagation keeps delegated oninput from running before flush(), exercising editHours's synchronous guard.
    hours.value = '0';
    hours.dispatchEvent(new Event('input', { bubbles: false }));
    flush();
    expect(hours.value).toBe('0');
    expect(hours.getAttribute('aria-invalid')).toBe('true');

    hours.value = '0.25';
    hours.dispatchEvent(new Event('input', { bubbles: true }));
    flush();
    expect(hours.value).toBe('0.25');
    expect(hours.getAttribute('aria-invalid')).toBeNull();
    const sent = await start();
    expect(sent.compute?.max_hours).toBe(0.25);
  });
});
