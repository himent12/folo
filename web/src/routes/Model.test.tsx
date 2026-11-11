// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { ModelEval } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Model from './Model';

/** Port of `mount(Model, { target })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof Model, o: { target: HTMLElement }) => render(() => <Model />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

const text = (sel: string) => target.querySelector(sel)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

/** The model page, once its evaluation (the per-class footer and the matrix's notes) and its run's dataset are in. */
async function open(id: string) {
  router.route = parse(`#/models/${id}`);
  c = mount(Model, { target });
  await until(() => target.querySelector('#per-class .foot') && target.querySelector('#cm .cm-notes'), 'the evaluation');
}

describe('Model: what was evaluated', () => {
  it('names the dataset and the validation split, and the matrix its own thresholds apart from precision and recall', async () => {
    await open('m_fast_n');
    await until(() => text('#pr-title').includes('fasteners'), "the run's dataset");
    expect(target.querySelector('[role="note"][aria-label="Evaluation split"]')).toBeNull();
    expect(text('#pr-title .meta')).toBe('fasteners · validation split · IoU 0.5');
    // The matrix's header and notes name its own thresholds: the KPIs above are at another operating point.
    expect(text('#cm-title .meta')).toContain('conf 0.25 · IoU 0.45');
    expect(text('#cm .cm-notes')).toContain('conf 0.25, IoU 0.45');
    expect(text('#cm .cm-notes')).not.toContain('IoU 0.5,');
    const conf = (await hub.api.modelEval('m_fast_n')).best_f1_conf!;
    expect(text('#cm .how-body')).toContain(`conf ${conf.toFixed(2)} (best F1), IoU 0.50`);
  });

  it('says once that a train-only dataset was evaluated on its training split, and never calls it validation', async () => {
    await open('m_mini_n');
    const notes = target.querySelectorAll('[role="note"][aria-label="Evaluation split"]');
    expect(notes).toHaveLength(1);
    // An imported file: its stored numbers are not this evaluation's (see 'an imported file' below), so the strip
    // names no split; the curves, the per-class table and Details do.
    expect(target.querySelector('section.kpis')?.getAttribute('aria-label')).toBe('Metrics stored with the file');
    expect(text('#pr-title .meta')).toBe('fasteners-mini · training split · IoU 0.5');
    expect(text('#per-class .foot')).toContain('24 training images of fasteners-mini');
    const dd = [...target.querySelectorAll('#details .kv')].find((kv) => kv.querySelector('dt')?.textContent === 'Evaluated on')?.querySelector('dd');
    expect(dd?.querySelector('a')?.getAttribute('href')).toContain('/datasets/ds_fasteners_mini');
    // "Validation" is said only where the notice says there is none.
    expect(target.textContent).not.toMatch(/validation (images of|split of|metrics)/i);
  });

  it("keeps an older server's evaluation honest: the run's dataset, the split and the matrix's thresholds not reported", async () => {
    const api = hub.api as MockApi;
    const real = api.modelEval.bind(api);
    api.modelEval = async (id: string) => {
      const { dataset_id, dataset_name, split, split_fallback, confusion_conf, confusion_iou, ...old } = await real(id);
      void [dataset_id, dataset_name, split, split_fallback, confusion_conf, confusion_iou];
      return old as ModelEval;
    };
    await open('m_fast_n');
    await until(() => text('#pr-title .meta') === 'fasteners · IoU 0.5', "the run's dataset");
    expect(target.querySelector('[role="note"][aria-label="Evaluation split"]')).toBeNull();
    expect(text('#per-class .foot')).not.toContain('validation');
    expect(text('#cm .cm-notes')).toContain('does not report');
    expect(text('#cm .cm-notes')).not.toContain('conf 0.25');
  });
});

describe('Model on a phone: the work before the summary', () => {
  const width = window.innerWidth;
  const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });
  afterEach(() => setWidth(width));

  it('puts the section picker first and opens every section, Curves included, on one folded line', async () => {
    setWidth(390);
    router.route = parse('#/models/m_fast_n');
    c = mount(Model, { target });
    await until(() => target.querySelector('.phone-pick [role="tab"]'), 'the section picker');
    // Nothing between the header and the picker: the summary heads its section, inside the tab panel.
    const panel = target.querySelector('#model-sec-panel')!;
    const kpi = target.querySelector('section.kpi-grid')!;
    expect(panel.contains(kpi)).toBe(true);
    expect(panel.firstElementChild).toBe(kpi);
    expect(target.querySelector('section.kpis')).toBeNull();
    // Curves: the headline on one line too, so the chart leads the section.
    expect(kpi.querySelectorAll('.kg-cell')).toHaveLength(0);
    expect(kpi.querySelector('.kpi-line')).not.toBeNull();
    // Classes: the headline on one line that unfolds the same four cells.
    target.querySelector<HTMLButtonElement>('#model-sec-classes')!.click();
    flushSync();
    const line = target.querySelector<HTMLButtonElement>('section.kpi-grid .kpi-line')!;
    expect(line.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelectorAll('section.kpi-grid .kg-cell')).toHaveLength(0);
    line.click();
    flushSync();
    expect(line.getAttribute('aria-expanded')).toBe('true');
    expect(target.querySelectorAll('section.kpi-grid .kg-cell')).toHaveLength(4);
  });
});

describe('Model: no dataset to evaluate on', () => {
  it('says it is not evaluated, with no Retry and no class dump', async () => {
    const api = hub.api as MockApi;
    api.modelEval = async () => {
      throw new ApiError(404, 'no_matching_dataset', "no registered dataset has this model's 80 classes (person, bicycle, car, motorcycle, …)", 'register a dataset');
    };
    router.route = parse('#/models/m_coco_s');
    c = mount(Model, { target });
    await until(() => text('#pr-title').length && target.querySelector('.empty'), 'the empty state');
    const panel = target.querySelector('.empty')!.textContent!;
    expect(panel).toContain('Not evaluated');
    expect(panel).toContain('80 classes');
    expect(panel).not.toContain('person');
    expect([...target.querySelectorAll('button')].some((b) => b.textContent?.trim() === 'Retry')).toBe(false);
    expect(target.textContent).toContain('Not evaluated: no registered dataset has its classes');
    expect(target.textContent).not.toContain('split not reported');
  });
});

describe('Model: an imported file', () => {
  // models F15: the list put an imported checkpoint under "split unknown" while its page said its numbers were on the
  // validation split of shapes (that was the server's own evaluation, the curves, not the numbers stored in the file).
  it("does not give the file's stored metrics the curves' dataset and split", async () => {
    await open('m_mini_n');
    const strip = target.querySelector('section.kpis')!;
    expect(strip.getAttribute('aria-label')).toBe('Metrics stored with the file');
    strip.querySelector<HTMLButtonElement>('.k-help button, button.k-help, .k-help')!.click();
    flushSync();
    const help = await until(() => target.querySelector('#metric-help'), 'the help');
    expect(help.textContent).toContain('Not recorded: these numbers came with the file');
    expect(help.textContent).toContain("this server's own evaluation, on fasteners-mini · training split");
  });
});
