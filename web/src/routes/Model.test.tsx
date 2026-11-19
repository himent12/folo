// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { ModelEval } from '$lib/api/types';
import { metric } from '$lib/format';
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

/** The model page, once its evaluation (the per-class table's footer, in the curves' panel, and the matrix's notes) and its run's dataset are in. */
async function open(id: string) {
  router.route = parse(`#/models/${id}`);
  c = mount(Model, { target });
  await until(() => target.querySelector('#pr .foot') && target.querySelector('#cm .cm-notes'), 'the evaluation');
}

describe('Model: what was evaluated', () => {
  it('names the dataset and the validation split, and the matrix its own thresholds apart from precision and recall', async () => {
    await open('m_fast_n');
    await until(() => text('#pr-title').includes('fasteners'), "the run's dataset");
    expect(target.querySelector('[role="note"][aria-label="Evaluation split"]')).toBeNull();
    expect(text('#pr-title .meta')).toBe('fasteners · validation split · IoU 0.5');
    // The run's dataset is registered: Evaluated on links it.
    const runDs = (await hub.api.run((await hub.api.model('m_fast_n')).run_id!)).config.dataset_id;
    const dd = [...target.querySelectorAll('#details .kv')].find((kv) => kv.querySelector('dt')?.textContent === 'Evaluated on')?.querySelector('dd');
    await until(() => dd?.querySelector('a'), 'the link');
    expect(dd?.querySelector('a')?.getAttribute('href')).toBe(`#/datasets/${runDs}`);
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
    expect(text('#pr .foot')).toContain('24 training images of fasteners-mini');
    const dd = [...target.querySelectorAll('#details .kv')].find((kv) => kv.querySelector('dt')?.textContent === 'Evaluated on')?.querySelector('dd');
    expect(dd?.querySelector('a')?.getAttribute('href')).toContain('/datasets/ds_fasteners_mini');
    // Details (design/UNIFIED.md §4 Model): three columns of label / value rows, the latency among them.
    expect(target.querySelectorAll('#details .col')).toHaveLength(3);
    const labels = [...target.querySelectorAll('#details .kv dt')].map((d) => d.textContent);
    expect(labels).toContain('Reference latency');
    expect(labels).not.toContain('Architecture latency');
    // "Validation" is said only where the notice says there is none.
    expect(target.textContent).not.toMatch(/validation (images of|split of|metrics)/i);
  });

  it("never labels an imported file's curves with the AP50 stored in the file", async () => {
    const api = hub.api as MockApi;
    const real = api.modelEval.bind(api);
    // This server measured nothing on its dataset; the file says it scored well elsewhere.
    api.modelEval = async (id: string) => {
      const e = await real(id);
      return { ...e, pr: e.pr.map((c) => ({ ...c, ap50: 0 })), per_class: e.per_class.map((c) => ({ ...c, map50: 0 })) };
    };
    await open('m_mini_n');
    expect((await api.model('m_mini_n')).metrics!.map50).toBeGreaterThan(0);
    const all = [...target.querySelectorAll('#pr .item')].find((i) => i.textContent?.includes('All classes'));
    expect(all?.querySelector('.num')?.textContent).toBe(metric(0));
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
    expect(text('#pr .foot')).not.toContain('validation');
    expect(text('#cm .cm-notes')).toContain('does not report');
    expect(text('#cm .cm-notes')).not.toContain('conf 0.25');
  });

  // Sweep 4 (#319): /eval names a run's removed dataset by the id it had; Evaluated on named it with a link that opened
  // "Dataset not found".
  it("names a run's removed dataset without linking to it", async () => {
    const api = hub.api as MockApi;
    const real = api.modelEval.bind(api);
    api.modelEval = async (id: string) => ({ ...(await real(id)), dataset_id: 'ds_gone', dataset_name: 'fasteners-old' });
    api.dataset = async () => {
      throw new ApiError(404, 'dataset_not_found', 'No dataset with id ds_gone.');
    };
    await open('m_fast_n');
    const dd = await until(
      () => [...target.querySelectorAll('#details .kv')].find((kv) => kv.querySelector('dt')?.textContent === 'Evaluated on')?.querySelector('dd'),
      'Evaluated on',
    );
    await until(() => dd.textContent?.includes('fasteners-old'), 'the evaluated dataset');
    expect(dd.querySelector('a')).toBeNull();
  });

  // Review of #335: removing the dataset leaves the models as they were, so nothing the page read about the model
  // changes; its datasets are read on their own, and Evaluated on (and Source) stop linking with the name as plain text.
  it('stops linking the evaluated dataset when it is removed while the page is open', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const api = hub.api as MockApi;
      await open('m_fast_n');
      const kv = (label: string) => [...target.querySelectorAll('#details .kv')].find((k) => k.querySelector('dt')?.textContent === label)?.querySelector('dd');
      const evald = await until(() => kv('Evaluated on')?.querySelector('a[href^="#/datasets/"]'), 'the link');
      const name = evald.textContent;
      await until(() => kv('Source')?.querySelector('a[href^="#/datasets/"]'), "the run's dataset link");
      const dsId = (await api.run((await api.model('m_fast_n')).run_id!)).config.dataset_id;
      await api.deleteDataset(dsId);
      await vi.advanceTimersByTimeAsync(5000);
      await until(() => !kv('Evaluated on')?.querySelector('a') && !kv('Source')?.querySelector('a[href^="#/datasets/"]'), 'no dataset link');
      expect(kv('Evaluated on')?.textContent).toContain(name);
    } finally {
      vi.useRealTimers();
    }
  });
});

// s4-models-1 (and its reviews): "Trained for" says the epochs the run finished, never the planned count: a completed
// or stopped run's checkpoint-backed `Run.epoch`, else its history where `Run.epoch` agrees; unknown, it says the plan.
describe('Model: how long it trained', () => {
  /** m_fast_n's run as `status` at `epoch`, with a history of `epochs` validated epochs (rejected when null). */
  function runAt(status: 'stopped' | 'failed', epoch: number, epochs: number | null) {
    const api = hub.api as MockApi;
    const run = api.run.bind(api);
    api.run = async (id: string) => ({ ...(await run(id)), status, epoch });
    const real = api.runMetrics.bind(api);
    api.runMetrics = async (id, s, every) => {
      if (epochs === null) throw new ApiError(500, 'internal', 'boom');
      const h = await real(id, s, every);
      return { ...h, epochs: h.epochs.slice(0, epochs) };
    };
  }

  // Re-review 1: a crash after epoch 3's `last.folo`, before its history line; the server's `Run.epoch` follows the
  // checkpoint (runs/history.rs reconcile), the history stays at 2.
  it('says how many of the planned epochs a stopped run finished, from its checkpoint-backed epoch over a lagging history', async () => {
    const tw = (await hub.api.model('m_fast_n')).trained_with!;
    runAt('stopped', 3, 2);
    await open('m_fast_n');
    const shown = await until(() => text('#model-trained-for dd').includes(' of ') && text('#model-trained-for dd'), 'the epochs trained');
    expect(shown).toBe(`3 of ${tw.epochs.toLocaleString('en-US')} epochs at ${tw.imgsz} px`);
  });

  it('labels the plan as planned when a failed run’s history does not load, or its epoch is past the history', async () => {
    const tw = (await hub.api.model('m_fast_n')).trained_with!;
    const planned = `${tw.epochs.toLocaleString('en-US')} epochs planned at ${tw.imgsz} px`;
    runAt('failed', 2, null);
    await open('m_fast_n');
    await until(() => text('#pr-title').includes('fasteners'), "the run's dataset");
    expect(text('#model-trained-for dd')).toBe(planned);
    unmount(c!);
    hub.api = new MockApi('normal', 0);
    runAt('failed', 3, 2);
    await open('m_fast_n');
    await until(() => text('#pr-title').includes('fasteners'), "the run's dataset");
    expect(text('#model-trained-for dd')).toBe(planned);
  });
});

// Re-review 1 (rev_user 4): the folder of the model's dataset is gone; the server's hint names endpoints and goes with
// the details, so the page says the way back itself.
describe('Model: evaluation when the dataset folder is gone', () => {
  it('shows how to recover, with Retry', async () => {
    const api = hub.api as MockApi;
    api.modelEval = () =>
      Promise.reject(
        new ApiError(
          404,
          'path_not_found',
          "the folder of dataset 'fasteners' this model trained on is gone: 'D:/data/fasteners' does not exist",
          'move the folder back (the model page evaluates it again), or validate it on another registered dataset with POST /api/val',
          'GET /api/models/m_fast_n/eval',
        ),
      );
    router.route = parse('#/models/m_fast_n');
    c = mount(Model, { target });
    const box = await until(() => [...target.querySelectorAll('#pr *')].find((n) => n.textContent?.includes('Retry')) && target.querySelector('#pr'), 'the error');
    const shown = box.textContent?.replace(/\s+/g, ' ') ?? '';
    expect(shown).toContain('is gone');
    expect(shown).toContain('Move the dataset’s folder back where it was, then Retry.');
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

describe('Model: one look (design/UNIFIED.md §4 Model)', () => {
  it('opens on a strip of five tiles: the headline, mAP50, the precision / recall pair, the parameters and the file', async () => {
    await open('m_fast_n');
    const tiles = [...target.querySelectorAll('section.kpis .kpi')];
    expect(tiles.map((t) => t.querySelector('.k-label')?.textContent?.trim())).toEqual(['mAP50-95', 'mAP50', 'Precision / Recall', 'Parameters', 'File size']);
    expect(tiles[0]!.classList.contains('hero')).toBe(true);
    const m = await hub.api.model('m_fast_n');
    expect(tiles[2]!.querySelector('.k-value')?.textContent?.replace(/\s+/g, ' ').trim()).toBe(`${metric(m.metrics!.precision)} / ${metric(m.metrics!.recall)}`);
    // The figure, then its unit one step smaller; GFLOPs and the file's format as captions.
    expect(tiles[3]!.querySelector('.k-value .unit')?.textContent?.trim()).toBe('M');
    expect(tiles[3]!.querySelector('.k-sub')?.textContent).toContain('GFLOPs');
    expect(tiles[4]!.querySelector('.k-value .unit')?.textContent?.trim()).toBe('MiB');
    expect(tiles[4]!.querySelector('.k-sub')?.textContent).toContain('.folo');
    // The headline names what it was measured on, in few words.
    await until(() => tiles[0]!.querySelector('.k-sub')?.textContent?.includes('fasteners · val'), 'the headline caption');
  });

  it('puts the per-class table in the curves panel as their legend, and Details in one full-width panel', async () => {
    await open('m_fast_n');
    // No separate Per class panel and no folded or sticky rail: the table, its footer and the trace controls are the
    // curves panel's.
    expect(target.querySelector('#per-class')).toBeNull();
    expect(target.querySelectorAll('#pr tbody button.cls')).toHaveLength(8);
    expect(target.querySelector('#pr button.trace')).not.toBeNull();
    expect(target.querySelector('.d-line')).toBeNull();
    expect(text('#details-title')).toBe('Details');
    const terms = [...target.querySelectorAll('#details dt')].map((d) => d.textContent?.trim());
    for (const t of ['Architecture', 'Input size', 'Parameters', 'Head', 'File size', 'Classes', 'Source', 'Evaluated on', 'Model id', 'Created'])
      expect(terms).toContain(t);
    // In page order: the curves, the matrix, then Details.
    const order = ['#pr', '#cm', '#details'].map((s) => target.querySelector(s)!);
    expect(order[0]!.compareDocumentPosition(order[1]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(order[1]!.compareDocumentPosition(order[2]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("switches the matrix's cells between the row's share and the objects counted, the shade staying the share", async () => {
    await open('m_fast_n');
    const ev = await hub.api.modelEval('m_fast_n');
    const cell = () => target.querySelector<HTMLElement>('#cm .cell[data-r="0"][data-c="0"]')!;
    const shade = cell().style.getPropertyValue('--s');
    const row = ev.confusion[0]!;
    expect(cell().querySelector('.v')?.textContent?.trim()).toBe(String(Math.round((row[0]! / row.reduce((a, b) => a + b, 0)) * 100)));
    expect(text('#cm-title .meta')).toContain('share of each true class');
    const counts = [...target.querySelectorAll<HTMLButtonElement>('#cm [role="radiogroup"][aria-label="Cell values"] [role="radio"]')].find((b) => b.textContent?.trim() === 'Counts')!;
    counts.click();
    flushSync();
    expect(counts.getAttribute('aria-checked')).toBe('true');
    expect(cell().querySelector('.v')?.textContent?.trim()).toBe(String(row[0]));
    expect(cell().style.getPropertyValue('--s')).toBe(shade);
    expect(text('#cm-title .meta')).toContain('objects per true class');
  });
});
