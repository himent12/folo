// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import type { ModelEval } from '$lib/api/types';
import { parse, router } from '$lib/router.svelte';
import Model from './Model.svelte';

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
    expect(target.querySelector('section.kpis')?.getAttribute('aria-label')).toBe('Validation metrics');
    expect(text('#pr-title .meta')).toBe('fasteners · validation split · IoU 0.5');
    expect(text('#per-class .foot')).toContain('Evaluated on 360 validation images of fasteners at IoU 0.50.');
    const notes = text('#cm .cm-notes');
    expect(notes).toContain('Counted at conf 0.25, IoU 0.45 on 360 validation images of fasteners.');
    // The matrix no longer claims precision and recall's IoU, and says the two thresholds differ.
    expect(notes).not.toContain('IoU 0.5,');
    const conf = (await hub.api.modelEval('m_fast_n')).best_f1_conf!;
    expect(text('#cm .how-body')).toContain(`Precision and recall are read at conf ${conf.toFixed(2)} (best F1), IoU 0.50`);
    expect(text('#cm .how-body')).toContain('confidence is above 0.25');
    const row = [...target.querySelectorAll('#details .kv')].find((kv) => kv.querySelector('dt')?.textContent === 'Evaluated on');
    expect(row?.querySelector('dd')?.textContent?.replace(/\s+/g, ' ').trim()).toBe('fasteners validation split, 360 images');
  });

  it('says once that a train-only dataset was evaluated on its training split, and never calls it validation', async () => {
    await open('m_mini_n');
    const notes = target.querySelectorAll('[role="note"][aria-label="Evaluation split"]');
    expect(notes).toHaveLength(1);
    expect(notes[0]!.querySelector('.c-title')?.textContent).toBe('Evaluated on the training split');
    expect(notes[0]!.querySelector('.c-msg')?.textContent?.trim()).toBe(
      'fasteners-mini has no validation images, so these scores are on images the model trained on, not new ones.',
    );
    expect(target.querySelector('section.kpis')?.getAttribute('aria-label')).toBe('Metrics on the training split');
    expect(text('#pr-title .meta')).toBe('fasteners-mini · training split · IoU 0.5');
    expect(text('#per-class .foot')).toContain('Evaluated on 24 training images of fasteners-mini at IoU 0.50.');
    expect(text('#cm .cm-notes')).toContain('Counted at conf 0.25, IoU 0.45 on 24 training images of fasteners-mini.');
    const dd = [...target.querySelectorAll('#details .kv')].find((kv) => kv.querySelector('dt')?.textContent === 'Evaluated on')?.querySelector('dd');
    expect(dd?.querySelector('a')?.getAttribute('href')).toContain('/datasets/ds_fasteners_mini');
    expect(dd?.textContent?.replace(/\s+/g, ' ').trim()).toBe('fasteners-mini training split, 24 images');
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
    expect(text('#per-class .foot')).toContain('Evaluated on 360 images of fasteners at IoU 0.50.');
    expect(text('#cm .cm-notes')).toContain('Counted on 360 images of fasteners, at thresholds this server does not report.');
    expect(text('#cm .how-body')).toContain('which this server does not report');
  });
});
