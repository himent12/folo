// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import RemoveDialog, { runsImpact, writtenByServer } from './RemoveDialog';

describe('runsImpact', () => {
  it('says what the runs that used the dataset keep and lose, and warns about one training on it now', () => {
    expect(runsImpact([])).toBeNull();
    expect(runsImpact([{ name: 'a', status: 'completed' }])).toEqual({
      lost: '1 run trained on it. It keeps its metrics and models, but its sample images will not show until the folder is registered again.',
      active: null,
    });
    const two = runsImpact([
      { name: 'a', status: 'completed' },
      { name: 'cells-s', status: 'running' },
    ])!;
    expect(two.lost).toMatch(/^2 runs trained on it\. They keep their metrics and models, but their sample images/);
    expect(two.active).toBe('cells-s is training or queued on it now: it carries on.');
    expect(runsImpact([{ name: 'a', status: 'queued' }, { name: 'b', status: 'running' }])!.active).toBe('2 runs are training or queued on it now: they carry on.');
  });
});

describe('writtenByServer', () => {
  it("knows a dataset in the data dir from a folder of the person's own, in any spelling", () => {
    expect(writtenByServer('D:/folo/datasets/shapes-synth', 'D:/folo')).toBe(true);
    expect(writtenByServer('d:\\FOLO\\datasets\\shapes-synth\\', 'D:/folo/')).toBe(true);
    expect(writtenByServer('/home/me/.folo/datasets/x', '/home/me/.folo')).toBe(true);
    expect(writtenByServer('D:/folo/datasets', 'D:/folo')).toBe(false);
    expect(writtenByServer('D:/data/shapes', 'D:/folo')).toBe(false);
    expect(writtenByServer('D:/folo-other/datasets/x', 'D:/folo')).toBe(false);
    expect(writtenByServer('D:/folo/datasets/x', undefined)).toBe(false);
  });
});

describe('RemoveDialog', () => {
  let target: HTMLElement;
  let c: { unmount(): void } | undefined;
  beforeEach(() => {
    hub.api = new MockApi('normal', 0);
    hub.mode = 'mock';
    HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
      this.open = true;
    };
    target = document.createElement('div');
    document.body.appendChild(target);
  });
  afterEach(() => {
    c?.unmount();
    target.remove();
    vi.restoreAllMocks();
  });

  it('names the runs that trained on the dataset, never promising they keep everything', async () => {
    const run = (name: string, status: Run['status'], dataset_id: string) => ({ name, status, config: { dataset_id } }) as Run;
    vi.spyOn(hub.api, 'runs').mockResolvedValue([run('a', 'completed', 'ds_x'), run('b', 'running', 'ds_x'), run('c', 'completed', 'ds_other')]);
    c = render(() => <RemoveDialog open dataset={{ id: 'ds_x', name: 'cells', path: 'D:/data/cells' }} />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('dialog')!.textContent).toContain('2 runs trained on it.');
    });
    const text = document.querySelector('dialog')!.textContent!;
    expect(text).toContain('b is training or queued on it now: it carries on.');
    expect(text).not.toContain('keep their results');
    // A run is training on it: its files are not offered for deletion.
    expect(document.querySelector('dialog .switch')).toBeNull();
  });

  it('offers to delete the images of a set FOLO generated, and says so on the button', async () => {
    vi.spyOn(hub.api, 'runs').mockResolvedValue([]);
    const sys = await hub.api.system();
    vi.spyOn(hub.api, 'system').mockResolvedValue({ ...sys, data_dir: 'D:/folo' });
    const del = vi.spyOn(hub.api, 'deleteDataset').mockResolvedValue();
    const dataset = { id: 'ds_x', name: 'shapes-synth', path: 'D:/folo/datasets/shapes-synth', splits: { train: 800, val: 200, test: 0 } };
    c = render(() => <RemoveDialog open dataset={dataset} />, target);
    const box = await vi.waitFor(() => {
      flushSync();
      const b = document.querySelector<HTMLInputElement>('dialog .switch input');
      if (!b) throw new Error('no switch yet');
      return b;
    });
    expect(document.querySelector('dialog .switch')!.textContent).toContain('Also delete the 1,000 generated images');
    box.click();
    flushSync();
    const confirm = [...document.querySelectorAll<HTMLButtonElement>('dialog button')].find((b) => b.textContent?.trim() === 'Remove and delete files')!;
    confirm.click();
    await vi.waitFor(() => expect(del).toHaveBeenCalledWith('ds_x', { files: true }));
  });

  it('does not offer the files while a run on any dataset is queued or training, and says when it will', async () => {
    const run = (name: string, status: Run['status'], dataset_id: string) => ({ name, status, config: { dataset_id } }) as Run;
    vi.spyOn(hub.api, 'runs').mockResolvedValue([run('elsewhere', 'queued', 'ds_other')]);
    const sys = await hub.api.system();
    vi.spyOn(hub.api, 'system').mockResolvedValue({ ...sys, data_dir: 'D:/folo' });
    const dataset = { id: 'ds_x', name: 'shapes-synth', path: 'D:/folo/datasets/shapes-synth', splits: { train: 800, val: 200, test: 0 } };
    c = render(() => <RemoveDialog open dataset={dataset} />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('dialog')!.textContent).toContain('Its 1,000 generated images can be deleted too once no run is queued or training.');
    });
    expect(document.querySelector('dialog .switch')).toBeNull();
    // No run trained on this dataset, so nothing is said about runs losing their samples.
    expect(document.querySelector('dialog')!.textContent).not.toContain('trained on it');
  });
});
