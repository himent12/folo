// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import RemoveDialog, { runsImpact } from './RemoveDialog';

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
  });
});
