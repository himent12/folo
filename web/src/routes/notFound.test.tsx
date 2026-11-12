// @vitest-environment happy-dom
// Detail pages whose object is gone say so once, through ErrorState's not-found state (lib/ui/ErrorState), with one
// way to the list; a registered dataset whose folder is gone is not that: it says its folder is missing and offers to
// remove it.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Dataset from './Dataset';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

function open(api: MockApi, path: string, page: () => JSX.Element) {
  hub.api = api;
  hub.mode = 'mock';
  history.replaceState(history.state, '', `#${path}`);
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(page, target);
}

const until = (what: string, ready: () => boolean) =>
  vi.waitFor(
    () => {
      flushSync();
      if (!ready()) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );

const buttons = (label: string) => [...target.querySelectorAll('a, button')].filter((b) => b.textContent?.trim() === label);

describe('not found', () => {
  it('a deleted run says so once, with one way to the runs', async () => {
    open(new MockApi('normal', 0), '/runs/run_nope', () => <Run />);
    await until('the not-found state', () => !!target.textContent?.includes('Run not found'));
    expect(target.textContent).toContain('This run was deleted or the link is wrong.');
    expect(target.textContent).not.toContain('Check the run id');
    expect(target.querySelectorAll('.err')).toHaveLength(1);
    expect(buttons('All runs')).toHaveLength(1);
  });

  it('a dataset whose folder is gone says the folder is missing and offers to remove it, not that it was deleted', async () => {
    const api = new MockApi('normal', 0);
    vi.spyOn(api, 'dataset').mockRejectedValue(
      new ApiError(404, 'path_not_found', "the folder of dataset 'shapes' is gone: 'D:/data/shapes' does not exist", undefined, 'GET /api/datasets/ds_shapes'),
    );
    open(api, '/datasets/ds_shapes', () => <Dataset />);
    await until('the folder-missing state', () => buttons('Remove from FOLO').length > 0);
    expect(target.textContent).toContain('Folder missing');
    expect(target.textContent).not.toContain('deleted or the link is wrong');
    expect(target.textContent).not.toContain('Dataset not found');
  });
});
