// @vitest-environment happy-dom
// Overview against the mock server: it reads the run list the sidebar reads (shell/runList), so the Training card
// never says "Nothing is training" beside the sidebar's "Training" (shell F13), and Best models catches up when a run
// ends (runs F4).
import { flush } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { watched } from '$lib/shell/runList';
import Overview from './Overview';

let target: HTMLElement;
let c: { unmount(): void } | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  watched.runs = undefined;
  watched.since = null;
});

function open(api: MockApi) {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse('#/');
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Overview />, target);
  flush();
}

/** The Training card's text once `ready` holds for it. */
const training = (ready: (text: string) => boolean, what: string) =>
  vi.waitFor(() => {
    flush();
    const text = target.querySelector('.live')?.textContent ?? '';
    if (!ready(text)) throw new Error(`still waiting for ${what}: ${text}`);
    return text;
  });

/** The mock's runs with none training, so the slot is whatever a test puts in it. */
async function idleRuns(): Promise<Run[]> {
  const list = await new MockApi('normal', 0).runs();
  return list.map((r) => (r.status === 'running' || r.status === 'stopping' || r.status === 'queued' ? { ...r, status: 'completed', queue_position: null } : r));
}

describe('Overview', () => {
  it('shows the run the sidebar shows, before its own list has loaded', async () => {
    const runs = await idleRuns();
    const run = runs.find((r) => r.status === 'completed')!;
    // This page's own requests never answer; the shell's list already has the run training.
    watched.runs = runs.map((r) => (r.id === run.id ? { ...r, status: 'running' } : r));
    open(new MockApi('loading'));
    const text = await training((t) => t.includes(run.name), 'the run');
    expect(text).not.toContain('Nothing is training');
  });

  it('shows a run the worker has taken as starting, not as nothing training', async () => {
    const runs = await idleRuns();
    const run = runs.find((r) => r.status === 'completed')!;
    watched.runs = runs.map((r) => (r.id === run.id ? { ...r, status: 'queued', queue_position: null, epoch: 0, step: 0, best: null } : r));
    open(new MockApi('loading'));
    const text = await training((t) => t.includes(run.name), 'the starting run');
    expect(text).toContain('Starting');
    expect(text).not.toContain('Queued');
  });

  it('refetches Best models when a run ends, and not on an unchanged list', async () => {
    const runs = await idleRuns();
    const run = runs.find((r) => r.status === 'completed')!;
    const api = new MockApi('normal', 0);
    const models = vi.spyOn(api, 'models');
    const list = runs.map((r) => (r.id === run.id ? { ...r, status: 'running' as const } : r));
    watched.runs = list;
    open(api);
    await training((t) => t.includes(run.name), 'the run');
    await vi.waitFor(() => expect(models).toHaveBeenCalledTimes(1));

    watched.runs = [...list];
    flush();
    expect(models).toHaveBeenCalledTimes(1);

    watched.runs = list.map((r) => (r.id === run.id ? { ...r, status: 'completed' } : r));
    flush();
    expect(models).toHaveBeenCalledTimes(2);
    await training((t) => t.includes('Nothing is training'), 'the empty slot');
  });
});
