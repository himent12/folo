// @vitest-environment happy-dom
import { flush } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import AlsoRunning from './AlsoRunning';

let target: HTMLElement | undefined;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  target = undefined;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

async function settle(ms = 0) {
  await vi.advanceTimersByTimeAsync(ms);
  for (let i = 0; i < 6; i++) {
    await vi.advanceTimersByTimeAsync(0);
    flush();
  }
}

describe('AlsoRunning', () => {
  it('preserves last known entry progress and surfaces a refresh failure when polling benchEntries fails transiently', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const [bench] = await api.benches();
    const [entry] = await api.benchEntries(bench!.id);
    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benches = async () => [{ ...bench!, active: true }];
    api.benchLoops = async () => [];

    let failEntries = false;
    api.benchEntries = async () => {
      if (failEntries) throw new ApiError(503, 'unavailable', 'busy');
      return [{ ...entry!, name: 'folo-n-best', status: 'running', progress: { done: 1479, total: 3219, condition: null, severity: null } }];
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();

    const detail = () => target!.querySelector('.activity .detail');
    expect(detail()?.textContent).toBe('folo-n-best · 45%');
    expect(detail()?.getAttribute('data-refresh-failed')).toBeNull();

    // Next poll fails transiently: retain last known entry progress instead of collapsing to generic fallback.
    failEntries = true;
    await settle(5000);
    expect(detail()?.textContent).toBe('folo-n-best · 45% · refresh failed');
    expect(detail()?.getAttribute('data-refresh-failed')).toBe('true');
    expect(target.textContent).not.toContain('An entry or a loop is running');

    // Subsequent poll succeeds: refresh failure note clears cleanly.
    failEntries = false;
    await settle(5000);
    expect(detail()?.textContent).toBe('folo-n-best · 45%');
    expect(detail()?.getAttribute('data-refresh-failed')).toBeNull();
  });

  it('preserves a running loop row and marks refresh failed when polling benchLoops fails transiently', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const benches = await api.benches();
    const allLoops = await Promise.all(benches.map((b) => api.benchLoops(b.id)));
    const idx = allLoops.findIndex((l) => l.length > 0);
    const bench = benches[idx]!;
    const loop = allLoops[idx]![0]!;
    const gen1 = {
      ...loop.generations[0]!,
      generation: 1,
      labels: null,
      run_id: null,
      model_id: null,
      model_name: null,
      entry_id: null,
      score: null,
      map50_95: null,
      ci95: null,
    };

    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benches = async () => [{ ...bench, active: true }];
    api.benchEntries = async () => [];
    let failLoops = false;
    api.benchLoops = async () => {
      if (failLoops) throw new ApiError(500, 'internal', 'boom');
      return [{ ...loop, status: 'running', phase: 'labeling', finished: null, verdict: null, generations: [loop.generations[0]!, gen1] }];
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();

    const detail = () => target!.querySelector('.activity .detail');
    expect(detail()?.textContent).toBe(`Labeling for gen 1 · ${bench.name}`);

    failLoops = true;
    await settle(5000);
    expect(detail()?.textContent).toBe(`Labeling for gen 1 · ${bench.name} · refresh failed`);
    expect(detail()?.getAttribute('data-refresh-failed')).toBe('true');
    expect(target.textContent).not.toContain('An entry or a loop is running');
  });

  it('surfaces initial activity refresh failure instead of generic fallback when no prior detail exists', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const [bench] = await api.benches();
    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benches = async () => [{ ...bench!, active: true }];
    api.benchEntries = async () => {
      throw new ApiError(503, 'unavailable', 'entries down');
    };
    api.benchLoops = async () => {
      throw new ApiError(503, 'unavailable', 'loops down');
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();

    const detail = target.querySelector('.activity .detail');
    expect(detail?.textContent).toBe('Could not refresh benchmark activity');
    expect(detail?.getAttribute('data-refresh-failed')).toBe('true');
  });
});
