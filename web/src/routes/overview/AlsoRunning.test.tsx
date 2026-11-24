// @vitest-environment happy-dom
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
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

async function benchWithLoop(api: MockApi) {
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
  return {
    bench,
    runningLoop: { ...loop, status: 'running' as const, phase: 'labeling' as const, finished: null, verdict: null, generations: [loop.generations[0]!, gen1] },
  };
}

describe('AlsoRunning', () => {
  it('preserves last known entry progress and surfaces a non-truncating refresh note when polling benchEntries fails transiently', async () => {
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
    const note = () => target!.querySelector('.activity .refresh-note');
    expect(detail()?.textContent).toBe('folo-n-best · 45%');
    expect(note()).toBeNull();

    // Next poll fails transiently: retain last known entry progress and show refresh failure in its own non-truncating slot.
    failEntries = true;
    await settle(5000);
    expect(detail()?.textContent).toBe('folo-n-best · 45%');
    expect(detail()?.classList.contains('truncate')).toBe(true);
    expect(note()?.textContent).toBe('Refresh failed');
    expect(note()?.classList.contains('truncate')).toBe(false);
    expect(note()?.hasAttribute('data-fact')).toBe(true);
    expect(note()?.getAttribute('data-refresh-failed')).toBe('true');
    expect(target.textContent).not.toContain('An entry or a loop is running');

    // Subsequent poll succeeds: refresh failure note clears cleanly.
    failEntries = false;
    await settle(5000);
    expect(detail()?.textContent).toBe('folo-n-best · 45%');
    expect(note()).toBeNull();
  });

  it('preserves a running loop row and renders refresh failed outside the ellipsized detail on phone widths', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const { bench, runningLoop } = await benchWithLoop(api);

    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benches = async () => [{ ...bench, active: true }];
    api.benchEntries = async () => [];
    let failLoops = false;
    api.benchLoops = async () => {
      if (failLoops) throw new ApiError(500, 'internal', 'boom');
      return [runningLoop];
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();

    const detail = () => target!.querySelector('.activity .detail');
    const note = () => target!.querySelector('.activity .refresh-note');
    expect(detail()?.textContent).toBe(`Labeling for gen 1 · ${bench.name}`);
    expect(note()).toBeNull();

    failLoops = true;
    await settle(5000);
    expect(detail()?.textContent).toBe(`Labeling for gen 1 · ${bench.name}`);
    expect(note()?.textContent).toBe('Refresh failed');
    expect(note()?.classList.contains('truncate')).toBe(false);
    expect(target.textContent).not.toContain('An entry or a loop is running');

    // Verify phone CSS wraps the refresh note intact rather than clipping it inside an ellipsized cell.
    const src = readFileSync(resolve(process.cwd(), 'src/routes/overview/AlsoRunning.tsx'), 'utf8');
    const phone = src.match(/@media \(max-width: 767px\) \{([\s\S]*?)\n {2}\}/)![1]!;
    expect(phone).toContain('flex-wrap: wrap');
    expect(phone).toContain('white-space: normal');
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

    expect(target.querySelector('.activity .detail')).toBeNull();
    const note = target.querySelector('.activity .refresh-note');
    expect(note?.textContent).toBe('Could not refresh benchmark activity');
    expect(note?.getAttribute('data-refresh-failed')).toBe('true');
  });

  it('handles partial failures independently: entries fail while loop succeeds, and loops fail while entry succeeds', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const { bench, runningLoop } = await benchWithLoop(api);
    const [entry] = await api.benchEntries(bench.id);
    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benches = async () => [{ ...bench, active: true }];

    let mode: 'entries-fail-loop-ok' | 'both-fail-with-prior-loop' | 'entry-ok-loops-fail' = 'entries-fail-loop-ok';
    api.benchEntries = async () => {
      if (mode === 'entry-ok-loops-fail') {
        return [{ ...entry!, name: 'folo-s-best', status: 'running', progress: { done: 25, total: 100, condition: null, severity: null } }];
      }
      throw new ApiError(503, 'unavailable', 'entries down');
    };
    api.benchLoops = async () => {
      if (mode === 'entries-fail-loop-ok') return [runningLoop];
      throw new ApiError(503, 'unavailable', 'loops down');
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();

    // Entries failed while loop succeeded: benchmark row reports entry failure, loop row stays clean.
    const rows = () => [...target!.querySelectorAll('.activity li')];
    expect(rows()).toHaveLength(2);
    expect(rows()[0]!.querySelector('.refresh-note')?.textContent).toBe('Could not refresh benchmark entries');
    expect(rows()[1]!.querySelector('.detail')?.textContent).toBe(`Labeling for gen 1 · ${bench.name}`);
    expect(rows()[1]!.querySelector('.refresh-note')).toBeNull();

    // Next poll both fail while prior loop is retained: benchmark row still specifically says entries (loop row carries loop failure).
    mode = 'both-fail-with-prior-loop';
    await settle(5000);
    expect(rows()).toHaveLength(2);
    expect(rows()[0]!.querySelector('.refresh-note')?.textContent).toBe('Could not refresh benchmark entries');
    expect(rows()[1]!.querySelector('.refresh-note')?.textContent).toBe('Refresh failed');

    // Next poll entries succeed (active entry) while loops fail with no prior loops (after clearing via 404 first).
    api.benchLoops = async () => {
      throw new ApiError(404, 'not_found', 'gone');
    };
    mode = 'entry-ok-loops-fail';
    await settle(5000);
    expect(rows()).toHaveLength(1);
    expect(rows()[0]!.querySelector('.detail')?.textContent).toBe('folo-s-best · 25%');
    expect(rows()[0]!.querySelector('.refresh-note')).toBeNull();

    // Now loops fail transiently while entry succeeds: entry row notes loop refresh failed.
    api.benchLoops = async () => {
      throw new ApiError(503, 'unavailable', 'loops down');
    };
    await settle(5000);
    expect(rows()).toHaveLength(1);
    expect(rows()[0]!.querySelector('.detail')?.textContent).toBe('folo-s-best · 25%');
    expect(rows()[0]!.querySelector('.refresh-note')?.textContent).toBe('Could not refresh loops');
  });

  it('clears prior entries/loops on 404 or 501 without marking a refresh failure', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const { bench, runningLoop } = await benchWithLoop(api);
    const [entry] = await api.benchEntries(bench.id);
    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benches = async () => [{ ...bench, active: true }];

    let status: 'ok' | 'missing' = 'ok';
    api.benchEntries = async () => {
      if (status === 'missing') throw new ApiError(404, 'not_found', 'benchmark removed');
      return [{ ...entry!, name: 'folo-n-best', status: 'running', progress: { done: 50, total: 100, condition: null, severity: null } }];
    };
    api.benchLoops = async () => {
      if (status === 'missing') throw new ApiError(501, 'not_implemented', 'no loops');
      return [runningLoop];
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();
    expect(target.querySelectorAll('.activity li')).toHaveLength(2);

    status = 'missing';
    await settle(5000);
    expect(target.querySelectorAll('.activity li')).toHaveLength(1);
    expect(target.querySelector('.activity .detail')?.textContent).toBe('An entry or a loop is running');
    expect(target.querySelector('.activity .refresh-note')).toBeNull();
  });

  it('surfaces top-level refresh failures both on visible rows and when no rows were previously active', async () => {
    const api = new MockApi('normal', 0);
    const system = await api.system();
    const [bench] = await api.benches();
    api.labelJobs = async () => [];
    api.cloudInstances = async () => [];
    api.benchEntries = async () => [];
    api.benchLoops = async () => [];

    let failBenches = false;
    api.benches = async () => {
      if (failBenches) throw new ApiError(503, 'unavailable', 'benches down');
      return [{ ...bench!, active: false }];
    };

    hub.api = api;
    hub.mode = 'mock';
    vi.useFakeTimers();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <AlsoRunning system={system} />, target);
    await settle();
    expect(target.querySelector('.activity')).toBeNull();

    // Top-level benches poll fails transiently with no active rows: section surfaces the stale resource.
    failBenches = true;
    await settle(5000);
    const notice = target.querySelector('.activity .stale-notice');
    expect(notice?.textContent).toBe('Could not refresh benchmarks');
    expect(notice?.getAttribute('data-refresh-failed')).toBe('true');

    // Recovers on next poll: section hides cleanly again.
    failBenches = false;
    await settle(5000);
    expect(target.querySelector('.activity')).toBeNull();
  });
});
