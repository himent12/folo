// @vitest-environment happy-dom
// Sprint 14 Round 14 playtest regression tests:
// 1. Capability-gated routes suppress 501 fetches, polling, and SSE retries when features are absent.
// 2. rowNav allows J/K while a passive hover tooltip popover (`role="tooltip"`) is open, while still guarding interactive popovers.
// 3. Leaderboard keeps Entry column readable at 1280px via compact desktop widths + container query expansion.
// 4. fadeScroll defers ResizeObserver DOM writes via coalesced requestAnimationFrame and cancels on destroy.
import { createRoot, flush, signal } from 'fw';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { ApiError, type Api } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { LiveBench } from '$lib/api/liveBench';
import { LiveLabelJob } from '$lib/api/liveLabel';
import { MockApi } from '$lib/api/mock';
import type { LabelImage, LabelJob as LabelJobType, SystemInfo } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { fadeScroll } from '$lib/ui/fadeScroll';
import { forgetRows, rowNav } from '$lib/ui/rowNav';
import Benchmark from './Benchmark';
import BenchCompare from './BenchCompare';
import BenchEntry from './BenchEntry';
import BenchLoop from './BenchLoop';
import LabelJob from './LabelJob';
import LabelJobs from './LabelJobs';
import NewBenchDialog from './bench/NewBenchDialog';

let target: HTMLElement;
const savedApi = hub.api;
const savedMode = hub.mode;
const savedFeatures = hub.features;

const MOCK_SYSTEM_NO_EXTRAS: SystemInfo = {
  version: '0.1.0-mock',
  features: ['queue_control'],
  device: { kind: 'cpu', name: 'CPU' },
  workspace: '/tmp/mock',
  disk_free_bytes: 1_000_000_000,
} as unknown as SystemInfo;

function spyMockServerApi() {
  const err501 = () => Promise.reject(new ApiError(501, 'not_implemented', 'Not implemented in mock mode'));
  const calls = {
    labelJobs: vi.fn(err501),
    labelAgents: vi.fn(err501),
    labelJob: vi.fn(err501),
    labelImages: vi.fn(err501),
    labelLogs: vi.fn(err501),
    labelEvents: vi.fn(() => () => {}),
    benches: vi.fn(err501),
    bench: vi.fn(err501),
    benchEntries: vi.fn(err501),
    benchLoops: vi.fn(err501),
    benchEvents: vi.fn(() => () => {}),
    benchCatalog: vi.fn(err501),
    datasets: vi.fn(() => Promise.resolve([])),
    models: vi.fn(() => Promise.resolve([])),
    runs: vi.fn(() => Promise.resolve([])),
    system: vi.fn(() => Promise.resolve(MOCK_SYSTEM_NO_EXTRAS)),
  };
  const api = {
    mock: false,
    ...calls,
  } as unknown as Api;
  return { api, calls };
}

beforeEach(() => {
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
    this.open = false;
  };
  window.matchMedia ??= (() => ({ matches: false, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  target.remove();
  document.body.innerHTML = '';
  forgetRows();
  hub.api = savedApi;
  hub.mode = savedMode;
  hub.features = savedFeatures;
  hub.epoch++;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('Round 14 playtest: capability-gated routes suppress 501 requests and retries', () => {
  it('LabelJobs and LabelJob issue zero label requests or polls when hub lacks agent_labeling', async () => {
    vi.useFakeTimers();
    const { api, calls } = spyMockServerApi();
    hub.api = api;
    hub.mode = 'live';
    hub.features = ['queue_control'];
    hub.epoch++;

    history.replaceState(history.state, '', '#/datasets/label');
    router.route = parse('#/datasets/label');
    const listView = render(() => <LabelJobs />, target);
    try {
      flush();
      await vi.advanceTimersByTimeAsync(11_000);
      flush();
      expect(calls.labelJobs).not.toHaveBeenCalled();
      expect(calls.labelAgents).not.toHaveBeenCalled();
      expect(target.textContent).toContain('cannot label');
    } finally {
      listView.unmount();
    }

    history.replaceState(history.state, '', '#/datasets/label/lj_1');
    router.route = parse('#/datasets/label/lj_1');
    const detailView = render(() => <LabelJob />, target);
    try {
      flush();
      await vi.advanceTimersByTimeAsync(5_000);
      flush();
      expect(calls.labelJob).not.toHaveBeenCalled();
      expect(calls.labelImages).not.toHaveBeenCalled();
      expect(calls.labelAgents).not.toHaveBeenCalled();
      expect(calls.labelEvents).not.toHaveBeenCalled();
    } finally {
      detailView.unmount();
    }
  });

  it('NewBenchDialog defers dataset/job fetches until opened and skips labelJobs when lacking agent_labeling', async () => {
    const { api, calls } = spyMockServerApi();
    hub.api = api;
    hub.mode = 'live';
    hub.features = ['queue_control', 'vision_bench'];
    hub.epoch++;

    const open = signal(false);
    const view = render(() => <NewBenchDialog bind:open={open} />, target);
    try {
      flush();
      await Promise.resolve();
      expect(calls.datasets).not.toHaveBeenCalled();
      expect(calls.labelJobs).not.toHaveBeenCalled();

      open.set(true);
      flush();
      await vi.waitFor(() => expect(calls.datasets).toHaveBeenCalledTimes(1));
      expect(calls.labelJobs).not.toHaveBeenCalled();
    } finally {
      view.unmount();
    }
  });

  it('Benchmark, BenchEntry, BenchLoop, and BenchCompare issue zero 501 bench/label fetches when lacking vision_bench', async () => {
    const { api, calls } = spyMockServerApi();
    hub.api = api;
    hub.mode = 'live';
    hub.features = ['queue_control'];
    hub.epoch++;

    const routes: [string, () => unknown][] = [
      ['#/benchmarks/bn_3c90', () => <Benchmark />],
      ['#/benchmarks/bn_3c90/entries/en_1', () => <BenchEntry />],
      ['#/benchmarks/bn_3c90/loops/lp_1', () => <BenchLoop />],
      ['#/benchmarks/bn_3c90/compare?entries=en_1,en_2', () => <BenchCompare />],
    ];
    for (const [hash, Factory] of routes) {
      history.replaceState(history.state, '', hash);
      router.route = parse(hash);
      const view = render(() => Factory() as Node, target);
      try {
        flush();
        await Promise.resolve();
        await Promise.resolve();
        flush();
      } finally {
        view.unmount();
      }
    }

    expect(calls.bench).not.toHaveBeenCalled();
    expect(calls.benchEntries).not.toHaveBeenCalled();
    expect(calls.benchLoops).not.toHaveBeenCalled();
    expect(calls.benchEvents).not.toHaveBeenCalled();
    expect(calls.benchCatalog).not.toHaveBeenCalled();
    expect(calls.labelJobs).not.toHaveBeenCalled();
    expect(calls.labelAgents).not.toHaveBeenCalled();
  });

  it('LiveBench and LiveLabelJob treat 501 not_implemented as terminal (no retry loop) and LiveLabelJob rejects stale pre-null put()', async () => {
    vi.useFakeTimers();
    const { api, calls } = spyMockServerApi();
    const unsubBench = vi.fn();
    const unsubLabel = vi.fn();
    calls.benchEvents.mockImplementation(() => unsubBench);
    calls.labelEvents.mockImplementation(() => unsubLabel);
    hub.api = api;
    hub.mode = 'live';
    hub.features = null; // Simulate legacy server without /api/health features
    hub.epoch++;

    let liveBench!: LiveBench;
    let liveLabel!: LiveLabelJob;
    const stop = createRoot((dispose) => {
      liveBench = new LiveBench(() => 'bn_501');
      liveLabel = new LiveLabelJob(() => 'lj_501');
      return dispose;
    });
    flush();
    await vi.advanceTimersByTimeAsync(15_000);
    flush();

    // Only the initial attempt runs; 501 stops SSE and prevents exponential-backoff retries.
    expect(calls.bench).toHaveBeenCalledTimes(1);
    expect(calls.labelJob).toHaveBeenCalledTimes(1);
    expect(unsubBench).toHaveBeenCalledTimes(1);
    expect(unsubLabel).toHaveBeenCalledTimes(1);
    expect(liveBench.error).toBeInstanceOf(ApiError);
    expect(liveLabel.error).toBeInstanceOf(ApiError);
    stop();

    // Symmetry check: LiveLabelJob rejects a stale mark() captured before id transitions to null.
    vi.useRealTimers();
    const mockApi = new MockApi('normal', 0);
    hub.api = mockApi;
    hub.mode = 'mock';
    hub.features = null;
    hub.epoch++;
    const jobs = await mockApi.labelJobs();
    const jobId = signal<string | null>(jobs[0]!.id);
    let labelStore!: LiveLabelJob;
    const stop2 = createRoot((dispose) => {
      labelStore = new LiveLabelJob(() => jobId.get());
      return dispose;
    });
    flush();
    await vi.waitFor(() => {
      flush();
      expect(labelStore.job?.id).toBe(jobs[0]!.id);
    });
    const staleMark = labelStore.mark();
    jobId.set(null);
    flush();
    labelStore.put([{ id: 'stale_img', name: 'stale.jpg' } as unknown as LabelImage], staleMark);
    expect(labelStore.images).toEqual([]);
    // Prevent unused type import lint.
    const _jobCheck: LabelJobType | undefined = labelStore.job;
    expect(_jobCheck).toBeUndefined();
    stop2();
  });
});

describe('Round 14 playtest: rowNav popover guard, Leaderboard widths, and fadeScroll ResizeObserver deferral', () => {
  it('rowNav allows J/K when a passive tooltip popover (`role="tooltip"`) is open, but blocks when an interactive popover is open', () => {
    const t = document.createElement('table');
    t.innerHTML = '<tbody><tr data-row data-id="r0"><td>r0</td></tr><tr data-row data-id="r1"><td>r1</td></tr></tbody>';
    document.body.appendChild(t);
    const rows = [...t.querySelectorAll<HTMLTableRowElement>('tr')];
    const nav = rowNav(t);

    // Simulate Chromium's native `:popover-open` matching via a test spy since happy-dom lacks Popover API pseudo-classes.
    const tip = document.createElement('div');
    tip.setAttribute('popover', 'manual');
    tip.setAttribute('role', 'tooltip');
    document.body.appendChild(tip);

    const infoPop = document.createElement('div');
    infoPop.setAttribute('popover', 'auto');
    document.body.appendChild(infoPop);

    let openEl: HTMLElement | null = tip;
    const origQS = document.querySelector.bind(document);
    vi.spyOn(document, 'querySelector').mockImplementation((sel: string) => {
      if (sel.includes(':popover-open')) {
        if (!openEl) return null;
        const base = sel.replace(':popover-open', '');
        return openEl.matches(base) ? openEl : null;
      }
      return origQS(sel);
    });

    try {
      // Passive hover tooltip open: J MUST enter the table.
      (document.activeElement as HTMLElement | null)?.blur();
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true, cancelable: true }));
      expect(document.activeElement).toBe(rows[0]);

      // Interactive popover open (no role="tooltip"): J MUST be blocked.
      openEl = infoPop;
      (document.activeElement as HTMLElement | null)?.blur();
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true, cancelable: true }));
      expect(document.activeElement).not.toBe(rows[0]);
    } finally {
      nav.destroy();
    }
  });

  it('Leaderboard uses compact desktop column widths with container-query widening so Entry stays readable at 1280px', () => {
    const src = readFileSync(resolve(import.meta.dirname, 'bench/Leaderboard.tsx'), 'utf8');
    expect(src).toContain('container-type: inline-size;');
    expect(src).toContain('@container (min-width: 1120px)');
    expect(src).toContain('width: 188px;');
    expect(src).toContain('width: 148px;');
  });

  it('fadeScroll defers ResizeObserver DOM mutations via coalesced requestAnimationFrame and cancels on destroy', () => {
    const roCallbacks: ResizeObserverCallback[] = [];
    const disconnect = vi.fn();
    class FakeResizeObserver {
      constructor(cb: ResizeObserverCallback) {
        roCallbacks.push(cb);
      }
      observe() {}
      unobserve() {}
      disconnect = disconnect;
    }
    const origRO = globalThis.ResizeObserver;
    globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof ResizeObserver;

    const rafQueue = new Map<number, FrameRequestCallback>();
    let nextId = 1;
    const rafSpy = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((cb) => {
      const id = nextId++;
      rafQueue.set(id, cb);
      return id;
    });
    const cafSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation((id) => {
      rafQueue.delete(id);
    });

    const el = document.createElement('div');
    document.body.appendChild(el);
    let scrollWidth = 300;
    let clientWidth = 300;
    Object.defineProperty(el, 'scrollWidth', { get: () => scrollWidth });
    Object.defineProperty(el, 'clientWidth', { get: () => clientWidth });

    const handle = fadeScroll(el);
    const fireRO = () => roCallbacks[0]?.([], {} as ResizeObserver);
    try {
      expect(el.dataset.fade).toBeUndefined();
      expect(el.hasAttribute('tabindex')).toBe(false);

      // Simulate overflow appearing during a ResizeObserver notification.
      scrollWidth = 500;
      fireRO();
      fireRO();

      // Coalesced into a single rAF; DOM attributes are NOT mutated synchronously inside the RO callback.
      expect(rafSpy).toHaveBeenCalledTimes(1);
      expect(el.dataset.fade).toBeUndefined();

      // Running the scheduled animation frame applies the fade attribute and tabIndex.
      const pending = [...rafQueue.values()];
      rafQueue.clear();
      for (const cb of pending) cb(16);
      expect(el.dataset.fade).toBe('end');
      expect(el.tabIndex).toBe(0);

      // Schedule another update then destroy before the frame runs: rAF is cancelled and disconnected.
      scrollWidth = 300;
      fireRO();
      expect(rafQueue.size).toBe(1);
      handle.destroy();
      expect(cafSpy).toHaveBeenCalledTimes(1);
      expect(rafQueue.size).toBe(0);
      expect(disconnect).toHaveBeenCalledTimes(1);
    } finally {
      globalThis.ResizeObserver = origRO;
    }
  });
});
