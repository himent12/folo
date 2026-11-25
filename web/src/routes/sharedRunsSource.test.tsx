// @vitest-environment happy-dom
// Routes that read GET /api/runs reuse the shell's RunWatch Resource (`watched.source`) via `startFrom`
// so navigating across pages never fires a redundant initial /api/runs request.
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { Resource } from '$lib/api/resource';
import { runsChanged } from '$lib/api/runs';
import { parse, router } from '$lib/router';
import { watched } from '$lib/shell/runList';
import BenchLoop from './BenchLoop';
import Models from './Models';
import Playground from './Playground';
import Run from './Run';
import RunCompare from './RunCompare';
import RunNew from './RunNew';
import Runs from './Runs';

let target: HTMLElement | undefined;
let pageView: { unmount(): void } | undefined;
let sourceView: { unmount(): void } | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  hub.epoch++;
});

afterEach(() => {
  pageView?.unmount();
  pageView = undefined;
  sourceView?.unmount();
  sourceView = undefined;
  watched.source = undefined;
  watched.runs = undefined;
  target?.remove();
  target = undefined;
  vi.restoreAllMocks();
});

const CASES: { name: string; page: Component<Record<string, never>>; path: (runId: string) => string }[] = [
  { name: 'Runs', page: Runs, path: () => '/runs' },
  { name: 'Run', page: Run, path: (id) => `/runs/${id}` },
  { name: 'Models', page: Models, path: () => '/models' },
  { name: 'Playground', page: Playground, path: () => '/playground' },
  { name: 'RunCompare', page: RunCompare, path: (id) => `/runs/compare?ids=${id}` },
  { name: 'RunNew', page: RunNew, path: () => '/runs/new' },
  { name: 'BenchLoop', page: BenchLoop, path: () => '/benchmarks/bench_01/loops/loop_01' },
];

describe('shared watched.source reuse across routes', () => {
  for (const { name, page: Page, path } of CASES) {
    it(`${name} reuses a fresh watched.source without issuing a duplicate api.runs() call`, async () => {
      const runsSpy = vi.spyOn(hub.api, 'runs');
      const shellHost = document.createElement('div');
      let shared!: Resource<Awaited<ReturnType<typeof hub.api.runs>>>;
      function ShellSource() {
        shared = new Resource((s) => hub.api.runs(s));
        watched.source = shared;
        return <div />;
      }
      sourceView = render(() => <ShellSource />, shellHost);
      await vi.waitFor(() => {
        flushSync();
        expect(shared.data).toBeDefined();
      });
      expect(runsSpy).toHaveBeenCalledTimes(1);

      router.route = parse(`#${path(shared.data![0]!.id)}`);
      target = document.body.appendChild(document.createElement('div'));
      pageView = render(() => <Page />, target);
      flushSync();
      await Promise.resolve();
      flushSync();

      expect(runsSpy).toHaveBeenCalledTimes(1);
    });
  }

  it('invalidates watched.source on runsChanged() so routes fetch post-mutation runs without clearing updated', async () => {
    const api = hub.api as MockApi;
    const baseRuns = await api.runs();
    let list = baseRuns;
    const runsSpy = vi.spyOn(api, 'runs').mockImplementation(async () => list);
    const shellHost = document.createElement('div');
    let shared!: Resource<Awaited<ReturnType<typeof hub.api.runs>>>;
    function ShellSource() {
      shared = new Resource((s) => hub.api.runs(s));
      watched.source = shared;
      return <div />;
    }
    sourceView = render(() => <ShellSource />, shellHost);
    await vi.waitFor(() => {
      flushSync();
      expect(shared.data).toBeDefined();
    });
    const updatedBefore = shared.updated;
    expect(updatedBefore).not.toBeNull();
    expect(runsSpy).toHaveBeenCalledTimes(1);

    // Simulate a mutation (e.g. run deleted) while holding shared's refresh in flight:
    // #lend() must reject the stale pre-mutation snapshot while keeping `updated` intact.
    let releaseRefresh!: () => void;
    const held = new Promise<void>((r) => (releaseRefresh = r));
    list = baseRuns.slice(1);
    let call = 0;
    runsSpy.mockImplementation(async () => {
      call++;
      if (call === 1) await held;
      return list;
    });
    runsChanged();
    expect(shared.updated).toBe(updatedBefore);

    router.route = parse('#/runs');
    target = document.body.appendChild(document.createElement('div'));
    pageView = render(() => <Runs />, target);
    flushSync();
    // Runs must NOT synchronously render the deleted run from the stale pre-mutation watched.source snapshot.
    expect(target.textContent).not.toContain(baseRuns[0]!.name);

    releaseRefresh();
    await vi.waitFor(() => {
      flushSync();
      expect(target!.textContent).toContain(baseRuns[1]!.name);
    });
    expect(target.textContent).not.toContain(baseRuns[0]!.name);
  });

  it('RunCompare confirms missing IDs with a fresh read before dropping them from a borrowed watched.source snapshot', async () => {
    const api = hub.api as MockApi;
    const baseRuns = (await api.runs()).filter((r) => r.status !== 'queued');
    const a = baseRuns[0]!;
    const fresh = { ...a, id: 'run_fresh_ext', name: 'fresh-ext' };
    let current = baseRuns;
    vi.spyOn(api, 'runs').mockImplementation(async () => current);
    const realMetrics = api.runMetrics.bind(api);
    vi.spyOn(api, 'runMetrics').mockImplementation((id, s) => realMetrics(id === fresh.id ? a.id : id, s));

    const shellHost = document.createElement('div');
    let shared!: Resource<Awaited<ReturnType<typeof hub.api.runs>>>;
    function ShellSource() {
      shared = new Resource((s) => hub.api.runs(s));
      watched.source = shared;
      return <div />;
    }
    sourceView = render(() => <ShellSource />, shellHost);
    await vi.waitFor(() => {
      flushSync();
      expect(shared.data).toBeDefined();
    });

    // External run created after watched.source's last poll; Compare must confirm and keep it, not drop it.
    current = [...baseRuns, fresh];
    router.route = parse(`#/runs/compare?ids=${a.id},${fresh.id}`);
    target = document.body.appendChild(document.createElement('div'));
    pageView = render(() => <RunCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(target!.textContent).toContain('fresh-ext');
    });
    expect(router.route.query.get('ids')).toBe(`${a.id},${fresh.id}`);

    // A genuinely deleted run ID IS dropped after confirming against the fresh read.
    pageView.unmount();
    pageView = undefined;
    target.remove();
    current = baseRuns;
    router.route = parse(`#/runs/compare?ids=${a.id},run_deleted`);
    target = document.body.appendChild(document.createElement('div'));
    pageView = render(() => <RunCompare />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('ids')).toBe(a.id);
    });
  });
});
