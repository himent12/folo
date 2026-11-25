// @vitest-environment happy-dom
// Routes that read GET /api/runs reuse the shell's RunWatch Resource (`watched.source`) via `startFrom`
// so navigating across pages never fires a redundant initial /api/runs request.
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { Resource } from '$lib/api/resource';
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

const CASES: { name: string; page: Component<Record<string, never>>; path: string }[] = [
  { name: 'Runs', page: Runs, path: '/runs' },
  { name: 'Run', page: Run, path: '/runs/run_01' },
  { name: 'Models', page: Models, path: '/models' },
  { name: 'Playground', page: Playground, path: '/playground' },
  { name: 'RunCompare', page: RunCompare, path: '/runs/compare?ids=run_01' },
  { name: 'RunNew', page: RunNew, path: '/runs/new' },
  { name: 'BenchLoop', page: BenchLoop, path: '/benchmarks/bench_01/loops/loop_01' },
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

      router.route = parse(`#${path}`);
      target = document.body.appendChild(document.createElement('div'));
      pageView = render(() => <Page />, target);
      flushSync();
      await Promise.resolve();
      flushSync();

      expect(runsSpy).toHaveBeenCalledTimes(1);
    });
  }
});
