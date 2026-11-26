// @vitest-environment happy-dom
// Sprint 14 Studio regression tests: Live* null-id reset & viewSeq/heard hygiene, Benchmarks & LabelJobs URL state
// sync, ZoomStage blur/Escape/wheel drag cancellation & HUD focus recovery, CommandPalette same-tick reopen, and
// rowNav overlay/G-chord guards.
import { createRoot, flush, signal } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, BenchEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { LiveRun } from '$lib/api/live';
import { LiveBench } from '$lib/api/liveBench';
import { LiveLabelJob } from '$lib/api/liveLabel';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchEntry } from '$lib/api/types';
import { parse, router } from '$lib/router';
import CommandPalette from '$lib/shell/CommandPalette';
import { buildPaletteItems } from '$lib/shell/commandPalette/providers';
import { forgetRows, rowNav } from '$lib/ui/rowNav';
import Benchmarks from './Benchmarks';
import LabelJobs from './LabelJobs';
import * as benchListState from './bench/listState';
import * as labelListState from './label/listState';
import ZoomStage from './playground/ZoomStage';

let target: HTMLElement;
const savedApi = hub.api;
const savedMode = hub.mode;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  hub.epoch++;
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
  vi.useRealTimers();
});

describe('Live* stores reset stale state when id becomes null and guard view switches', () => {
  it('resets LiveRun, LiveBench, and LiveLabelJob when id() transitions to null', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    const runs = await api.runs();
    const benches = await api.benches();
    const jobs = await api.labelJobs();

    const runId = signal<string | null>(runs[0]!.id);
    const benchId = signal<string | null>(benches[0]!.id);
    const jobId = signal<string | null>(jobs[0]!.id);

    let liveRun!: LiveRun;
    let liveBench!: LiveBench;
    let liveJob!: LiveLabelJob;
    const stop = createRoot((dispose) => {
      liveRun = new LiveRun(() => runId.get());
      liveBench = new LiveBench(() => benchId.get());
      liveJob = new LiveLabelJob(() => jobId.get());
      return dispose;
    });
    flush();

    await vi.waitFor(() => {
      flush();
      expect(liveRun.run?.id).toBe(runs[0]!.id);
      expect(liveBench.bench?.id).toBe(benches[0]!.id);
      expect(liveJob.job?.id).toBe(jobs[0]!.id);
    });

    runId.set(null);
    benchId.set(null);
    jobId.set(null);
    flush();

    expect(liveRun.run).toBeUndefined();
    expect(liveRun.steps).toEqual([]);
    expect(liveRun.connection).toBe('idle');
    expect(liveRun.loading).toBe(false);

    expect(liveBench.bench).toBeUndefined();
    expect(liveBench.entries).toEqual([]);
    expect(liveBench.connection).toBe('idle');
    expect(liveBench.loading).toBe(false);

    expect(liveJob.job).toBeUndefined();
    expect(liveJob.images).toEqual([]);
    expect(liveJob.connection).toBe('idle');
    expect(liveJob.loading).toBe(false);
    stop();
  });

  it('LiveBench clears #heard on switch and rejects stale act() answers across benchmark switches', async () => {
    let handlers: BenchEventHandlers = {};
    const mkBench = (id: string, name: string) => ({ id, name, entries: 0 }) as unknown as Bench;
    const mkEntry = (id: string, status: BenchEntry['status']) =>
      ({ id, bench_id: 'bn_1', name: id, status, created: '2026-11-09T10:00:00Z', score: null, progress: { done: 0, total: 10, condition: null, severity: null } }) as unknown as BenchEntry;

    hub.api = {
      mock: false,
      bench: (id: string) => Promise.resolve(mkBench(id, `name-${id}`)),
      benchEntries: () => Promise.resolve([]),
      benchLoops: () => Promise.resolve([]),
      benchEvents: (_id: string, h: BenchEventHandlers) => {
        handlers = h;
        return () => {};
      },
    } as unknown as Api;
    hub.mode = 'live';

    const id = signal<string | null>('bn_1');
    let live!: LiveBench;
    const stop = createRoot((dispose) => {
      live = new LiveBench(() => id.get());
      return dispose;
    });
    flush();
    await vi.waitFor(() => expect(live.bench?.id).toBe('bn_1'));

    // Start an act() and capture a pre-switch mark() on bn_1, then switch to bn_2 and back to bn_1 before they settle.
    let resolveStale!: (b: Bench) => void;
    const staleAct = live.act(() => new Promise<Bench>((r) => (resolveStale = r)));
    const staleMark = live.mark();
    handlers.entry?.(mkEntry('shared_e', 'stopped'));

    id.set(null);
    flush();
    // A late mutation using a mark captured before null-id transition must NOT repopulate the cleared store.
    live.putEntry(mkEntry('stale_after_null', 'queued'), staleMark);
    expect(live.entries).toEqual([]);

    id.set('bn_2');
    flush();
    await vi.waitFor(() => expect(live.bench?.id).toBe('bn_2'));
    id.set('bn_1');
    flush();
    await vi.waitFor(() => expect(live.bench?.id).toBe('bn_1'));

    // Stale act() and stale pre-switch mark() from the earlier bn_1 view must NOT overwrite the reloaded bn_1.
    resolveStale(mkBench('bn_1', 'stale-overwrite'));
    await staleAct;
    expect(live.bench?.name).toBe('name-bn_1');
    live.putEntry(mkEntry('stale_after_switch', 'queued'), staleMark);
    expect(live.entries).toEqual([]);

    // #heard from the previous view was cleared, so a new action mark on `shared_e` applies cleanly.
    const mark = live.mark();
    live.putEntry(mkEntry('shared_e', 'queued'), mark);
    expect(live.entries.map((e) => e.id)).toEqual(['shared_e']);
    stop();
  });
});

describe('Benchmarks and LabelJobs URL query-param state sync', () => {
  it('parses, serializes, and summarizes Benchmarks listState with defaults omitted and ?new preserved', () => {
    expect(benchListState.readState(new URLSearchParams(''))).toEqual(benchListState.DEFAULT_STATE);
    expect(benchListState.readState(new URLSearchParams('source=bogus&sort=nope&dir=sideways'))).toEqual(benchListState.DEFAULT_STATE);
    expect(benchListState.readState(new URLSearchParams('source=active&sort=name'))).toEqual({
      filter: 'active',
      sort: 'name',
      dir: 'asc',
      q: '',
    });
    expect(
      benchListState.writeState(
        { filter: 'dataset', sort: 'leader', dir: 'desc', q: 'fast' },
        new URLSearchParams('new=1&source=all'),
      ),
    ).toBe('new=1&source=dataset&q=fast&sort=leader&dir=desc');
    expect(benchListState.writeState(benchListState.DEFAULT_STATE)).toBe('');
    expect(benchListState.emptyFilterSummary({ filter: 'active', q: 'haze' })).toBe('No running benchmarks match "haze".');
  });

  it('hydrates Benchmarks filter/search/sort from URL and writes changes back to URL', async () => {
    history.replaceState(history.state, '', '#/benchmarks?source=dataset&q=no-match-xyz');
    router.route = parse('#/benchmarks?source=dataset&q=no-match-xyz');
    const view = render(() => <Benchmarks />, target);
    try {
      await vi.waitFor(() => {
        flush();
        expect(target.textContent).toContain('No dataset benchmarks match "no-match-xyz".');
      });

      const clearBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
      clearBtn.click();
      flush();

      expect(router.location.path).toBe('/benchmarks');
      expect(router.location.query.toString()).toBe('');

      const leaderBtn = [...target.querySelectorAll<HTMLButtonElement>('thead th button')].find((b) => b.textContent?.trim() === 'Leader')!;
      leaderBtn.click();
      flush();
      expect(router.location.query.get('sort')).toBe('leader');
      expect(router.location.query.get('dir')).toBe('desc');
    } finally {
      view.unmount();
    }
  });

  it('hydrates LabelJobs status/search from URL and preserves them when stripping ?new=1', async () => {
    expect(labelListState.emptyFilterSummary({ filter: 'failed', q: 'bolt' })).toBe('No failed labeling jobs match "bolt".');
    history.replaceState(history.state, '', '#/datasets/label?new=1&status=idle&q=no-match-xyz');
    router.route = parse('#/datasets/label?new=1&status=idle&q=no-match-xyz');
    const view = render(() => <LabelJobs />, target);
    try {
      await vi.waitFor(() => {
        flush();
        expect(target.textContent).toContain('No idle labeling jobs match "no-match-xyz".');
      });
      // `?new=1` was stripped without losing `status` or `q`.
      expect(router.location.query.get('new')).toBeNull();
      expect(router.location.query.get('status')).toBe('idle');
      expect(router.location.query.get('q')).toBe('no-match-xyz');
    } finally {
      view.unmount();
    }
  });

  it('palette New benchmark / New labeling job preserve existing list query params on same route', () => {
    const items = buildPaletteItems({
      pageCommands: [],
      runs: [],
      datasets: [],
      models: [],
      benches: [],
      benchesOk: true,
      labelJobs: [],
      loops: [],
      cloud: false,
      lacksLabel: false,
      touchOnly: false,
      onshortcuts: () => {},
    });
    const benchNew = items.find((i) => i.id === 'bench-new')!;
    const labelNew = items.find((i) => i.id === 'label-new')!;

    history.replaceState(history.state, '', '#/benchmarks?source=dataset&q=coco&sort=leader&dir=desc');
    router.route = parse('#/benchmarks?source=dataset&q=coco&sort=leader&dir=desc');
    benchNew.run();
    expect(router.location.name).toBe('benchmarks');
    expect(router.location.query.get('new')).toBe('1');
    expect(router.location.query.get('source')).toBe('dataset');
    expect(router.location.query.get('q')).toBe('coco');
    expect(router.location.query.get('sort')).toBe('leader');

    history.replaceState(history.state, '', '#/datasets/label?status=active&q=bus');
    router.route = parse('#/datasets/label?status=active&q=bus');
    labelNew.run();
    expect(router.location.name).toBe('label-jobs');
    expect(router.location.query.get('new')).toBe('1');
    expect(router.location.query.get('status')).toBe('active');
    expect(router.location.query.get('q')).toBe('bus');
  });

  it('restores Benchmarks and LabelJobs filters to match the URL on Back after editing during a slow detail hold', async () => {
    const releaseHold = router.holdWhile((to) => to.name === 'benchmark' || to.name === 'label-job');
    try {
      // Benchmarks: start at ?q=before, hold navigation to a detail route, edit search to `after`, then return to ?q=before.
      history.replaceState(history.state, '', '#/benchmarks?q=before');
      router.route = parse('#/benchmarks?q=before');
      const bView = render(() => <Benchmarks />, target);
      try {
        flush();
        const input = target.querySelector<HTMLInputElement>('input[type="search"]')!;
        expect(input.value).toBe('before');

        router.route = parse('#/benchmarks/bn_slow');
        flush();
        input.value = 'after';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        flush();
        expect(input.value).toBe('after');

        // Back arrives before the slow detail chunk finishes loading.
        router.route = parse('#/benchmarks?q=before');
        flush();
        expect(input.value).toBe('before');
      } finally {
        bView.unmount();
      }

      // LabelJobs: same route-hold + edit + Back scenario.
      history.replaceState(history.state, '', '#/datasets/label?q=before');
      router.route = parse('#/datasets/label?q=before');
      const lView = render(() => <LabelJobs />, target);
      try {
        flush();
        const input = target.querySelector<HTMLInputElement>('input[type="search"]')!;
        expect(input.value).toBe('before');

        router.route = parse('#/datasets/label/lj_slow');
        flush();
        input.value = 'after';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        flush();
        expect(input.value).toBe('after');

        router.route = parse('#/datasets/label?q=before');
        flush();
        expect(input.value).toBe('before');
      } finally {
        lView.unmount();
      }
    } finally {
      releaseHold();
    }
  });
});

describe('ZoomStage drag cancellation and HUD keyboard focus recovery', () => {
  it('cancels an active pan drag on window blur and Escape so panning never stays stuck, suppresses click on release, and preserves HUD focus at zoom boundaries', async () => {
    const onBoxClick = vi.fn();
    const view = render(
      () => (
        <ZoomStage resetKey="img_1">
          <div class="det-box" style="width: 200px; height: 200px" onclick={onBoxClick}>
            canvas
          </div>
        </ZoomStage>
      ),
      target,
    );
    try {
      flush();
      const stage = target.querySelector<HTMLDivElement>('.zoom-stage')!;
      const box = target.querySelector<HTMLDivElement>('.det-box')!;
      stage.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300, x: 0, y: 0, toJSON() {} });
      const zoomIn = target.querySelector<HTMLButtonElement>('button[aria-label="Zoom in"]')!;
      const zoomOut = target.querySelector<HTMLButtonElement>('button[aria-label="Zoom out"]')!;

      // Zoom in so panning is active.
      zoomIn.click();
      flush();
      expect(stage.classList.contains('zoomed')).toBe(true);

      // Start dragging past click slop, then blur window before pointerup.
      box.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 1, clientX: 100, clientY: 100 }));
      stage.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 1, pointerId: 1, clientX: 130, clientY: 120 }));
      flush();
      expect(stage.classList.contains('panning')).toBe(true);

      window.dispatchEvent(new Event('blur'));
      flush();
      expect(stage.classList.contains('panning')).toBe(false);

      // A subsequent drag with a new pointerId works immediately (not blocked by stale `drag`).
      box.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 2, clientX: 100, clientY: 100 }));
      stage.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 1, pointerId: 2, clientX: 140, clientY: 120 }));
      flush();
      expect(stage.classList.contains('panning')).toBe(true);

      // Escape cancels active drag and stops propagation to page shortcuts.
      const bubbled = vi.fn();
      window.addEventListener('keydown', bubbled);
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      window.removeEventListener('keydown', bubbled);
      flush();
      expect(stage.classList.contains('panning')).toBe(false);
      expect(bubbled).not.toHaveBeenCalled();

      // Releasing the mouse button over the detection box after Escape-canceled pan must NOT trigger a box click.
      box.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 2, clientX: 140, clientY: 120 }));
      box.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      expect(onBoxClick).not.toHaveBeenCalled();

      // Wheel zoom mid-drag also cancels pan and suppresses click on mouse release.
      box.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 3, clientX: 100, clientY: 100 }));
      stage.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 1, pointerId: 3, clientX: 140, clientY: 120 }));
      flush();
      stage.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: -100, clientX: 140, clientY: 120 }));
      flush();
      expect(stage.classList.contains('panning')).toBe(false);
      box.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 3, clientX: 140, clientY: 120 }));
      box.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      expect(onBoxClick).not.toHaveBeenCalled();

      // A normal click afterwards selects the detection box normally.
      box.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 4, clientX: 100, clientY: 100 }));
      box.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 4, clientX: 100, clientY: 100 }));
      box.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      expect(onBoxClick).toHaveBeenCalledTimes(1);

      // Keyboard zoom-out back to Fit disables Zoom out and hands focus to Zoom in instead of dropping to body.
      zoomOut.focus();
      expect(document.activeElement).toBe(zoomOut);
      // Zoomed twice (button + wheel), click zoomOut until Fit.
      while (!zoomOut.disabled) {
        zoomOut.click();
        flush();
      }
      await Promise.resolve();
      flush();
      expect(zoomOut.disabled).toBe(true);
      expect(document.activeElement).toBe(zoomIn);
    } finally {
      view.unmount();
    }
  });
});

describe('CommandPalette same-tick reopen and rowNav overlay guards', () => {
  it('reopens CommandPalette when Escape closes and open flips back true in the same tick', () => {
    const open = signal(true);
    const view = render(() => <CommandPalette bind:open={open} onshortcuts={() => {}} />, target);
    try {
      flush();
      const dialog = target.querySelector<HTMLDialogElement>('dialog.palette')!;
      expect(dialog.open).toBe(true);

      // Simulate Escape (synchronous close via oncancel) followed immediately by Ctrl+K before flush.
      dialog.dispatchEvent(new Event('cancel', { cancelable: true }));
      expect(dialog.open).toBe(false);
      expect(open.peek()).toBe(false);
      open.set(true);
      flush();
      expect(dialog.open).toBe(true);

      // A stale queued native `close` event firing after reopen must not close or reset the reopened dialog.
      dialog.dispatchEvent(new Event('close'));
      flush();
      expect(dialog.open).toBe(true);
      expect(open.peek()).toBe(true);
    } finally {
      view.unmount();
    }
  });

  it('rowNav ignores global J/K list entry while a menu, info popover, or G-chord indicator is open', () => {
    const t = document.createElement('table');
    t.innerHTML = '<tbody><tr data-row data-id="r0"><td>r0</td></tr><tr data-row data-id="r1"><td>r1</td></tr></tbody>';
    document.body.appendChild(t);
    const rows = [...t.querySelectorAll<HTMLTableRowElement>('tr')];
    const nav = rowNav(t);
    try {
      for (const setup of [
        () => {
          const m = document.createElement('div');
          m.setAttribute('role', 'menu');
          document.body.appendChild(m);
          return m;
        },
        () => {
          const p = document.createElement('div');
          p.className = 'info-pop open';
          document.body.appendChild(p);
          return p;
        },
        () => {
          const g = document.createElement('div');
          g.className = 'g-chord';
          document.body.appendChild(g);
          return g;
        },
      ]) {
        const overlay = setup();
        (document.activeElement as HTMLElement | null)?.blur();
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true, cancelable: true }));
        expect(document.activeElement).not.toBe(rows[0]);
        overlay.remove();
      }
      // Once overlays close, J enters the list normally.
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true, cancelable: true }));
      expect(document.activeElement).toBe(rows[0]);
    } finally {
      nav.destroy();
    }
  });
});
