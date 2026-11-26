// @vitest-environment happy-dom
// The screen of `main` while the next screen's chunk loads, and when it fails to load.
import { bindable, flush, onMount } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../framework/testing';

// happy-dom's Web Animations reject when an intro/outro is cancelled mid-flight (see Toaster.test.tsx).
vi.mock('fw', async (orig) => ({ ...(await orig<typeof import('fw')>()), fly: () => ({}) }));
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { restoringScroll } from '$lib/shell/entryMemory';
import { toast } from '$lib/ui/toast';
import App from './App';
import * as RunScreen from './routes/Run';
import { Screens, type ScreenModule, type ScreenName } from './routes';
import { entryScript, leaveDemoUrl } from './ScreenError';

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((a, b) => ((resolve = a), (reject = b)));
  return { promise, resolve, reject };
}

async function until(get: () => unknown, what: string): Promise<void> {
  await vi.waitFor(
    () => {
      flush();
      if (!get()) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
}

/** Lets pending imports and effects settle without waiting for anything in particular. */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flush();
  }
}

const screen = (text: string): ScreenModule => ({ default: () => <p class="fake">{text}</p> });

let c: Rendered | undefined;
let target: HTMLElement;

beforeEach(() => {
  vi.spyOn(hub, 'connect').mockResolvedValue();
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.body.appendChild(document.createElement('div'));
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  hub.demoError = null;
  hub.demoSlow = false;
  for (const t of toast.items) toast.dismiss(t.id);
  vi.restoreAllMocks();
  router.route = parse('#/');
});

const main = () => target.querySelector('main')!;

describe('the screen kept on display while the next one loads', () => {
  it('keeps its own route: Run stays on run_7d1e and asks nothing of model_b', async () => {
    const run = vi.spyOn(MockApi.prototype, 'run');
    const events = vi.spyOn(MockApi.prototype, 'runEvents');
    const model = deferred<ScreenModule>();
    const loaders: Partial<Record<ScreenName, () => Promise<ScreenModule>>> = {
      // Already loaded (imported above): transforming Run's module graph took seconds of the test's own waits on a
      // loaded machine. The screen still arrives through a loader's promise, as a chunk does.
      run: () => Promise.resolve(RunScreen),
      model: () => model.promise,
    };
    const screens = new Screens((name) => loaders[name]);
    router.route = parse('#/runs/run_7d1e');
    c = render(() => <App screens={screens} />, target);
    await until(() => run.mock.calls.length > 0 && main().querySelector('.page'), 'the run page');

    router.route = parse('#/models/model_b');
    await settle();
    expect(router.location.params.id).toBe('model_b');
    expect(router.route.params.id).toBe('run_7d1e');
    expect(main().querySelector('.page')).not.toBeNull();

    model.resolve(screen('the model page'));
    await until(() => main().querySelector('.fake'), 'the model page');
    expect(router.route.params.id).toBe('model_b');
    expect(main().textContent).toBe('the model page');
    await settle();
    const ids = [...run.mock.calls, ...events.mock.calls].map((call) => call[0]);
    expect(ids.length).toBeGreaterThan(0);
    expect(ids.every((id) => id === 'run_7d1e')).toBe(true);
  });
});

describe('a screen whose chunk fails to load', () => {
  // An upgrade of folo serve under an open tab (shell F11): the old build's chunks are gone, so Retry cannot help.
  for (const [what, served, says] of [
    ['another build', '/assets/index-NEW.js', 'Studio was updated'],
    ['the same build', '/assets/index-OLD.js', 'Could not load this screen'],
  ] as const) {
    it(`asks for a reload only when the server serves ${what} of Studio`, async () => {
      // This page's entry script (a real <script> would be loaded by happy-dom).
      const query = document.querySelector.bind(document);
      vi.spyOn(document, 'querySelector').mockImplementation(((sel: string) =>
        sel === 'script[type="module"][src]' ? Object.assign(document.createElement('div'), { getAttribute: () => '/assets/index-OLD.js' }) : query(sel)) as typeof document.querySelector);
      const fetch = vi.fn(async () => new Response(`<!doctype html><script>theme()</script><script type="module" crossorigin src="${served}"></script>`));
      vi.stubGlobal('fetch', fetch);
      try {
        const screens = new Screens((name) => (name === 'runs' ? () => Promise.reject(new TypeError('Failed to fetch dynamically imported module')) : undefined));
        router.route = parse('#/runs');
        c = render(() => <App screens={screens} />, target);
        await until(() => main().querySelector('[role="alert"]') && fetch.mock.calls.length > 0, 'the error');
        await settle();
        expect(main().querySelector('.title')?.textContent).toBe(says);
        const buttons = [...main().querySelectorAll('button')].map((b) => b.textContent?.trim());
        expect(buttons).toEqual(says === 'Studio was updated' ? ['Reload'] : ['Retry']);
      } finally {
        vi.unstubAllGlobals();
      }
    });
  }

  it("reads a built index.html's entry script", () => {
    expect(entryScript('<script>x()</script>\n<script type="module" crossorigin src="/assets/index-C2L3TdP8.js"></script>')).toBe('/assets/index-C2L3TdP8.js');
    expect(entryScript('<p>no app here</p>')).toBeNull();
  });

  it('shows an error with Retry, which loads it again', async () => {
    let tries = 0;
    const screens = new Screens((name) =>
      name === 'models' ? () => (++tries === 1 ? Promise.reject(new TypeError('Failed to fetch dynamically imported module')) : Promise.resolve(screen('the models'))) : undefined,
    );
    router.route = parse('#/models');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('[role="alert"]'), 'the error');
    expect(main().querySelector('h1')?.textContent).toBe('Models');
    expect(main().querySelector('.title')?.textContent).toBe('Could not load this screen');
    expect(main().textContent).toContain('Failed to fetch dynamically imported module');
    const retry = [...main().querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Retry')!;
    retry.focus();
    retry.click();
    await until(() => main().querySelector('.fake'), 'the models page');
    expect(tries).toBe(2);
    expect(main().querySelector('[role="alert"]')).toBeNull();
    // The focused Retry is gone with the error: focus moves into the recovered screen, not to the body.
    await until(() => document.activeElement !== document.body && main().contains(document.activeElement), 'focus in the screen');
  });

  it('leaves focus the user moved out of main before the recovered screen takes it', async () => {
    let tries = 0;
    const screens = new Screens((name) =>
      name === 'models' ? () => (++tries === 1 ? Promise.reject(new TypeError('Failed to fetch dynamically imported module')) : Promise.resolve(screen('the models'))) : undefined,
    );
    router.route = parse('#/models');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('[role="alert"]'), 'the error');
    await settle();
    const frames: FrameRequestCallback[] = [];
    vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((f) => (frames.push(f), frames.length));
    const outside = target.appendChild(document.createElement('button'));
    const retry = [...main().querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Retry')!;
    retry.focus();
    retry.click();
    await until(() => main().querySelector('.fake') && frames.length > 0, 'the models page and its focus frame');
    outside.focus();
    for (const f of frames.splice(0)) f(0);
    flush();
    expect(document.activeElement).toBe(outside);
  });

  it('shows the screen once a later preload (a link hovered) brings the chunk in', async () => {
    let tries = 0;
    const screens = new Screens((name) =>
      name === 'models' ? () => (++tries === 1 ? Promise.reject(new TypeError('Failed to fetch dynamically imported module')) : Promise.resolve(screen('the models'))) : undefined,
    );
    router.route = parse('#/models');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('[role="alert"]'), 'the error');
    await settle();
    // Not tried again by itself: a failure waits for Retry or another request.
    expect(tries).toBe(1);
    screens.preloadHref('#/models');
    await until(() => main().querySelector('.fake'), 'the models page');
    expect(main().textContent).toBe('the models');
  });

  it('keeps the screen on display, and its route, until the failure, then shows the error in its place', async () => {
    const model = deferred<ScreenModule>();
    const screens = new Screens((name) => (name === 'runs' ? () => Promise.resolve(screen('the runs')) : name === 'model' ? () => model.promise : undefined));
    router.route = parse('#/runs');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('.fake'), 'the runs page');
    router.route = parse('#/models/model_b');
    await settle();
    expect(main().textContent).toBe('the runs');
    expect(router.route.name).toBe('runs');
    model.reject(new TypeError('Failed to fetch dynamically imported module'));
    await until(() => main().querySelector('[role="alert"]'), 'the error');
    expect(router.route.name).toBe('model');
    expect(main().querySelector('.fake')).toBeNull();
  });

  it('offers Reload instead of Retry when a lazy route chunk throws while evaluating', async () => {
    const screens = new Screens((name) =>
      name === 'models' ? () => Promise.reject(new SyntaxError('Unexpected token <')) : undefined,
    );
    router.route = parse('#/models');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('[role="alert"]'), 'the eval error');
    expect(main().querySelector('.title')?.textContent).toBe('This screen could not be loaded');
    expect(main().textContent).toContain('Unexpected token <');
    const buttons = [...main().querySelectorAll('button')].map((b) => b.textContent?.trim());
    expect(buttons).toEqual(['Reload']);
  });

  it('shows a recoverable DemoError with Reload and Leave demo data when the built-in demo chunk fails to evaluate', async () => {
    hub.mode = 'connecting';
    hub.demoError = new ReferenceError('mockInit is not defined');
    const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : undefined));
    router.route = parse('#/');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('[role="alert"]'), 'the demo error');
    expect(main().querySelector('.title')?.textContent).toBe('Could not load demo data');
    expect(main().textContent).toContain('mockInit is not defined');
    const buttons = [...main().querySelectorAll('button')].map((b) => b.textContent?.trim());
    expect(buttons).toEqual(['Reload', 'Leave demo data']);
    const conn = target.querySelector('.conn')!;
    expect(conn.querySelector('.label')?.textContent ?? conn.getAttribute('aria-label')).toBe('Demo failed');
    expect(leaveDemoUrl('http://localhost:5173/?mock=1&scenario=error&keep=yes#/runs')).toBe('http://localhost:5173/?keep=yes#/runs');
  });
});

describe('the scroll an entry kept, when its screen waits for its chunk', () => {
  it('is brought back once the screen shows, and its focus does not send it to the top (review R3)', async () => {
    vi.useFakeTimers();
    try {
      // A reload on Runs, scrolled to 600; the layout is tall from the start (the shell while Runs's chunk loads).
      let y = 0;
      vi.spyOn(window, 'scrollY', 'get').mockImplementation(() => y);
      vi.spyOn(window, 'scrollTo').mockImplementation(((_x: number, to: number) => {
        y = to;
      }) as typeof window.scrollTo);
      vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockImplementation(() => 5000);
      router.route = parse('#/runs');
      sessionStorage.setItem('folo.entries', JSON.stringify([[router.entry, { y: 600 }]]));
      const runs = deferred<ScreenModule>();
      const screens = new Screens((name) => (name === 'runs' ? () => runs.promise : undefined));
      c = render(() => <App screens={screens} />, target);
      await vi.advanceTimersByTimeAsync(1000);
      flush();
      expect(main().textContent).toBe('');
      expect(restoringScroll()).toBe(true);

      runs.resolve(screen('the runs'));
      await vi.advanceTimersByTimeAsync(1000);
      flush();
      expect(main().textContent).toBe('the runs');
      expect([restoringScroll(), y]).toEqual([false, 600]);
    } finally {
      sessionStorage.clear();
      vi.useRealTimers();
    }
  });
});

// s6-shell-2: Settings threw while it was created (a stored server address of the wrong type): the URL said #/settings,
// the sidebar too, but `main` kept the Overview and nothing said that anything had failed.
describe('a screen that throws while it is created', () => {
  it('shows the error with Reload instead of keeping the screen before, and the next page still opens', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const loaders: Partial<Record<ScreenName, () => Promise<ScreenModule>>> = {
      overview: () => Promise.resolve(screen('the overview')),
      settings: () =>
        Promise.resolve({
          default: () => {
            throw new TypeError('e.trim is not a function');
          },
        }),
      runs: () => Promise.resolve(screen('the runs')),
    };
    const screens = new Screens((name) => loaders[name]);
    router.route = parse('#/');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    router.route = parse('#/settings');
    await until(() => !main().querySelector('.fake'), 'the overview to go');
    expect(main().textContent).toContain('This page could not be shown');
    expect(main().textContent).toContain('e.trim is not a function');
    expect([...main().querySelectorAll('button')].some((b) => b.textContent?.includes('Reload'))).toBe(true);

    router.route = parse('#/runs');
    await until(() => main().textContent === 'the runs', 'the runs');
  });

  // Review round 1: a child built before its sibling threw kept its effects (polling, listeners) on the detached page.
  it('disposes what the screen built before it threw: none of its effects run', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let started = 0;
    let stopped = 0;
    const Poller = () => {
      onMount(() => {
        started++;
        return () => stopped++;
      });
      return <i>polls</i>;
    };
    const Thrower = (): JSX.Element => {
      throw new TypeError('the sibling broke');
    };
    const loaders: Partial<Record<ScreenName, () => Promise<ScreenModule>>> = {
      overview: () => Promise.resolve(screen('the overview')),
      settings: () =>
        Promise.resolve({
          default: () => (
            <div>
              <Poller />
              <Thrower />
            </div>
          ),
        }),
    };
    const screens = new Screens((name) => loaders[name]);
    router.route = parse('#/');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    router.route = parse('#/settings');
    await until(() => main().textContent?.includes('the sibling broke'), 'the crash');
    await settle();
    expect(started - stopped).toBe(0);
  });
});

describe('G-chord indicator, guarded/staggered route preloading, and lazy ShortcutsDialog', () => {
  it('guards G-key preloadAll and shows the G-chord indicator only on valid G chords', async () => {
    const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : name === 'runs' ? () => Promise.resolve(screen('the runs')) : undefined));
    const preloadAll = vi.spyOn(screens, 'preloadAll');
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    router.route = parse('#/');
    c = render(() => <App screens={screens} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    // Typing 'g' in an input or pressing Ctrl+G does not trigger preloadAll or show the indicator.
    const input = target.appendChild(document.createElement('input'));
    input.focus();
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'g', bubbles: true, cancelable: true }));
    flush();
    expect(preloadAll).not.toHaveBeenCalled();
    expect(target.querySelector('.g-chord')).toBeNull();
    input.remove();

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'g', ctrlKey: true, bubbles: true, cancelable: true }));
    flush();
    expect(preloadAll).not.toHaveBeenCalled();
    expect(target.querySelector('.g-chord')).toBeNull();

    // Bare 'g' at rest shows the subtle indicator and calls preloadAll; 'r' clears it and navigates.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'g', bubbles: true, cancelable: true }));
    flush();
    expect(preloadAll).toHaveBeenCalledTimes(1);
    expect(target.querySelector('.g-chord')).not.toBeNull();
    expect(target.querySelector('.g-chord')?.textContent).toContain('Runs');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', bubbles: true, cancelable: true }));
    flush();
    await until(() => !target.querySelector('.g-chord'), 'g-chord outro');
    expect(go).toHaveBeenCalledWith('/runs');
  });

  it('staggers idle route preloading (nav before detail screens) and cancels cleanly on unmount', async () => {
    vi.useFakeTimers();
    const origIdle = (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback;
    delete (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback;
    try {
      const loaded: ScreenName[] = [];
      const make = (name: ScreenName) => () => {
        loaded.push(name);
        return Promise.resolve(screen(name));
      };
      const screens = new Screens((name) =>
        (['overview', 'run', 'datasets', 'runs', 'models', 'benchmarks', 'playground'] as ScreenName[]).includes(name) ? make(name) : undefined,
      );
      router.route = parse('#/');
      c = render(() => <App screens={screens} />, target);
      await vi.advanceTimersByTimeAsync(0);
      flush();
      expect(loaded).toEqual(['overview']);

      // After IDLE_PRELOAD_MS (1000ms), only the first nav batch (2 routes) starts — detail screen 'run' waits behind nav.
      await vi.advanceTimersByTimeAsync(1000);
      expect(loaded).toEqual(['overview', 'datasets', 'runs']);

      // After the inter-batch pause (250ms), the second nav batch loads while the rest wait.
      await vi.advanceTimersByTimeAsync(250);
      expect(loaded).toEqual(['overview', 'datasets', 'runs', 'models', 'benchmarks']);

      // Unmounting cancels the remaining batches (including detail screen 'run') so nothing else loads after teardown.
      c.unmount();
      c = undefined;
      await vi.advanceTimersByTimeAsync(5000);
      expect(loaded).toEqual(['overview', 'datasets', 'runs', 'models', 'benchmarks']);
    } finally {
      if (origIdle !== undefined) (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback = origIdle;
      vi.useRealTimers();
    }
  });

  it('uses requestIdleCallback for staggered preloading and cancels pending idle callbacks on unmount', async () => {
    vi.useFakeTimers();
    const origIdle = (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback;
    const origCancel = (globalThis as { cancelIdleCallback?: unknown }).cancelIdleCallback;
    const idleCallbacks = new Map<number, () => void>();
    let nextId = 0;
    const cancelSpy = vi.fn((id: number) => {
      idleCallbacks.delete(id);
    });
    (globalThis as { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback = (cb) => {
      const id = ++nextId;
      idleCallbacks.set(id, cb);
      return id;
    };
    (globalThis as { cancelIdleCallback?: (id: number) => void }).cancelIdleCallback = cancelSpy;
    try {
      const loaded: ScreenName[] = [];
      const make = (name: ScreenName) => () => {
        loaded.push(name);
        return Promise.resolve(screen(name));
      };
      const screens = new Screens((name) =>
        (['overview', 'datasets', 'runs', 'models'] as ScreenName[]).includes(name) ? make(name) : undefined,
      );
      router.route = parse('#/');
      c = render(() => <App screens={screens} />, target);
      await vi.advanceTimersByTimeAsync(1000);
      expect(loaded).toEqual(['overview']);
      expect(idleCallbacks.size).toBe(1);

      // Firing the first idle callback loads only the first batch.
      const [id1, cb1] = [...idleCallbacks.entries()][0]!;
      idleCallbacks.delete(id1);
      cb1();
      await vi.advanceTimersByTimeAsync(250);
      expect(loaded).toEqual(['overview', 'datasets', 'runs']);
      expect(idleCallbacks.size).toBe(1);

      // Unmounting cancels the queued second idle callback.
      c.unmount();
      c = undefined;
      expect(cancelSpy).toHaveBeenCalledTimes(1);
      expect(idleCallbacks.size).toBe(0);
    } finally {
      if (origIdle !== undefined) (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback = origIdle;
      else delete (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback;
      if (origCancel !== undefined) (globalThis as { cancelIdleCallback?: unknown }).cancelIdleCallback = origCancel;
      else delete (globalThis as { cancelIdleCallback?: unknown }).cancelIdleCallback;
      vi.useRealTimers();
    }
  });

  it('skips idle staggered preloading when Save-Data is enabled', async () => {
    vi.useFakeTimers();
    const origConn = Object.getOwnPropertyDescriptor(navigator, 'connection');
    Object.defineProperty(navigator, 'connection', { configurable: true, value: { saveData: true } });
    try {
      const loaded: ScreenName[] = [];
      const make = (name: ScreenName) => () => {
        loaded.push(name);
        return Promise.resolve(screen(name));
      };
      const screens = new Screens((name) =>
        (['overview', 'datasets', 'runs'] as ScreenName[]).includes(name) ? make(name) : undefined,
      );
      router.route = parse('#/');
      c = render(() => <App screens={screens} />, target);
      await vi.advanceTimersByTimeAsync(5000);
      flush();
      expect(loaded).toEqual(['overview']);
    } finally {
      if (origConn) Object.defineProperty(navigator, 'connection', origConn);
      else delete (navigator as { connection?: unknown }).connection;
      vi.useRealTimers();
    }
  });

  it('lazy-loads ShortcutsDialog on first open, caches it across close/reopen, and retries after a fetch failure', async () => {
    let attempts = 0;
    const loadShortcuts = vi.fn(async () => {
      attempts++;
      if (attempts === 1) throw new TypeError('Failed to fetch dynamically imported module');
      return {
        default: (p: { open?: boolean }) => {
          const open = bindable(p, 'open', false);
          return (
            <div class="lazy-shortcuts" data-open={String(open.get())}>
              <button class="close-shortcuts" onclick={() => open.set(false)}>
                Close
              </button>
            </div>
          );
        },
      };
    });
    const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : undefined));
    router.route = parse('#/');
    c = render(() => <App screens={screens} loadShortcuts={loadShortcuts} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    // Never fetched until the user actually asks for shortcuts.
    expect(loadShortcuts).not.toHaveBeenCalled();
    expect(target.querySelector('.lazy-shortcuts')).toBeNull();

    // First attempt fails with a transient fetch error (no Reload needed) and resets so a retry works.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    await settle();
    expect(loadShortcuts).toHaveBeenCalledTimes(1);
    expect(target.querySelector('.lazy-shortcuts')).toBeNull();
    const errToast = toast.items.find((t) => t.tone === 'error' && t.title === 'Could not load keyboard shortcuts');
    expect(errToast).toBeDefined();
    expect(errToast?.action).toBeUndefined();

    // Second attempt loads, clears the error toast, and opens the dialog; closing and reopening reuses the cached chunk.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    await until(() => target.querySelector('.lazy-shortcuts[data-open="true"]'), 'lazy shortcuts dialog');
    expect(loadShortcuts).toHaveBeenCalledTimes(2);
    expect(toast.items.some((t) => t.key === 'shortcuts')).toBe(false);

    target.querySelector<HTMLButtonElement>('.close-shortcuts')!.click();
    flush();
    expect(target.querySelector('.lazy-shortcuts')?.getAttribute('data-open')).toBe('false');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    flush();
    expect(target.querySelector('.lazy-shortcuts')?.getAttribute('data-open')).toBe('true');
    expect(loadShortcuts).toHaveBeenCalledTimes(2);
  });

  it('Escape cancels a pending lazy ShortcutsDialog open without navigating back on a detail page', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const pending = deferred<{ default: (p: { open?: boolean }) => JSX.Element }>();
    const loadShortcuts = vi.fn(() => pending.promise);
    const screens = new Screens((name) => (name === 'run' ? () => Promise.resolve(screen('the run detail')) : undefined));
    router.route = parse('#/runs/run_7d1e');
    c = render(() => <App screens={screens} loadShortcuts={loadShortcuts} />, target);
    await until(() => main().querySelector('.fake'), 'the run detail');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    flush();
    expect(loadShortcuts).toHaveBeenCalledTimes(1);

    // Pressing Escape while the chunk is still loading cancels the open instead of leaving the detail page.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    flush();
    expect(go).not.toHaveBeenCalled();

    // Resolving the import caches the component closed instead of popping it open after Escape.
    pending.resolve({
      default: (p: { open?: boolean }) => <div class="lazy-shortcuts" data-open={String(!!p.open)} />,
    });
    await until(() => target.querySelector('.lazy-shortcuts'), 'cached shortcuts component');
    expect(target.querySelector('.lazy-shortcuts')?.getAttribute('data-open')).toBe('false');
  });

  it('offers Reload on the error toast when lazy ShortcutsDialog or CommandPalette fails to evaluate', async () => {
    const loadShortcuts = vi.fn(async () => {
      throw new SyntaxError('Unexpected token <');
    });
    const loadPalette = vi.fn(async () => {
      throw new ReferenceError('paletteInit is not defined');
    });
    const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : undefined));
    router.route = parse('#/');
    c = render(() => <App screens={screens} loadShortcuts={loadShortcuts} loadPalette={loadPalette} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    await until(() => toast.items.some((t) => t.key === 'shortcuts'), 'shortcuts eval error toast');
    expect(toast.items.find((t) => t.key === 'shortcuts')?.action?.label).toBe('Reload');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    await until(() => toast.items.some((t) => t.key === 'palette'), 'palette eval error toast');
    expect(toast.items.find((t) => t.key === 'palette')?.action?.label).toBe('Reload');
  });
});

describe('lazy CommandPalette', () => {
  beforeEach(() => {
    HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
      this.open = true;
    };
    HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
      this.open = false;
      this.dispatchEvent(new Event('close'));
    };
  });

  it('defers loading until first Ctrl+K, retries after fetch failure, and caches across close/reopen via Cmd+K and header button', async () => {
    let attempts = 0;
    const loadPalette = vi.fn(async () => {
      attempts++;
      if (attempts === 1) throw new TypeError('Failed to fetch dynamically imported module');
      return {
        default: (p: { open?: boolean; onshortcuts: () => void; takeInitialQuery?: () => string }) => {
          const open = bindable(p, 'open', false);
          return (
            <div class="lazy-palette" data-open={String(open.get())}>
              <button class="close-palette" onclick={() => open.set(false)}>
                Close
              </button>
            </div>
          );
        },
      };
    });
    const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : undefined));
    router.route = parse('#/');
    c = render(() => <App screens={screens} loadPalette={loadPalette} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    // Never fetched on initial render.
    expect(loadPalette).not.toHaveBeenCalled();
    expect(target.querySelector('.lazy-palette')).toBeNull();

    // First Ctrl+K attempt fails with a transient fetch error (no Reload needed) and resets so a retry works.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    await until(() => toast.items.some((t) => t.tone === 'error' && t.title === 'Could not load command palette'), 'error toast');
    expect(toast.items.find((t) => t.key === 'palette')?.action).toBeUndefined();
    expect(loadPalette).toHaveBeenCalledTimes(1);
    expect(target.querySelector('.lazy-palette')).toBeNull();

    // Second open via Topbar search button loads, clears error toast, and opens the palette.
    target.querySelector<HTMLButtonElement>('header button[aria-label="Search and commands"]')!.click();
    await until(() => target.querySelector('.lazy-palette[data-open="true"]'), 'lazy palette open');
    expect(loadPalette).toHaveBeenCalledTimes(2);
    expect(toast.items.some((t) => t.key === 'palette')).toBe(false);

    // Close and reopen via Cmd+K: reuses the cached chunk with no extra load.
    target.querySelector<HTMLButtonElement>('.close-palette')!.click();
    flush();
    expect(target.querySelector('.lazy-palette')?.getAttribute('data-open')).toBe('false');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true, bubbles: true, cancelable: true }));
    flush();
    expect(target.querySelector('.lazy-palette')?.getAttribute('data-open')).toBe('true');
    expect(loadPalette).toHaveBeenCalledTimes(2);
  });

  it('buffers keystrokes typed while the palette chunk loads, suppresses background shortcuts, and focuses the real input', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const pending = deferred<typeof import('$lib/shell/CommandPalette')>();
    const loadPalette = vi.fn(() => pending.promise);
    const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : undefined));
    router.route = parse('#/');
    c = render(() => <App screens={screens} loadPalette={loadPalette} />, target);
    await until(() => main().querySelector('.fake'), 'the overview');

    // Open with Ctrl+K and immediately type 'nx' + Backspace + 'ew' before the chunk resolves.
    // 'n' is normally a single-key shortcut ("New training run"), so it must be suppressed while loading.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    flush();
    for (const key of ['n', 'x', 'Backspace', 'e', 'w']) {
      window.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
    }
    flush();
    expect(go).not.toHaveBeenCalled();
    expect(loadPalette).toHaveBeenCalledTimes(1);

    // Resolve with the real CommandPalette module: it opens with 'new' already populated and the input focused.
    const realPalette = await import('$lib/shell/CommandPalette');
    pending.resolve(realPalette);
    await until(() => target.querySelector<HTMLDialogElement>('dialog.palette[open] input'), 'real CommandPalette open');

    const input = target.querySelector<HTMLInputElement>('dialog.palette[open] input')!;
    expect(input.value).toBe('new');
    await until(() => document.activeElement === input, 'palette search input focused');
    expect(input.selectionStart).toBe(3);

    // Closing and reopening starts with an empty query (buffered seed was consumed once).
    target.querySelector<HTMLButtonElement>('dialog.palette button[aria-label="Close"]')!.click();
    flush();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    flush();
    expect(input.value).toBe('');
  });

  it('Escape and second Ctrl+K cancel a pending palette load without navigating back or popping open late', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const pending = deferred<{ default: (p: { open?: boolean; onshortcuts: () => void; takeInitialQuery?: () => string }) => JSX.Element }>();
    const loadPalette = vi.fn(() => pending.promise);
    const screens = new Screens((name) => (name === 'run' ? () => Promise.resolve(screen('the run detail')) : undefined));
    router.route = parse('#/runs/run_7d1e');
    c = render(() => <App screens={screens} loadPalette={loadPalette} />, target);
    await until(() => main().querySelector('.fake'), 'the run detail');

    // First open + type + Escape cancels without navigating back.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    flush();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true, cancelable: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    flush();
    expect(go).not.toHaveBeenCalled();

    // Re-open while still pending + second Ctrl+K toggles off again (no duplicate import call).
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    flush();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'b', bubbles: true, cancelable: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    flush();
    expect(loadPalette).toHaveBeenCalledTimes(1);

    let seenSeed = 'UNSET';
    pending.resolve({
      default: (p: { open?: boolean; onshortcuts: () => void; takeInitialQuery?: () => string }) => {
        const open = bindable(p, 'open', false);
        return (
          <div class="lazy-palette" data-open={String(open.get())}>
            <button class="read-seed" onclick={() => (seenSeed = p.takeInitialQuery?.() ?? '')}>
              Seed
            </button>
          </div>
        );
      },
    });
    await until(() => target.querySelector('.lazy-palette'), 'cached palette component');
    expect(target.querySelector('.lazy-palette')?.getAttribute('data-open')).toBe('false');
    target.querySelector<HTMLButtonElement>('.read-seed')!.click();
    expect(seenSeed).toBe('');
  });

  it('closes an open ShortcutsDialog on Ctrl+K, navigates cleanly to the chosen route, and focuses the destination heading', async () => {
    const realPalette = await import('$lib/shell/CommandPalette');
    const realShortcuts = await import('$lib/shell/ShortcutsDialog');
    const screens = new Screens((name) =>
      name === 'overview'
        ? () => Promise.resolve({ default: () => <h1 tabindex={-1} data-route-focus>Overview</h1> })
        : name === 'models'
          ? () => Promise.resolve({ default: () => <h1 tabindex={-1} data-route-focus>Models</h1> })
          : undefined,
    );
    router.route = parse('#/');
    c = render(
      () => <App screens={screens} loadPalette={() => Promise.resolve(realPalette)} loadShortcuts={() => Promise.resolve(realShortcuts)} />,
      target,
    );
    await until(() => main().querySelector('h1')?.textContent === 'Overview', 'Overview rendered');

    // Open Shortcuts with `?`.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    await until(() => target.querySelector<HTMLDialogElement>('dialog.shortcuts[open]'), 'ShortcutsDialog open');

    // Press Ctrl+K while Shortcuts is open: Shortcuts closes immediately and CommandPalette opens.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    await until(() => target.querySelector<HTMLDialogElement>('dialog.palette[open] input'), 'CommandPalette open');
    expect(target.querySelector('dialog.shortcuts[open]')).toBeNull();

    // Filter to Models and accept with Enter: navigates to #/models, no open dialogs remain, focus lands on Models heading.
    const input = target.querySelector<HTMLInputElement>('dialog.palette[open] input')!;
    input.value = 'models';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flush();
    await until(
      () => target.querySelector('dialog.palette [role="option"][aria-selected="true"]')?.textContent?.includes('Models'),
      'Models option selected',
    );
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    flush();

    await until(() => router.route.name === 'models' && main().querySelector('h1')?.textContent === 'Models', 'Models route rendered');
    expect(target.querySelector('dialog[open]')).toBeNull();
    await until(() => document.activeElement === main().querySelector('h1'), 'focus on Models heading');
  });

  it('closes ShortcutsDialog and CommandPalette on independent path or query navigation', async () => {
    const realPalette = await import('$lib/shell/CommandPalette');
    const realShortcuts = await import('$lib/shell/ShortcutsDialog');
    const screens = new Screens((name) =>
      name === 'dataset'
        ? () => Promise.resolve(screen('the dataset'))
        : name === 'models'
          ? () => Promise.resolve(screen('the models'))
          : undefined,
    );
    router.route = parse('#/datasets/ds_1');
    c = render(
      () => <App screens={screens} loadPalette={() => Promise.resolve(realPalette)} loadShortcuts={() => Promise.resolve(realShortcuts)} />,
      target,
    );
    await until(() => main().querySelector('.fake'), 'dataset');

    // Same-page query change (e.g. Forward restoring ?image=img_1) closes ShortcutsDialog.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
    await until(() => target.querySelector('dialog.shortcuts[open]'), 'ShortcutsDialog open');
    router.route = parse('#/datasets/ds_1?image=img_1');
    await until(() => !target.querySelector('dialog.shortcuts[open]'), 'ShortcutsDialog closed on query change');

    // Independent route change (e.g. browser Back/Forward) also closes CommandPalette.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
    await until(() => target.querySelector('dialog.palette[open]'), 'CommandPalette open');
    router.route = parse('#/models');
    await until(() => main().textContent === 'the models', 'models');
    expect(target.querySelector('dialog.palette[open]')).toBeNull();
  });

  it('ignores a stale queued native close event when ShortcutsDialog is reopened via CommandPalette', async () => {
    const queuedCloses: HTMLDialogElement[] = [];
    const origClose = HTMLDialogElement.prototype.close;
    HTMLDialogElement.prototype.close = function (this: HTMLDialogElement) {
      this.open = false;
      queuedCloses.push(this);
    };
    try {
      const realPalette = await import('$lib/shell/CommandPalette');
      const realShortcuts = await import('$lib/shell/ShortcutsDialog');
      const screens = new Screens((name) => (name === 'overview' ? () => Promise.resolve(screen('the overview')) : undefined));
      router.route = parse('#/');
      c = render(
        () => <App screens={screens} loadPalette={() => Promise.resolve(realPalette)} loadShortcuts={() => Promise.resolve(realShortcuts)} />,
        target,
      );
      await until(() => main().querySelector('.fake'), 'overview');

      // Open Shortcuts -> Ctrl+K closes Shortcuts (close event queued) -> choose "Keyboard shortcuts" to reopen it.
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
      await until(() => target.querySelector('dialog.shortcuts[open]'), 'ShortcutsDialog open');
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true }));
      await until(() => target.querySelector('dialog.palette[open] input'), 'CommandPalette open');
      const input = target.querySelector<HTMLInputElement>('dialog.palette[open] input')!;
      input.value = 'keyboard shortcuts';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flush();
      await until(
        () => target.querySelector('dialog.palette [role="option"][aria-selected="true"]')?.textContent?.includes('Keyboard shortcuts'),
        'Keyboard shortcuts option selected',
      );
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      await until(() => target.querySelector('dialog.shortcuts[open]'), 'ShortcutsDialog reopened');

      // Now deliver all queued native `close` events: the reopened ShortcutsDialog must stay open.
      for (const dlg of queuedCloses.splice(0)) dlg.dispatchEvent(new Event('close'));
      flush();
      expect(target.querySelector('dialog.shortcuts[open]')).not.toBeNull();
      expect(target.querySelector('dialog.palette[open]')).toBeNull();
    } finally {
      HTMLDialogElement.prototype.close = origClose;
    }
  });
});
