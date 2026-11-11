// @vitest-environment happy-dom
// The screen of `main` while the next screen's chunk loads, and when it fails to load.
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import App from './App';
import { Screens, type ScreenModule, type ScreenName } from './routes';

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
      run: () => import('./routes/Run'),
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
});
