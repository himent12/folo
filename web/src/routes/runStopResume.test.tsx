// @vitest-environment happy-dom
// The run page's Stop and Resume (issue #382, review of PR #389): Stop goes through the live view (`LiveRun.stop`, so a
// late answer never lands over a newer end), and the palette offers Resume except on the mock server.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { LiveRun } from '$lib/api/live';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { pageCommandList } from '$lib/shell/pageCommands';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  hub.serverMock = null;
  vi.restoreAllMocks();
});

const until = (what: string, ready: () => boolean) =>
  vi.waitFor(
    () => {
      flushSync();
      if (!ready()) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );

async function open(status: string): Promise<MockApi> {
  const api = new MockApi('normal', 0);
  const run = (await api.runs()).find((r) => r.status === status)!;
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#/runs/${run.id}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Run />, target);
  await until('the page', () => !!target.querySelector('button[aria-label="More run actions"]'));
  return api;
}

describe('the run page', () => {
  it('stops a run through its live view', async () => {
    const stop = vi.spyOn(LiveRun.prototype, 'stop');
    await open('running');
    const header = () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Stop');
    await until('Stop', () => !!header());
    header()!.click();
    const confirm = () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Stop run');
    await until('the confirm', () => !!confirm());
    confirm()!.click();
    await until('the stop', () => stop.mock.calls.length > 0);
    await until('the confirm to close', () => !confirm());
  });

  it('offers Resume in the palette except on the mock server and in the in-browser demo', async () => {
    await open('stopped');
    const resume = () => pageCommandList.items.some((x) => x.id === 'run-resume');
    await until('the commands', () => pageCommandList.items.length > 0);
    // The demo answers Resume 501 as `folo serve --mock` does (review of #389, round 3).
    expect(resume()).toBe(false);
    // As a real server: offered, also when its health has not said (`serverMock` unknown).
    hub.mode = 'live';
    hub.serverMock = false;
    // The page loads the run again for the new mode.
    await until('Resume', resume);
    hub.serverMock = true;
    flushSync();
    expect(resume()).toBe(false);
    hub.serverMock = null;
    flushSync();
    expect(resume()).toBe(true);
  });

  it('prevents duplicate resumeRun calls when triggered twice while in flight', async () => {
    const api = await open('stopped');
    hub.mode = 'live';
    hub.serverMock = false;
    await until('Resume command', () => pageCommandList.items.some((x) => x.id === 'run-resume'));
    const run = (await api.runs()).find((r) => r.status === 'stopped')!;
    let finishResume!: (r: typeof run) => void;
    const resumeSpy = vi.spyOn(hub.api, 'resumeRun').mockImplementation(() => new Promise((r) => (finishResume = r)));

    const cmd = pageCommandList.items.find((x) => x.id === 'run-resume')!;
    cmd.run();
    cmd.run();
    flushSync();
    expect(resumeSpy).toHaveBeenCalledTimes(1);

    finishResume({ ...run, status: 'running' });
    await Promise.resolve();
    await Promise.resolve();
  });

  it('resets Stop dialog state when navigating to another run and ignores late errors from the previous run', async () => {
    const api = await open('running');
    const runs = await api.runs();
    const other = runs.find((r) => r.status === 'completed')!;
    let rejectStop!: (e: Error) => void;
    vi.spyOn(api, 'stopRun').mockImplementation(() => new Promise((_, rej) => (rejectStop = rej)));

    const header = () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Stop');
    await until('Stop', () => !!header());
    header()!.click();
    const confirm = () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Stop run');
    await until('the confirm', () => !!confirm());
    confirm()!.click();
    flushSync();

    // Navigate to another run while stop is still in flight: dialog must close cleanly.
    router.route = parse(`#/runs/${other.id}`);
    await until('other run', () => target.textContent?.includes(other.name) ?? false);
    expect(document.querySelector('dialog[open]')).toBeNull();

    // Rejecting the stale stop must not leave busy set or open any dialog on the new run.
    rejectStop(new Error('network down'));
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(document.querySelector('dialog[open]')).toBeNull();
  });
});
