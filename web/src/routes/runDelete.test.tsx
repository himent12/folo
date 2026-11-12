// @vitest-environment happy-dom
// Deleting a run from its own page (v-shell N10): the list takes the dead page's place in the history, so Back does not
// land on "Run not found".
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
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

describe('deleting a run from its page', () => {
  it('replaces its history entry with the runs list', async () => {
    const api = new MockApi('normal', 0);
    const done = (await api.runs()).find((r) => r.status === 'completed')!;
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#/runs/${done.id}`);
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <Run />, target);
    const menu = () => target.querySelector<HTMLButtonElement>('button[aria-label="More run actions"]');
    await until('the page', () => !!menu());
    menu()!.click();
    flushSync();
    [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((e) => e.textContent?.trim() === 'Delete run')!.click();
    const confirm = () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => /^Delete run/.test(b.textContent?.trim() ?? ''));
    await until('the confirm', () => !!confirm());
    confirm()!.click();
    await until('the navigation', () => go.mock.calls.length > 0);
    expect(go).toHaveBeenCalledWith('/runs', { replace: true });
  });
});
