// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import LiveRunCard from './LiveRunCard.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

function render(run: Run | null) {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(LiveRunCard, { target, props: { run } });
  flushSync();
  return target.querySelector('section.card')!;
}

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

describe('LiveRunCard: the lime hairline means "training now"', () => {
  it('is off on the loading placeholder', () => {
    expect(render(null).classList.contains('live')).toBe(false);
  });

  it('is on once a running run has loaded', async () => {
    const runs = await new MockApi('normal', 0).runs();
    const running = runs.find((r) => r.status === 'running')!;
    expect(render(running).classList.contains('live')).toBe(true);
  });

  it('is off for a run that is not running', async () => {
    const runs = await new MockApi('normal', 0).runs();
    const done = runs.find((r) => r.status === 'completed')!;
    expect(render(done).classList.contains('live')).toBe(false);
  });
});
