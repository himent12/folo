// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LogLine } from '$lib/api/types';
import Logs from './Logs.svelte';

const line = (n: number, level: LogLine['level'] = 'info'): LogLine => ({ t: `2026-11-09T10:00:${String(n).padStart(2, '0')}Z`, level, message: `epoch ${n} done` });

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  vi.useFakeTimers();
  target = document.createElement('div');
  document.body.appendChild(target);
  globalThis.ResizeObserver ??= class {
    observe() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});
afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  vi.useRealTimers();
});

describe('Logs: announcements', () => {
  it('is silent by default, and announces new lines in batches once asked', async () => {
    const props = $state({ logs: [line(1)], version: 0, live: true });
    const logs = props.logs;
    c = mount(Logs, { target, props });
    flushSync();
    const log = target.querySelector('[role="log"]')!;
    expect(log.getAttribute('aria-live')).toBe('off');
    const status = () => {
      flushSync();
      return target.querySelector('[role="status"]')!.textContent;
    };
    const toggle = target.querySelector<HTMLButtonElement>('button[aria-label="Announce new lines"]')!;
    expect(toggle.getAttribute('aria-pressed')).toBe('false');

    // Off: lines arrive, nothing is said.
    logs.push(line(2));
    props.version++;
    flushSync();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(status()).toBe('');

    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-pressed')).toBe('true');
    // Three lines in one interval: one announcement, how many and the latest.
    logs.push(line(3), line(4), line(5, 'warn'));
    props.version++;
    flushSync();
    await vi.advanceTimersByTimeAsync(9_000);
    expect(status()).toBe('');
    await vi.advanceTimersByTimeAsync(1_000);
    expect(status()).toBe('3 new log lines. Latest: warn: epoch 5 done');
    // One line: the line itself.
    logs.push(line(6));
    props.version++;
    flushSync();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(status()).toBe('epoch 6 done');

    // Off again: the status line empties and stays so.
    toggle.click();
    flushSync();
    logs.push(line(7));
    props.version++;
    await vi.advanceTimersByTimeAsync(10_000);
    flushSync();
    expect(status()).toBe('');
  });

  it('offers no toggle once the run is over', () => {
    c = mount(Logs, { target, props: { logs: [line(1)], version: 0, live: false } });
    flushSync();
    expect(target.querySelector('button[aria-label="Announce new lines"]')).toBeNull();
    expect(target.querySelector('[role="log"]')?.getAttribute('aria-live')).toBe('off');
  });
});
