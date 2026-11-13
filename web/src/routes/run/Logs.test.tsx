// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { AppendLog } from '$lib/api/appendLog';
import type { LogLine } from '$lib/api/types';
import Logs from './Logs';

const line = (n: number, level: LogLine['level'] = 'info'): LogLine => ({ t: `2026-11-09T10:00:${String(n).padStart(2, '0')}Z`, level, message: `epoch ${n} done` });

/**
 * Port of `mount(Logs, { target, props })` with `props` a `$state` of `{ logs, version, live }`: the lines are the
 * run's append log (`live.logsLog`, what the port reads), `logs.push` appends to it and `props.version++` bumps.
 */
function mount(_: typeof Logs, o: { target: HTMLElement; props: { logs: LogLine[]; version: number; live: boolean } }) {
  const log = new AppendLog<LogLine>();
  log.push(...o.props.logs);
  const version = signal(o.props.version);
  const props = {
    logs: log,
    get version() {
      return version.peek();
    },
    set version(v: number) {
      version.set(v);
    },
  };
  return { props, c: render(() => <Logs log={log} version={version.get()} live={o.props.live} />, o.target) };
}

let target: HTMLElement;
let c: Rendered | undefined;

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
  c?.unmount();
  c = undefined;
  target.remove();
  vi.useRealTimers();
});

describe('Logs: announcements', () => {
  it('is silent by default, and announces new lines in batches once asked', async () => {
    const m = mount(Logs, { target, props: { logs: [line(1)], version: 0, live: true } });
    c = m.c;
    const props = m.props;
    const logs = props.logs;
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
    c = mount(Logs, { target, props: { logs: [line(1)], version: 0, live: false } }).c;
    flushSync();
    expect(target.querySelector('button[aria-label="Announce new lines"]')).toBeNull();
    expect(target.querySelector('[role="log"]')?.getAttribute('aria-live')).toBe('off');
  });
});

// The port's streaming path (not in the Svelte tests): rows keyed by line, the counts and filters kept as lines arrive
// and leave, always what the full filter over the log gives.
describe('Logs: streaming', () => {
  const rows = () => [...target.querySelectorAll<HTMLElement>('.line')];
  const texts = () => rows().map((p) => p.querySelector('.m')!.textContent);
  const LEVELS: LogLine['level'][] = ['debug', 'info', 'info', 'info', 'warn', 'error'];
  const order = { debug: 0, info: 1, warn: 2, error: 3 } as const;

  it('adds one row per new line and drops one per trimmed line, the others kept as they are', () => {
    const log = new AppendLog<LogLine>();
    const version = signal(0);
    for (let i = 0; i < 600; i++) log.push({ t: `t${i}`, level: 'info', message: `line ${i}` });
    c = render(() => <Logs log={log} version={version.get()} live />, target);
    flushSync();
    expect(rows().length).toBe(500);
    const kept = rows().slice(1);
    // One line in, the cap's oldest out (as LiveRun does past 2000 lines).
    log.push({ t: 't600', level: 'info', message: 'line 600' });
    log.trimFront(1);
    version.set(1);
    flushSync();
    const now = rows();
    expect(now.length).toBe(500);
    expect(now.slice(0, 499)).toEqual(kept);
    expect(texts().at(-1)).toBe('line 600');
  });

  it('counts and filters by level as the full filter over the log would, through appends, trims and a replace', () => {
    let seed = 3;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    const log = new AppendLog<LogLine>();
    const version = signal(0);
    c = render(() => <Logs log={log} version={version.get()} live />, target);
    flushSync();
    let level: 'all' | 'warn' | 'error' = 'all';
    let n = 0;
    const option = (name: string) => [...target.querySelectorAll<HTMLElement>('[aria-label="Log level"] button, [aria-label="Log level"] [role="radio"]')].find((b) => b.textContent?.trim().startsWith(name));
    for (let round = 0; round < 120; round++) {
      const add = 1 + Math.floor(rnd() * 60);
      for (let k = 0; k < add; k++, n++) log.push({ t: `t${n}`, level: LEVELS[Math.floor(rnd() * LEVELS.length)]!, message: `line ${n}` });
      if (log.length > 2000) log.trimFront(log.length - 2000);
      if (round === 80) log.replace(log.items.slice(100));
      version.set(round + 1);
      flushSync();
      // Pick a level now and then, as a person would.
      if (round % 17 === 16) {
        level = (['warn', 'error', 'all'] as const)[Math.floor(round / 17) % 3]!;
        option(level === 'all' ? 'All' : level === 'warn' ? 'Warn' : 'Error')!.click();
        flushSync();
      }
      const min = level === 'all' ? 0 : order[level];
      expect(texts()).toEqual(
        log.items
          .filter((l) => order[l.level] >= min)
          .slice(-500)
          .map((l) => l.message),
      );
      const warns = log.items.filter((l) => l.level === 'warn').length;
      const errors = log.items.filter((l) => l.level === 'error').length;
      expect(option('Warn')!.textContent?.replace('Warn', '').trim()).toBe(warns ? String(warns) : '');
      expect(option('Error')!.textContent?.replace('Error', '').trim()).toBe(errors ? String(errors) : '');
    }
  });
});

describe('Logs: a log longer than the box shows', () => {
  const open = (n: number, earlier = 0) => {
    const log = new AppendLog<LogLine>();
    log.push(...Array.from({ length: n }, (_, i) => ({ t: `2026-11-12T10:00:00Z`, level: 'info' as const, message: `epoch ${i + 1}` })));
    c = render(() => <Logs log={log} version={0} live={false} earlier={earlier} />, target);
    flushSync();
  };
  const head = () => target.querySelector('.earlier')?.textContent?.replace(/\s+/g, ' ').trim();
  const rows = () => target.querySelectorAll('.line').length;

  it('counts every line, says how many come before the first shown, and shows them when asked', () => {
    open(704);
    expect(target.textContent).toContain('704');
    expect(rows()).toBe(500);
    expect(head()).toBe('204 lines earlier Show them');
    [...target.querySelectorAll<HTMLButtonElement>('.earlier button')][0]!.click();
    flushSync();
    expect(rows()).toBe(704);
    expect(target.querySelector('.line')?.textContent).toContain('epoch 1');
    expect(head()).toBeUndefined();
  });

  it('says where the lines older than the ones it holds are', () => {
    open(2000, 704);
    expect(target.querySelector('header, .head')?.textContent ?? target.textContent).toContain('2704');
    expect(head()).toBe('2,204 lines earlier Show 1,500 lines');
    target.querySelector<HTMLButtonElement>('.earlier button')!.click();
    flushSync();
    expect(head()).toBe("704 lines earlier are in the run folder's log.jsonl.");
  });

  it('counts errors among the lines Warn shows', () => {
    const log = new AppendLog<LogLine>();
    log.push(...Array.from({ length: 550 }, (_, i) => ({ t: `2026-11-12T10:00:00Z`, level: i % 5 === 0 ? ('error' as const) : ('warn' as const), message: `w ${i}` })));
    c = render(() => <Logs log={log} version={0} live={false} />, target);
    flushSync();
    [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.startsWith('Warn'))!.click();
    flushSync();
    expect(rows()).toBe(500);
    expect(head()).toBe('50 lines earlier Show them');
  });

  it('says nothing above a log the box shows whole', () => {
    open(30);
    expect(head()).toBeUndefined();
  });
});
