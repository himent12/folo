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

describe('Logs: search, copy visible, static tokens, and single-reflow pin', () => {
  it('filters visible lines by keyword alongside level filter and shows match counts and empty state', () => {
    const log = new AppendLog<LogLine>();
    log.push(
      { t: '2026-11-12T10:00:01Z', level: 'info', message: 'epoch 1 loss=0.42' },
      { t: '2026-11-12T10:00:02Z', level: 'warn', message: 'NaN gradient skipped' },
      { t: '2026-11-12T10:00:03Z', level: 'error', message: 'CUDA out of memory' },
      { t: '2026-11-12T10:00:04Z', level: 'info', message: 'checkpoint saved to /very/long/path/that/exceeds/thirty/six/characters/checkpoint.folo' },
    );
    c = render(() => <Logs log={log} version={0} live={false} />, target);
    flushSync();

    // Long tokens render statically with .w.long and short tokens with .w.
    const words = [...target.querySelectorAll<HTMLElement>('.line .w')];
    expect(words.some((w) => w.classList.contains('long') && w.textContent?.includes('checkpoint.folo'))).toBe(true);
    expect(words.some((w) => !w.classList.contains('long') && w.textContent === 'saved')).toBe(true);

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Search logs"]')!;
    expect(search).not.toBeNull();
    search.value = 'MEMORY';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    expect([...target.querySelectorAll('.line .m')].map((el) => el.textContent)).toEqual(['CUDA out of memory']);
    expect(target.querySelector('.matches')?.textContent?.trim()).toBe('1 line');

    // Empty search result shows clear message and disables Copy visible.
    search.value = 'nonexistent-token';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('.empty')?.textContent).toContain('No log lines match "nonexistent-token"');
    const copyBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Copy visible'))!;
    expect(copyBtn.disabled).toBe(true);

    // Clearing search restores all lines.
    search.value = '';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelectorAll('.line')).toHaveLength(4);
    expect(copyBtn.disabled).toBe(false);
  });

  it('copies only the currently visible filtered lines via Copy visible', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const log = new AppendLog<LogLine>();
    log.push(
      { t: '2026-11-12T10:00:01Z', level: 'info', message: 'epoch 1 done' },
      { t: '2026-11-12T10:00:02Z', level: 'warn', message: 'slow loader' },
      { t: '2026-11-12T10:00:03Z', level: 'error', message: 'loader failed' },
    );
    c = render(() => <Logs log={log} version={0} live={false} />, target);
    flushSync();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Search logs"]')!;
    search.value = 'loader';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    const copyBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Copy visible'))!;
    copyBtn.click();
    await Promise.resolve();
    expect(writeText).toHaveBeenCalledTimes(1);
    const copied = writeText.mock.calls[0]![0] as string;
    const lines = copied.split('\n');
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain('warn slow loader');
    expect(lines[1]).toContain('error loader failed');
  });

  it('pins to whole top line using inner.children without reading scrollHeight twice or calling querySelectorAll', () => {
    const log = new AppendLog<LogLine>();
    for (let i = 0; i < 20; i++) log.push(line(i));
    const version = signal(0);
    c = render(() => <Logs log={log} version={version.get()} live />, target);
    flushSync();

    const boxEl = target.querySelector<HTMLDivElement>('.log')!;
    const rows = [...boxEl.querySelectorAll<HTMLElement>('.line')];
    rows.forEach((row, idx) => {
      Object.defineProperty(row, 'offsetTop', { value: 8 + idx * 20, configurable: true });
    });

    let scrollHeightReads = 0;
    let tail = 0;
    Object.defineProperty(boxEl, 'clientHeight', { value: 110, configurable: true });
    Object.defineProperty(boxEl, 'scrollHeight', {
      get: () => {
        scrollHeightReads++;
        return 8 + rows.length * 20 + 8 + tail;
      },
      configurable: true,
    });
    const origSetProp = boxEl.style.setProperty.bind(boxEl.style);
    vi.spyOn(boxEl.style, 'setProperty').mockImplementation((prop, val) => {
      if (prop === '--tail') tail = parseFloat(val ?? '0') || 0;
      origSetProp(prop, val);
    });
    const qsaSpy = vi.spyOn(boxEl, 'querySelectorAll');

    // Trigger a re-pin via appending a line.
    log.push(line(20));
    version.set(1);
    flushSync();
    const newRow = boxEl.querySelectorAll<HTMLElement>('.line')[20]!;
    Object.defineProperty(newRow, 'offsetTop', { value: 8 + 20 * 20, configurable: true });
    qsaSpy.mockClear();
    scrollHeightReads = 0;

    // Trigger follow-pin via another append (all rows now have mock offsetTop).
    log.push(line(21));
    version.set(2);
    flushSync();

    expect(qsaSpy).not.toHaveBeenCalled();
    expect(scrollHeightReads).toBe(1);
    // Base height = 8 + 20*20 + 8 = 416, clientHeight = 110 -> end = 306.
    // First row with offsetTop - 8 >= 305.5 is row 16 (offsetTop = 328 -> 320 >= 305.5), next = 14px, scrollTop = 320.
    expect(tail).toBe(14);
    expect(boxEl.scrollTop).toBe(320);
  });
});
