// The run page's schedule warnings read as the log streams (./warningsLog) are `scheduleWarnings` over the whole log
// after every update: through appends, the cap trimming warnings off the front, and a history load replacing the list.
import { describe, expect, it } from 'vitest';
import { AppendLog } from '$lib/api/appendLog';
import type { LogLine } from '$lib/api/types';
import { scheduleWarnings } from './warnings';
import { WarningsLog } from './warningsLog';

const SHORT = 'this run takes only 125 optimizer steps (5 epochs × 25: the model barely learns); Train for at least 8 epochs';
const WARMUP = "warmup takes 75 of the run's 125 optimizer steps";

describe('WarningsLog', () => {
  it('finds what scheduleWarnings finds, after every update', () => {
    let seed = 9;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    const log = new AppendLog<LogLine>();
    const w = new WarningsLog();
    let n = 0;
    const line = (): LogLine => {
      const u = rnd();
      const t = `t${n++}`;
      if (u < 0.01) return { t, level: 'warn', message: SHORT };
      if (u < 0.02) return { t, level: 'warn', message: WARMUP };
      if (u < 0.03) return { t, level: 'info', message: SHORT };
      return { t, level: u < 0.1 ? 'warn' : 'info', message: `line ${n}` };
    };
    let last: string[] | undefined;
    for (let round = 0; round < 400; round++) {
      for (let k = Math.floor(rnd() * 30); k >= 0; k--) log.push(line());
      if (log.length > 300) log.trimFront(log.length - 300);
      if (round % 97 === 96) log.replace(log.items.slice(50));
      const got = w.sync(log);
      const want = scheduleWarnings(log.items);
      expect(got).toEqual(want);
      // An update that changes nothing keeps the same list (its readers see no change).
      if (last && JSON.stringify(last) === JSON.stringify(got)) expect(got).toBe(last);
      last = got;
    }
  });
});
