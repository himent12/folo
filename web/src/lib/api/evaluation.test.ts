import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fromModelEval } from './evaluation';
import { MockApi } from './mock';

describe('model evaluation (contract of GET /api/models/{id}/eval)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.parse('2026-11-08T14:30:00Z'));
  });
  afterEach(() => vi.useRealTimers());

  it('serves 101-point envelopes, a background-padded confusion and matching AP', async () => {
    const api = new MockApi('normal', 0);
    const e = await api.modelEval('m_fast_n');
    const n = (await api.model('m_fast_n')).classes.length;
    expect(e.pr).toHaveLength(n);
    for (const c of e.pr) {
      expect(c.recall).toHaveLength(101);
      expect(c.precision).toHaveLength(101);
      for (let i = 1; i < 101; i++) expect(c.precision[i]!).toBeLessThanOrEqual(c.precision[i - 1]!);
      expect(c.ap50).toBeCloseTo(c.precision.reduce((a, b) => a + b, 0) / 101, 3);
      expect(e.per_class[c.class]!.map50).toBe(c.ap50);
    }
    expect(e.confusion).toHaveLength(n + 1);
    expect(e.confusion[n]![n]).toBe(0);
  });

  it('converts to view data with instances from confusion rows', async () => {
    const api = new MockApi('normal', 0);
    const ev = fromModelEval(await api.modelEval('m_shapes_n'));
    expect(ev.curves).toHaveLength(3);
    expect(ev.mean).toHaveLength(101);
    expect(ev.perClass[0]!.instances).toBe(ev.confusion[0]!.reduce((a, b) => a + b, 0));
  });

  it('serves log history and the api generation', async () => {
    const api = new MockApi('normal', 0);
    expect((await api.health()).api).toBe('v1');
    const logs = await api.runLogs('run_7d1e', 5);
    expect(logs.lines.length).toBeLessThanOrEqual(5);
    expect(logs.lines.map((l) => l.t)).toEqual([...logs.lines.map((l) => l.t)].sort());
  });
});
