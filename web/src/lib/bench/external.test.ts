import { describe, expect, it } from 'vitest';
import type { BenchEntry, BenchPlan, BenchPrediction } from '../api/types';
import { ApiError } from '../api/client';
import { checkPredictions, estimateBytes, ExternalSendError, PART_BYTES, parsePredictions, partsOf, recordBytes, sendExternal, type ExternalSender, type PredictionContext } from './external';

const plan: BenchPlan = { conditions: [{ condition: 'haze', severities: [1, 3], weight: 1 }, { condition: 'jpeg', severities: [5], weight: 1 }] };
const ctx: PredictionContext = {
  imageIds: ['1', '2'],
  classes: 3,
  plan,
  catalog: ['haze', 'jpeg', 'low_light'],
  names: { haze: 'Haze', jpeg: 'JPEG compression', low_light: 'Low light' },
};
const box = { class: 0, score: 0.9, x1: 1, y1: 2, x2: 30, y2: 40 };
const clean = [
  { image: '1', boxes: [box] },
  { image: '2', boxes: [] },
];

describe('parsePredictions', () => {
  it('takes { "predictions": [...] } or a bare array, and says why anything else is not', () => {
    expect(parsePredictions('{"predictions":[{"image":"1","boxes":[]}]}')).toEqual({ records: [{ image: '1', boxes: [] }] });
    expect(parsePredictions('[]')).toEqual({ records: [] });
    expect(parsePredictions('{"boxes":[]}')).toEqual({ error: expect.stringMatching(/"predictions" array/) });
    expect(parsePredictions('{nope')).toEqual({ error: expect.stringMatching(/^The file is not JSON/) });
  });
});

describe('checkPredictions', () => {
  it('passes every unchanged image answered, conditions whole or left out, and says what it holds', () => {
    const r = checkPredictions(clean, ctx);
    expect(r.ok && r.summary).toEqual({ images: 2, conditions: [], missing: ['haze', 'jpeg'], boxes: 1, scores: true });
    const withHaze = [...clean, ...['1', '2'].flatMap((image) => [1, 3].map((severity) => ({ image, condition: 'haze', severity, boxes: [] })))];
    const h = checkPredictions(withHaze, ctx);
    expect(h.ok && h.summary.conditions).toEqual(['haze']);
    expect(h.ok && h.summary.missing).toEqual(['jpeg']);
    // No score anywhere: the model gives no confidences.
    const bare = checkPredictions([{ image: '1', boxes: [{ class: 1, x1: 0, y1: 0, x2: 5, y2: 5 }] }, { image: '2', boxes: [] }], ctx);
    expect(bare.ok && bare.summary.scores).toBe(false);
    // Without the image list, it checks the rest and counts what it got.
    const unlisted = checkPredictions(clean, { ...ctx, imageIds: null });
    expect(unlisted.ok && unlisted.summary.images).toBe(2);
  });

  it('names the first record that breaks a rule, as the server does', () => {
    const err = (records: unknown[]) => {
      const r = checkPredictions(records, ctx);
      return r.ok ? null : r.error;
    };
    expect(err([])).toMatch(/no records/);
    expect(err([...clean, { image: '9', boxes: [] }])).toMatch(/^Record 3: the benchmark has no image "9"/);
    expect(err([{ image: '1', boxes: [{ ...box, class: 3 }] }])).toMatch(/^Record 1: box 1: "class" must be a class index from 0 to 2/);
    expect(err([{ image: '1', boxes: [{ ...box, score: 2 }] }])).toMatch(/"score" must be between 0 and 1/);
    expect(err([{ image: '1', boxes: [{ class: 0, x1: 'a' }] }])).toMatch(/must be numbers/);
    // A coordinate the server's 32-bit floats overflow, and corners the wrong way round (x, y, w, h sent as corners).
    expect(err([{ image: '1', boxes: [{ ...box, x2: 1e39 }] }])).toMatch(/must be numbers/);
    expect(err([{ image: '1', boxes: [{ ...box, x1: box.x2 + 1 }] }])).toMatch(/^Record 1: box 1: "x2" or "y2" is less than "x1" or "y1".*not x, y, width, height/);
    expect(err([{ image: '1', boxes: [{ ...box, y1: box.y2 + 1 }] }])).toMatch(/is less than/);
    expect(err([{ image: '1', boxes: [{ ...box, x2: box.x1 }] }, clean[1]])).toBeNull();
    expect(err([{ image: '1' }])).toMatch(/"boxes" must be an array/);
    expect(err([...clean, { image: '1', boxes: [] }])).toMatch(/^Record 3: image "1" is answered twice/);
    expect(err([...clean, { image: '1', condition: 'rain', severity: 1, boxes: [] }])).toMatch(/unknown condition "rain"/);
    expect(err([...clean, { image: '1', condition: 'low_light', severity: 1, boxes: [] }])).toMatch(/Low light is not in this benchmark's plan/);
    expect(err([...clean, { image: '1', condition: 'haze', severity: 2, boxes: [] }])).toMatch(/Haze runs at severities 1, 3, not 2/);
    expect(err([...clean, { image: '1', condition: 'haze', boxes: [] }])).toMatch(/needs a "severity"/);
    expect(err([{ image: '1', severity: 2, boxes: [] }])).toMatch(/"severity" without a "condition"/);
    expect(err([clean[0]])).toMatch(/Image "2" has no record without a condition/);
    expect(err([...clean, { image: '1', condition: 'haze', severity: 1, boxes: [] }])).toMatch(/Haze is answered in part: image "1" at severity 3 is missing|image "2" at severity 1 is missing/);
  });
});

describe('a large file, sent in parts', () => {
  const rec = (image: string, condition: string | null, severity: number | null, n: number): BenchPrediction => ({
    image,
    ...(condition ? { condition, severity } : {}),
    boxes: Array.from({ length: n }, () => ({ ...box })),
  });
  // Every image under every pass of the plan, in a shuffled order (a file need not be in pass order).
  const all = [
    ...['2', '1'].map((i) => rec(i, 'jpeg', 5, 3)),
    ...['1', '2'].map((i) => rec(i, null, null, 3)),
    ...['1', '2'].flatMap((i) => [3, 1].map((s) => rec(i, 'haze', s, 3))),
  ];

  it('groups the parts by pass, the unchanged images first, then the plan’s order; every record in exactly one', () => {
    const parts = partsOf(all, plan, 10_000_000);
    expect(parts.map((p) => p.map((r) => `${r.condition ?? '-'}${r.severity ?? ''}:${r.image}`))).toEqual([
      ['-:1', '-:2'],
      ['haze1:1', 'haze1:2'],
      ['haze3:1', 'haze3:2'],
      ['jpeg5:2', 'jpeg5:1'],
    ]);
    expect(parts.flat()).toHaveLength(all.length);
    expect(new Set(parts.flat())).toEqual(new Set(all));
  });

  it('splits a pass further by size, never a part over the limit unless one record is, never an empty one', () => {
    const big = Array.from({ length: 50 }, (_, k) => rec(String(k + 1), null, null, 40));
    const one = recordBytes(big[0]!);
    const limit = 32 + one * 7 + 10;
    const parts = partsOf(big, plan, limit);
    expect(parts.map((p) => p.length)).toEqual([7, 7, 7, 7, 7, 7, 7, 1]);
    for (const p of parts) expect(estimateBytes(p)).toBeLessThanOrEqual(limit);
    // A record over the limit on its own is a part of its own.
    expect(partsOf([rec('1', null, null, 1000)], plan, 100).map((p) => p.length)).toEqual([1]);
    expect(partsOf([], plan)).toEqual([]);
    // The estimate is on the high side of the JSON as sent.
    for (const r of all) expect(recordBytes(r)).toBeGreaterThan(JSON.stringify(r).length);
    // At the real limit, the real case (300 images × 20 passes, 300 boxes an image) is many parts each under 8 MiB.
    const real = Array.from({ length: 300 }, (_, k) => rec(String(k), null, null, 300));
    const rp = partsOf(real, plan);
    expect(rp.length).toBeGreaterThan(1);
    for (const p of rp) expect(estimateBytes(p)).toBeLessThanOrEqual(PART_BYTES);
  });

  it('takes a record repeated with the same boxes once, as the server does, and refuses one repeated with other boxes', () => {
    // docs/API.md: "A record that repeats one with exactly the same boxes ... is accepted and ignored".
    const r = checkPredictions([...clean, { ...clean[0]!, boxes: [{ ...box }] }], ctx);
    expect(r.ok && r.summary).toEqual({ images: 2, conditions: [], missing: ['haze', 'jpeg'], boxes: 1, scores: true });
    expect(r.ok && r.predictions).toHaveLength(2);
    const other = checkPredictions([...clean, { image: '1', boxes: [{ ...box, x2: 31 }] }], ctx);
    expect(!other.ok && other.error).toMatch(/^Record 3: image "1" is answered twice/);
  });

  it('compares a repeat by its boxes as the server reads them, not by their JSON', () => {
    // crates/folo-server/src/bench/external.rs `content`: per box in order, class, score (1 when absent; not at all
    // for an unscored entry) and corners as 32-bit floats. The order of a box's fields and other fields don't count.
    const same = (b: Record<string, unknown>, c: PredictionContext = ctx) => checkPredictions([...clean, { image: '1', boxes: [b] }], c);
    const { class: cls, score, ...corners } = box;
    expect(same({ ...corners, score, class: cls }).ok).toBe(true);
    expect(same({ ...box, note: 'mine' }).ok).toBe(true);
    expect(same({ ...box, x1: box.x1 + 1e-9 }).ok).toBe(true);
    // A missing score is 1 to the server.
    const first = { image: '1', boxes: [{ ...box, score: 1 }] };
    expect(checkPredictions([first, clean[1], { image: '1', boxes: [{ class: cls, ...corners }] }], ctx).ok).toBe(true);
    // Another score is another answer, unless the entry gives no confidences.
    expect(!same({ ...box, score: 0.5 }).ok).toBe(true);
    expect(same({ ...box, score: 0.5 }, { ...ctx, scored: false }).ok).toBe(true);
    // Zero's sign survives on the server (`to_bits`): -0 is another corner.
    const zero = [{ image: '1', boxes: [{ ...box, x1: 0 }] }, clean[1]];
    expect(!checkPredictions([...zero, { image: '1', boxes: [{ ...box, x1: -0 }] }], ctx).ok).toBe(true);
  });

  it('refuses a record with more than 1000 boxes, and a part that repeats a record received before', () => {
    const r = checkPredictions([rec('1', null, null, 1001), clean[1]], ctx);
    expect(!r.ok && r.error).toMatch(/^Record 1: 1,001 boxes/);
    const seen = new Set<string>();
    expect(checkPredictions([clean[0]], ctx, { part: true, seen }).ok).toBe(true);
    const again = checkPredictions([clean[1], clean[0]], ctx, { part: true, seen: new Set(seen) });
    expect(!again.ok && again.error).toMatch(/^Record 2: image "1" is answered twice/);
    // A part need not be complete: the finish says what is missing.
    expect(checkPredictions([{ image: '1', condition: 'haze', severity: 1, boxes: [] }], ctx, { part: true, seen: new Set() }).ok).toBe(true);
  });

  /** A server that records the calls; `failPart` refuses the n-th part (1-based), `dropPart` loses its answer once
   *  (the network), `failFinish` refuses the finish, `hangPart` never answers, `failStop` / `failRemove` fail the clean-up. */
  function fake(o: { failPart?: number; dropPart?: number[]; failFinish?: boolean; dropFinish?: boolean; hangPart?: number; failStop?: boolean; failRemove?: boolean } = {}) {
    const calls: string[] = [];
    const entry = (status: BenchEntry['status']) => ({ id: 'be_x', status }) as BenchEntry;
    let n = 0;
    /** What the server holds: done once a finish went through, whether or not its answer arrived. */
    let held: BenchEntry['status'] = 'running';
    const api: ExternalSender = {
      start: async (b) => (calls.push(`start open=${(b.contestant as { open?: boolean }).open ?? false} records=${(b.contestant as { predictions?: unknown[] }).predictions?.length}`), entry(b.contestant.kind === 'external' && (b.contestant as { open?: boolean }).open ? 'running' : 'done')),
      send: async (_, ps) => {
        n++;
        calls.push(`send ${ps.length}`);
        if (n === o.hangPart) return new Promise<BenchEntry>(() => {});
        if (n === o.failPart) throw new ApiError(400, 'invalid_body', 'Record 1: the benchmark has no image "9".');
        if (o.dropPart?.includes(n)) throw new ApiError(0, 'network', 'Cannot reach the FOLO server.');
        return entry('running');
      },
      finish: async () => {
        calls.push('finish');
        if (o.failFinish) throw new ApiError(400, 'invalid_body', 'Image "2" has no record without a condition.');
        if (held !== 'running') throw new ApiError(409, 'bench_entry_not_open', `big is not an open external entry (it is ${held}).`);
        held = 'done';
        if (o.dropFinish) throw new ApiError(0, 'network', 'Cannot reach the FOLO server.');
        return entry('done');
      },
      get: async () => (calls.push('get'), entry(held)),
      stop: async () => {
        calls.push('stop');
        if (o.failStop) throw new ApiError(0, 'network', 'Cannot reach the FOLO server.');
        if (held !== 'running') throw new ApiError(409, 'bench_entry_not_active', 'big has ended.');
        return entry('stopped');
      },
      remove: async () => {
        calls.push('remove');
        if (o.failRemove) throw new ApiError(0, 'network', 'Cannot reach the FOLO server.');
      },
    };
    return { api, calls };
  }
  const body = { name: 'big', scored: true, predictions: all, plan };

  it('sends a small file in one request, a large one open, in parts, then finishes, telling the records sent', async () => {
    const small = fake();
    expect((await sendExternal(small.api, body)).status).toBe('done');
    expect(small.calls).toEqual([`start open=false records=${all.length}`]);
    const large = fake();
    const seen: number[] = [];
    const e = await sendExternal(large.api, body, (p) => seen.push(p.sent), undefined, 200);
    expect(e.status).toBe('done');
    expect(large.calls[0]).toBe('start open=true records=0');
    expect(large.calls.at(-1)).toBe('finish');
    expect(large.calls.filter((c) => c.startsWith('send')).length).toBeGreaterThanOrEqual(4);
    expect(seen[0]).toBe(0);
    expect(seen.at(-1)).toBe(all.length);
    expect(seen).toEqual([...seen].sort((a, b) => a - b));
  });

  it('leaves nothing behind when a part or the finish fails, or the person stops it: the entry is stopped and deleted', async () => {
    const part = fake({ failPart: 2 });
    await expect(sendExternal(part.api, body, () => {}, undefined, 200)).rejects.toMatchObject({ message: expect.stringMatching(/no image "9"/), left: null });
    expect(part.calls.slice(-2)).toEqual(['stop', 'remove']);
    expect(part.calls).not.toContain('finish');
    // A refused part is not retried: the server said no.
    expect(part.calls.filter((c) => c.startsWith('send'))).toHaveLength(2);
    const fin = fake({ failFinish: true });
    await expect(sendExternal(fin.api, body, () => {}, undefined, 200)).rejects.toThrow(/has no record/);
    expect(fin.calls.slice(-3)).toEqual(['finish', 'stop', 'remove']);
    // Stopped by the person while a part is in flight: not waited for, the entry stopped and deleted.
    const hang = fake({ hangPart: 2 });
    const ctl = new AbortController();
    const p = sendExternal(hang.api, body, (s) => s.sent > 0 && ctl.abort(), ctl.signal, 200);
    await expect(p).rejects.toMatchObject({ name: 'AbortError', left: null });
    expect(hang.calls.slice(-2)).toEqual(['stop', 'remove']);
  });

  it('says exactly what is left when the clean-up fails too', async () => {
    const stopFails = fake({ failPart: 1, failStop: true });
    const e1 = await sendExternal(stopFails.api, body, () => {}, undefined, 200).catch((e: unknown) => e);
    expect(e1).toBeInstanceOf(ExternalSendError);
    expect((e1 as ExternalSendError).left).toMatch(/“big”.*stop and delete it/);
    expect((e1 as ExternalSendError).cause).toMatchObject({ code: 'invalid_body' });
    const removeFails = fake({ failPart: 1, failRemove: true });
    const e2 = (await sendExternal(removeFails.api, body, () => {}, undefined, 200).catch((e: unknown) => e)) as ExternalSendError;
    expect(e2.left).toMatch(/“big” \(stopped\).*: delete it/);
  });

  it('takes the entry as scored when the finish went through but its answer was lost', async () => {
    // The finish is sent again; the server, which scored it the first time, answers 409 bench_entry_not_open.
    const lost = fake({ dropFinish: true });
    const e = await sendExternal(lost.api, body, () => {}, undefined, 200);
    expect(e.status).toBe('done');
    expect(lost.calls).not.toContain('stop');
    expect(lost.calls).not.toContain('remove');
  });

  it('sends a part again once when its answer was lost, and gives up the second time', async () => {
    const once = fake({ dropPart: [2] });
    expect((await sendExternal(once.api, body, () => {}, undefined, 200)).status).toBe('done');
    const sends = once.calls.filter((c) => c.startsWith('send'));
    // Part 2 went twice: the server takes the same records again without adding them.
    expect(sends.length).toBe(partsOf(all, plan, 200).length + 1);
    const twice = fake({ dropPart: [2, 3] });
    await expect(sendExternal(twice.api, body, () => {}, undefined, 200)).rejects.toMatchObject({ cause: { code: 'network' }, left: null });
    expect(twice.calls.slice(-2)).toEqual(['stop', 'remove']);
  });
});
