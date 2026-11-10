import { describe, expect, it } from 'vitest';
import type { BenchPlan } from '../api/types';
import { checkPredictions, parsePredictions, type PredictionContext } from './external';

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
    expect(err([{ image: '1' }])).toMatch(/"boxes" must be an array/);
    expect(err([...clean, clean[0]])).toMatch(/^Record 3: image "1" is answered twice/);
    expect(err([...clean, { image: '1', condition: 'rain', severity: 1, boxes: [] }])).toMatch(/unknown condition "rain"/);
    expect(err([...clean, { image: '1', condition: 'low_light', severity: 1, boxes: [] }])).toMatch(/Low light is not in this benchmark's plan/);
    expect(err([...clean, { image: '1', condition: 'haze', severity: 2, boxes: [] }])).toMatch(/Haze runs at severities 1, 3, not 2/);
    expect(err([...clean, { image: '1', condition: 'haze', boxes: [] }])).toMatch(/needs a "severity"/);
    expect(err([{ image: '1', severity: 2, boxes: [] }])).toMatch(/"severity" without a "condition"/);
    expect(err([clean[0]])).toMatch(/Image "2" has no record without a condition/);
    expect(err([...clean, { image: '1', condition: 'haze', severity: 1, boxes: [] }])).toMatch(/Haze is answered in part: image "1" at severity 3 is missing|image "2" at severity 1 is missing/);
  });

  it('without the image list, checks the rest and counts what it got', () => {
    const r = checkPredictions(clean, { ...ctx, imageIds: null });
    expect(r.ok && r.summary.images).toBe(2);
  });
});
