import { describe, expect, it } from 'vitest';
import { jsonLines, jsonText, jsonTokens } from './json';

describe('jsonText', () => {
  const v = {
    model_id: 'm_1',
    image: { width: 640, height: 480 },
    boxes: [
      { class: 0, name: 'bolt', score: 0.93, x1: 1, y1: 2, x2: 3, y2: 4 },
      { class: 1, name: 'nut "m6"', score: 0.5, x1: 5, y1: 6, x2: 7, y2: 8 },
    ],
    empty: [],
    ok: true,
    none: null,
  };

  it('round-trips to the same value', () => {
    expect(JSON.parse(jsonText(v))).toEqual(v);
  });

  it('keeps flat objects on one line', () => {
    const lines = jsonText(v).split('\n');
    expect(lines).toContain('  "image": { "width": 640, "height": 480 },');
    expect(lines.filter((l) => l.includes('"score"'))).toHaveLength(2);
  });

  it('classifies tokens for tinting', () => {
    const t = jsonTokens({ a: 'x', b: 1, c: false });
    expect(t.filter((x) => x.t === 'key').map((x) => x.v)).toEqual(['"a"', '"b"', '"c"']);
    expect(t.find((x) => x.t === 'num')?.v).toBe('1');
    expect(t.find((x) => x.t === 'lit')?.v).toBe('false');
  });

  it('splits into lines with their indents, for a view that wraps long ones', () => {
    const lines = jsonLines(jsonTokens(v));
    expect(lines.map((l) => ' '.repeat(l.indent) + l.tokens.map((t) => t.v).join(''))).toEqual(jsonText(v).split('\n'));
    expect(lines.find((l) => l.tokens.some((t) => t.v === '"bolt"'))?.indent).toBe(4);
    expect(lines.every((l) => l.tokens.every((t) => !t.v.includes('\n')))).toBe(true);
  });
});
