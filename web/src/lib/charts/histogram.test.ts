import { describe, expect, it } from 'vitest';
import { binAt, categoryBars, edgeX } from './histogram';

describe('categoryBars', () => {
  it('spaces every bin evenly, whatever its value span', () => {
    expect(categoryBars(4, 40, 440)).toEqual([
      { x: 40, w: 100 },
      { x: 140, w: 100 },
      { x: 240, w: 100 },
      { x: 340, w: 100 },
    ]);
    expect(categoryBars(0, 0, 100)).toEqual([]);
  });
});

describe('edgeX', () => {
  // The server's box-size bins: <1%, 1–2%, 2–5%, 5–10%, 10–20%, 20–50%, ≥50%.
  const edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
  const bars = categoryBars(7, 0, 700);
  it('puts a bin edge on the boundary between its two bars', () => {
    expect(edgeX(edges, bars, 0.05)).toBeCloseTo(300);
    expect(edgeX(edges, bars, 0)).toBe(0);
    expect(edgeX(edges, bars, 1)).toBe(700);
  });
  it('places a value inside a bin proportionally inside its bar, and clamps outside values', () => {
    expect(edgeX(edges, bars, 0.35)).toBeCloseTo(550);
    expect(edgeX(edges, bars, -1)).toBe(0);
    expect(edgeX(edges, bars, 3)).toBe(700);
  });
});

describe('binAt', () => {
  const bars = [
    { x: 40, w: 10 },
    { x: 51, w: 30 },
    { x: 82, w: 4 },
  ];
  it('finds the bar under the pointer', () => {
    expect(binAt(bars, 45)).toBe(0);
    expect(binAt(bars, 60)).toBe(1);
    expect(binAt(bars, 84)).toBe(2);
  });
  it('gives a gap between bars to the nearer bar, and nothing far outside', () => {
    expect(binAt(bars, 50.4)).toBe(0);
    expect(binAt(bars, 50.8)).toBe(1);
    expect(binAt(bars, 87)).toBe(2);
    expect(binAt(bars, 10)).toBeNull();
    expect(binAt(bars, 200)).toBeNull();
  });
});
