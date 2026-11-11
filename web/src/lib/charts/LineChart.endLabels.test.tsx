// @vitest-environment happy-dom
import { deep, flush as flushSync } from 'fw';
import { describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';

// Counts how often the chart places its end labels (the costly step), passing every call through.
const placed = vi.hoisted(() => ({ n: 0 }));
vi.mock('./endLabels', async (load) => {
  const real = await load<typeof import('./endLabels')>();
  return {
    ...real,
    placeEndLabels: real.placeEndLabels,
    memoPlaceEndLabels: () => {
      const place = real.memoPlaceEndLabels();
      let last: unknown;
      return (...args: Parameters<typeof real.placeEndLabels>) => {
        const out = place(...args);
        if (out !== last) placed.n++;
        last = out;
        return out;
      };
    },
  };
});

const { default: LineChart } = await import('./LineChart');

describe('LineChart end labels', () => {
  it('places them once per change of domain, not on every frame of its ease, each label riding with its line', () => {
    // The ease runs on animation frames: fake ones, stepped by hand.
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame', 'performance'] });
    const pts = (f: (x: number) => number) => Array.from({ length: 30 }, (_, x) => ({ x, y: f(x) }));
    const series = [
      { id: 'a', label: 'fasteners-s', color: 'red', points: pts((x) => 0.1 + x / 60) },
      { id: 'b', label: 'shapes-hard-s', color: 'blue', points: pts((x) => 0.05 + x / 120) },
    ];
    const props = deep({ series, label: 'cmp', endLabels: true, yDomain: [0, 1] as [number, number] });
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(() => <LineChart series={props.series} label={props.label} endLabels={props.endLabels} yDomain={props.yDomain} />, target);
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 900 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const label = (t: string) => [...target.querySelectorAll<SVGElement>('.end-label')].find((e) => e.textContent === t)!;
    const yOf = (t: string) => Number(label(t).getAttribute('y'));
    expect(label('fasteners-s')).toBeDefined();
    const before = placed.n;
    const startY = yOf('fasteners-s');
    // The axis re-fits: the lines glide to the new scale over 240 ms.
    props.yDomain = [0, 2];
    flushSync();
    const seen = new Set<number>();
    for (let t = 0; t < 400; t += 16) {
      vi.advanceTimersByTime(16);
      flushSync();
      seen.add(yOf('fasteners-s'));
    }
    // It moved through the ease (more than one height on the way)...
    expect(seen.size).toBeGreaterThan(2);
    expect(yOf('fasteners-s')).toBeGreaterThan(startY);
    // ...but was placed once for the new domain.
    expect(placed.n - before).toBe(1);
    // Re-rendering with equal data in new arrays places nothing.
    props.series = series.map((s) => ({ ...s, points: s.points.map((p) => ({ ...p })) }));
    flushSync();
    expect(placed.n - before).toBe(1);
    c.unmount();
    vi.useRealTimers();
  });
});
