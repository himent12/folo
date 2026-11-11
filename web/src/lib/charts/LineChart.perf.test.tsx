// @vitest-environment happy-dom
// The live-metrics path: what an idle or growing chart costs. LineChart is where training metrics stream in (Overview,
// a run's Loss and Metrics tabs), so these pin the port's three savings over the Svelte original: no y-axis ease (and
// no animation frames) for a domain that did not move, path strings rebuilt only for the lines whose points changed,
// and a cursor that touches no line.
import { deep, flush } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import LineChart from './LineChart';
import { memoPathsOf } from './linePathsMemo';
import { pathsOf } from './linePaths';
import type { Series } from './lineSeries';

const pts = (n: number, f: (x: number) => number) => Array.from({ length: n }, (_, x) => ({ x, y: f(x) }));
const copy = (series: Series[]) => series.map((s) => ({ ...s, points: s.points.map((p) => ({ ...p })) }));

function mountAt(props: { series: Series[]; yDomain?: [number, number] }, w = 640) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = render(() => <LineChart series={props.series} yDomain={props.yDomain} label="perf" endDots />, target);
  Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
  flush();
  return { target, c, svg: target.querySelector<SVGSVGElement>('svg[aria-roledescription=chart]')! };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('LineChart on a live page', () => {
  it('schedules no animation frame when a poll brings an unchanged domain, and stops its loop once an ease ends', () => {
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame', 'performance'] });
    const raf = vi.spyOn(window, 'requestAnimationFrame');
    const series = [
      { id: 'a', label: 'box', color: 'red', points: pts(40, (x) => 2 - x / 40) },
      { id: 'b', label: 'cls', color: 'blue', points: pts(40, (x) => 1.5 - x / 60) },
    ];
    const props = deep({ series });
    const { c } = mountAt(props);
    raf.mockClear();
    // Ten polls of equal data, in new arrays, 250 ms apart (the live page's refresh): the domain does not move.
    for (let i = 0; i < 10; i++) {
      props.series = copy(series);
      flush();
      vi.advanceTimersByTime(250);
    }
    const idle = raf.mock.calls.length;
    expect(idle).toBe(0);
    // A domain that moves eases over 240 ms, then the loop stops: no frame is asked for after it ends.
    props.series = [...copy(series), { id: 'c', label: 'dfl', color: 'green', points: pts(40, (x) => 4 - x / 20) }];
    flush();
    vi.advanceTimersByTime(400);
    const eased = raf.mock.calls.length;
    expect(eased).toBeGreaterThan(5);
    vi.advanceTimersByTime(1000);
    expect(raf.mock.calls.length).toBe(eased);
    console.info(`[perf] idle chart, 10 polls over 2.5 s: ${idle} animation frames; one re-fit: ${eased} frames, then none`);
    c.unmount();
  });

  it('leaves every line, area and end mark untouched while the cursor moves', () => {
    const series = [
      { id: 'a', label: 'box', color: 'red', area: true, points: pts(40, (x) => 2 - x / 40) },
      { id: 'b', label: 'cls', color: 'blue', pattern: 'dot' as const, end: 'live' as const, points: pts(30, (x) => 1.5 - x / 60) },
    ];
    const { target, c, svg } = mountAt({ series });
    const data = target.querySelector('g[clip-path]')!;
    const lines = [...data.querySelectorAll('path')];
    const ends = [...svg.querySelectorAll('.end-dot')];
    expect(lines.length).toBeGreaterThan(2);
    expect(ends.length).toBe(2);
    const seen: MutationRecord[] = [];
    const watch = new MutationObserver((m) => seen.push(...m));
    watch.observe(data, { subtree: true, childList: true, attributes: true, characterData: true });
    for (const e of ends) watch.observe(e, { attributes: true });
    for (let px = 60; px < 600; px += 20) {
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: px, clientY: 50, bubbles: true }));
      flush();
    }
    expect(target.querySelector('.cursor')).not.toBeNull();
    seen.push(...watch.takeRecords());
    watch.disconnect();
    expect(seen).toEqual([]);
    // The same nodes: nothing was re-created either.
    expect([...data.querySelectorAll('path')]).toEqual(lines);
    expect([...svg.querySelectorAll('.end-dot')]).toEqual(ends);
    c.unmount();
  });
});

describe('memoPathsOf', () => {
  const scale = (k: number) => (v: number) => v * k;
  const x = scale(10);
  const y = scale(-50);
  const margin = { top: 8, right: 12, bottom: 22, left: 40 };
  const run = (id: string, n: number) => ({ id, label: id, color: 'red', points: pts(n, (i) => Math.sin(i / 50) + i / 1e4) });

  it('draws what pathsOf draws, and rebuilds only the line whose points changed', () => {
    const memo = memoPathsOf();
    const runs: Series[] = [run('a', 5000), run('b', 5000), run('c', 5000), { ...run('d', 600), lower: pts(600, (i) => i / 1e3) }];
    expect(memo(runs, 'linear', 900, margin, 200, x, y, false)).toEqual(pathsOf(runs, 'linear', 900, margin, 200, x, y, false));
    // A poll with equal data in new arrays: nothing is rebuilt.
    memo.stats.built = 0;
    expect(memo(copy(runs), 'linear', 900, margin, 200, x, y, false)).toEqual(pathsOf(runs, 'linear', 900, margin, 200, x, y, false));
    expect(memo.stats.built).toBe(0);
    // One run gets a new point: that one line is rebuilt.
    const grown = [...runs.slice(0, 3), { ...runs[3]!, points: [...runs[3]!.points, { x: 600, y: 0.4 }], lower: [...runs[3]!.lower!, { x: 600, y: 0.3 }] }];
    expect(memo(grown, 'linear', 900, margin, 200, x, y, false)).toEqual(pathsOf(grown, 'linear', 900, margin, 200, x, y, false));
    expect(memo.stats.built).toBe(1);
    // A new scale redraws everything.
    expect(memo(grown, 'linear', 900, margin, 200, scale(11), y, false)).toEqual(pathsOf(grown, 'linear', 900, margin, 200, scale(11), y, false));
    expect(memo.stats.built).toBe(5);

    // What it saves on a live chart: three long runs and one growing one, one new point per poll (timed after a warm-up
    // round, on a 900 px plot's scales).
    const px = (v: number) => 40 + v * 0.17;
    const py = (v: number) => 178 - v * 50;
    const polls = 200;
    const live = Array.from({ length: polls }, (_, i) => [...copy(runs.slice(0, 3)), run('d', 600 + i)]);
    const perPoll = (f: (s: Series[]) => unknown) => {
      for (const s of live.slice(0, 20)) f(s);
      const t0 = performance.now();
      for (const s of live) f(s);
      return (performance.now() - t0) / polls;
    };
    const fresh = memoPathsOf();
    const full = perPoll((s) => pathsOf(s, 'linear', 900, margin, 200, px, py, false));
    const kept = perPoll((s) => fresh(s, 'linear', 900, margin, 200, px, py, false));
    console.info(`[perf] a poll of 3 × 5000 points + 1 growing line: pathsOf ${full.toFixed(3)} ms, memoPathsOf ${kept.toFixed(3)} ms`);
  });
});
