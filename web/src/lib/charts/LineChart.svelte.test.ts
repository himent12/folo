// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import LineChart from './LineChart.svelte';

/** Mounts a chart with a plot width: happy-dom reports 0, and the plot renders once it has one. */
function mountAt(props: Record<string, unknown>, w = 640) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(LineChart, { target, props: props as never });
  Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
  target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
  flushSync();
  return { target, c, svg: target.querySelector<SVGSVGElement>('svg[aria-roledescription=chart]')! };
}
const pts = (f: (x: number) => number) => Array.from({ length: 30 }, (_, x) => ({ x, y: f(x) }));
const yTicks = (target: Element) => [...target.querySelectorAll('.ticks text[text-anchor="end"]')].map((t) => t.textContent);

describe('LineChart keyboard', () => {
  it('speaks keyboard-scrubbed values and goes quiet on pointer movement', async () => {
    const series = [{ id: 'a', label: 'mAP', color: 'red', points: [0, 1, 2, 3].map((x) => ({ x, y: x / 10 })) }];
    const { target, c, svg } = mountAt({ series, label: 'test chart', xTitle: (x: number) => `Epoch ${x}` }, 400);
    const live = target.querySelector('[aria-live]')!;
    // Focusable and interactive, so a group with a chart role description, not an image (whose content is opaque).
    expect(svg.getAttribute('role')).toBe('group');
    const key = async (k: string) => {
      svg.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
      flushSync();
      await tick();
      await tick();
    };
    await key('End');
    expect(live.textContent).toContain('Epoch 3');
    svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 10, bubbles: true }));
    flushSync();
    expect(live.textContent).toBe('');
    // End → Escape → End announces again (Escape clears the region, so the same text is new).
    await key('End');
    await key('Escape');
    expect(live.textContent).toBe('');
    await key('End');
    expect(live.textContent).toContain('Epoch 3');
    unmount(c);
  });

  it('hints at the arrow keys while the plot has keyboard focus, until a value is read out; never on a mouse focus', async () => {
    const { target, c, svg } = mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x / 30) }], label: 't', legendNote: 'Last' });
    // happy-dom has no :focus-visible: a mouse focus matches nothing, a keyboard focus matches it.
    svg.matches = (() => false) as unknown as typeof svg.matches;
    svg.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(target.querySelector('.legend .note')!.textContent).toBe('Last');
    svg.dispatchEvent(new FocusEvent('blur'));
    svg.matches = ((sel: string) => sel === ':focus-visible') as unknown as typeof svg.matches;
    svg.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(target.querySelector('.legend .note')!.textContent).toBe('← → to inspect');
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    flushSync();
    await tick();
    expect(target.querySelector('.legend .note')!.textContent).toMatch(/^\d+$/);
    svg.dispatchEvent(new FocusEvent('blur'));
    flushSync();
    expect(target.querySelector('.legend .note')!.textContent).toBe('Last');
    unmount(c);
  });
});

describe('LineChart touch readout', () => {
  it('pins the cursor on a tap, keeps it when the finger lifts, and lets go on a tap outside', () => {
    const series = [{ id: 'a', label: 'long', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) }];
    const { target, c, svg } = mountAt({ series, label: 'touch', xTitle: (x: number) => `Epoch ${x}` }, 400);
    const touch = (type: string, clientX: number, clientY = 50) => svg.dispatchEvent(new PointerEvent(type, { pointerType: 'touch', clientX, clientY, bubbles: true }));
    touch('pointerdown', 300);
    touch('pointerup', 300);
    touch('pointerleave', 300);
    flushSync();
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    // The cursor's x reads out on the axis.
    expect(target.querySelector('.pill text')!.textContent).toMatch(/^Epoch \d$/);
    document.body.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });

  it('lets go when the pinned cursor is tapped again, or on Escape', () => {
    const series = [{ id: 'a', label: 'a', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) }];
    const { target, c, svg } = mountAt({ series, label: 'touch' }, 600);
    const tap = (clientX: number) => {
      svg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX, clientY: 50, bubbles: true }));
      svg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX, clientY: 50, bubbles: true }));
      flushSync();
    };
    tap(300);
    const at = Number(target.querySelector('.cursor')!.getAttribute('x1'));
    tap(at + 4);
    expect(target.querySelector('.cursor')).toBeNull();
    tap(300);
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });
});

describe('LineChart pick', () => {
  it('picks out the line nearest the pointer where it is drawn, and a click acts on it', () => {
    // Round 26: the PR chart's grey class curves could be told apart only from the table.
    const flat = (id: string, y: number) => ({ id, label: id, color: 'grey', context: true, points: [0, 5, 10].map((x) => ({ x, y })) });
    const picks: string[] = [];
    const props = $state({
      series: [flat('c0', 0.2), flat('c1', 0.8), { id: 'mean', label: 'mean', color: 'black', points: [0, 10].map((x) => ({ x, y: 0.5 })) }],
      label: 'pr',
      height: 200,
      yDomain: [0, 1] as [number, number],
      legend: false,
      pick: (s: { id: string }) => s.id.startsWith('c'),
      picked: null as string | null,
      onpick: (id: string) => picks.push(id),
    });
    const { c, svg } = mountAt(props, 400);
    // The plot runs from y = 8 (1.0) to y = 178 (0.0): 0.8 is drawn at y = 42, 0.2 at 144, the mean at 93.
    const move = (clientX: number, clientY: number) => {
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX, clientY, bubbles: true }));
      flushSync();
    };
    move(200, 47);
    expect(props.picked).toBe('c1');
    svg.dispatchEvent(new PointerEvent('pointerup', { clientX: 200, clientY: 47, button: 0, bubbles: true }));
    expect(picks).toEqual(['c1']);
    move(200, 140);
    expect(props.picked).toBe('c0');
    // Near no pickable line (the mean is not one): nothing.
    move(200, 93);
    expect(props.picked).toBeNull();
    svg.dispatchEvent(new PointerEvent('pointerup', { clientX: 200, clientY: 93, button: 0, bubbles: true }));
    expect(picks).toEqual(['c1']);
    unmount(c);
  });
});

describe('LineChart ticks', () => {
  it('labels a short x domain with decimals instead of repeating integers', () => {
    const series = [{ id: 'a', label: 'loss', color: 'red', points: [0, 0.25, 0.5].map((x) => ({ x, y: 1 - x })) }];
    const { target, c } = mountAt({ series, label: 'short', xDomain: [0, 1] as [number, number] }, 600);
    const xs = [...target.querySelectorAll('.ticks text')].map((t) => t.textContent).slice(-6);
    expect(new Set(xs).size).toBe(xs.length);
    unmount(c);
  });

  it('draws 3–5 y ticks in nice steps on a domain the caller fixed, with the decimals the step needs', () => {
    const { target, c } = mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x / 30) }], label: 't', yDomain: [0, 1] });
    expect(yTicks(target)).toEqual(['0.00', '0.25', '0.50', '0.75', '1.00']);
    unmount(c);
  });

  it('labels every 1-2-5 value of a log axis up to six, each with its gridline; past that the ends and the decades', () => {
    const one = (d: [number, number]) => mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => 0.3 + x / 3) }], label: 't', yLog: true, yDomain: d });
    const a = one([0.2, 10]);
    expect(yTicks(a.target)).toEqual(['0.2', '0.5', '1', '2', '5', '10']);
    expect(a.target.querySelectorAll('.grid line').length).toBe(6);
    unmount(a.c);
    const b = one([0.01, 10]);
    expect(yTicks(b.target)).toEqual(['0.01', '0.1', '1', '10']);
    expect(b.target.querySelectorAll('.grid line.minor').length).toBe(6);
    unmount(b.c);
  });

  it('marks a phase change on x inside the domain, and nothing outside it', () => {
    const series = [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x) }];
    const inside = mountAt({ series, label: 't', xMark: { x: 25, label: 'mosaic off' } });
    expect(inside.target.querySelector('.mark-line')).not.toBeNull();
    unmount(inside.c);
    const out = mountAt({ series, label: 't', xMark: { x: 40, label: 'mosaic off' } });
    expect(out.target.querySelector('.mark-line')).toBeNull();
    unmount(out.c);
  });
});

describe('LineChart end labels and marks', () => {
  it('names the status after a label, which it never cuts, and keeps labels off a narrow plot', () => {
    const series = [
      { id: 'a', label: 'fasteners-s', color: 'red', end: 'live' as const, points: pts((x) => x / 30).slice(0, 12) },
      { id: 'b', label: 'shapes-hard-s', color: 'blue', end: 'stopped' as const, points: pts((x) => x / 40).slice(0, 20) },
      { id: 'c', label: 'a-very-long-run-name-indeed', color: 'green', end: 'failed' as const, points: pts((x) => x / 50).slice(0, 8) },
      { id: 'd', label: 'fasteners-n', color: 'orange', points: pts((x) => x / 35) },
    ];
    const wide = mountAt({ series, label: 'cmp', endLabels: true, endDots: true, xDomain: [0, 29] }, 900);
    const labels = [...wide.target.querySelectorAll('.end-label')].map((t) => t.textContent).sort();
    expect(labels).toEqual(['a-very-long-run-n… · failed', 'fasteners-n', 'fasteners-s · live', 'shapes-hard-s · stopped']);
    unmount(wide.c);
    const narrow = mountAt({ series, label: 'cmp', endLabels: true, endDots: true, xDomain: [0, 29] }, 340);
    expect(narrow.target.querySelector('.end-label')).toBeNull();
    unmount(narrow.c);
  });

  it('pulses the best ring and focuses it when their counts go up, never for the counts it mounted with', () => {
    const series = [{ id: 'd', label: 'd', color: 'ink', points: pts((x) => x / 30) }];
    const marker = { x: 20, y: 0.66, label: 'best', quiet: true };
    const props = $state({ series, label: 'best', markers: [{ ...marker, pulse: 3, focus: 1 }] as (typeof marker & { pulse?: number; focus?: number })[] });
    const { target, c } = mountAt(props);
    // A tab switch or a finished run mounts with a count: nothing replays.
    expect(target.querySelector('.marker .ring')).toBeTruthy();
    expect(target.querySelector('.marker .pulse')).toBeNull();
    expect(target.querySelector('.marker .focus')).toBeNull();
    props.markers = [{ ...marker, pulse: 4, focus: 2 }];
    flushSync();
    expect(target.querySelector('.marker .pulse')).toBeTruthy();
    expect(target.querySelector('.marker .focus')).toBeTruthy();
    unmount(c);
  });
});
