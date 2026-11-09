// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import LineChart from './LineChart.svelte';

describe('LineChart announcements', () => {
  it('speaks keyboard-scrubbed values and goes quiet on pointer movement', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [{ id: 'a', label: 'mAP', color: 'red', points: [0, 1, 2, 3].map((x) => ({ x, y: x / 10 })) }];
    const c = mount(LineChart, { target, props: { series, label: 'test chart', xTitle: (x: number) => `Epoch ${x}` } });
    // The plot renders once it has a width; happy-dom reports 0, so fake one.
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 400 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const live = target.querySelector('[aria-live]')!;
    const svg = target.querySelector('svg[role=img]');
    expect(svg).toBeTruthy();
    if (svg) {
      svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
      flushSync();
      await tick();
      await tick();
      expect(live.textContent).toContain('Epoch 3');
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 10, bubbles: true }));
      flushSync();
      expect(live.textContent).toBe('');
      // End → Escape → End announces again (Escape clears the region, so the same text is new).
      const key = (k: string) => svg.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
      key('End');
      flushSync();
      await tick();
      await tick();
      key('Escape');
      flushSync();
      expect(live.textContent).toBe('');
      key('End');
      flushSync();
      await tick();
      await tick();
      expect(live.textContent).toContain('Epoch 3');
    }
    unmount(c);
  });
});

describe('LineChart x ticks', () => {
  it('labels a short domain with decimals instead of repeating integers', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [{ id: 'a', label: 'loss', color: 'red', points: [0, 0.25, 0.5].map((x) => ({ x, y: 1 - x })) }];
    const c = mount(LineChart, { target, props: { series, label: 'short', xDomain: [0, 1] as [number, number] } });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 600 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const labels = [...target.querySelectorAll('.ticks text')].map((t) => t.textContent);
    const xs = labels.slice(-6);
    expect(new Set(xs).size).toBe(xs.length);
    unmount(c);
  });
});

describe('LineChart touch readout', () => {
  it('pins the cursor on a tap, keeps it when the finger lifts, and lets go on a tap outside', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [
      { id: 'a', label: 'long', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) },
      { id: 'b', label: 'short', color: 'blue', points: [0, 1].map((x) => ({ x, y: x / 5 })) },
    ];
    const c = mount(LineChart, { target, props: { series, label: 'touch', legendNote: 'Latest · ep 4', xTitle: (x: number) => `Epoch ${x}` } });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 400 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector('svg[role=img]')!;
    const touch = (type: string, clientX: number, clientY = 50) =>
      svg.dispatchEvent(new PointerEvent(type, { pointerType: 'touch', clientX, clientY, bubbles: true }));
    // At rest the note says which value the legend shows, on a narrow plot too.
    expect(target.querySelector('.legend .note')?.textContent).toBe('Latest · ep 4');
    touch('pointerdown', 300);
    touch('pointerup', 300);
    touch('pointerleave', 300);
    flushSync();
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    // The cursor's x reads out on the axis, not in a note.
    expect(target.querySelector('.legend .note')).toBeNull();
    expect(target.querySelector('.pill text')!.textContent).toMatch(/^Epoch \d$/);
    // The shorter series is painted last (on top).
    const lines = [...target.querySelectorAll('path.line')].map((p) => (p as SVGElement).style.stroke);
    expect(lines).toEqual(['red', 'blue']);
    document.body.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });

  it('lets go when the pinned cursor is tapped again, or on Escape', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [{ id: 'a', label: 'a', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) }];
    const c = mount(LineChart, { target, props: { series, label: 'touch' } });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 600 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector('svg[role=img]')!;
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

describe('LineChart layering and ticks', () => {
  const mountAt = (props: Record<string, unknown>, w = 640) => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(LineChart, { target, props: props as never });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    return { target, c };
  };
  const series = [
    { id: 'sky', label: 'sky', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) },
    { id: 'teal', label: 'teal', color: 'blue', pattern: 'dot' as const, points: [0, 1, 2, 3, 4, 5, 6].map((x) => ({ x, y: x / 10 })) },
  ];

  it('paints the highlighted series last, with its halo, and keeps its dot rhythm', () => {
    const { target, c } = mountAt({ series, label: 'cmp', highlight: 'teal' });
    const lines = [...target.querySelectorAll('path.line')].map((p) => (p as SVGElement).style.stroke);
    expect(lines).toEqual(['red', 'blue']);
    // Drawn once (no duplicate copy on top), emphasised, the dots spaced for the heavier stroke.
    const teal = target.querySelectorAll('path.line')[1] as SVGElement;
    expect(teal.classList.contains('emphasis')).toBe(true);
    expect(teal.style.strokeDasharray).toMatch(/^0\.014,?\s*6\.3/);
    // The halo sits directly under its line: the same dots, 1.5 px wider on each side.
    const halo = teal.previousElementSibling as SVGElement;
    expect(halo.classList.contains('halo')).toBe(true);
    expect(halo.style.strokeWidth).toBe('6.25px');
    expect(halo.style.strokeDasharray).toBe(teal.style.strokeDasharray);
    // Nothing is faded: the other run keeps its colour and weight.
    const sky = target.querySelectorAll('path.line')[0] as SVGElement;
    expect(sky.classList.contains('emphasis')).toBe(false);
    expect(target.querySelector('.faded')).toBeNull();
    unmount(c);
  });

  it('with recede, draws the other runs as context curves and the highlighted one in its hue, until it is let go', async () => {
    const props = $state({ series, label: 'cmp', highlight: 'sky' as string | null, recede: true, endLabels: true });
    const { target, c } = mountAt(props);
    const lines = () => [...target.querySelectorAll('path.line')] as SVGElement[];
    // The context run first, in chart.context (no inline hue) and keeping its dotted rhythm; the pinned run last in its hue.
    expect(lines().map((l) => [l.classList.contains('context'), l.style.stroke])).toEqual([
      [true, ''],
      [false, 'red'],
    ]);
    expect(lines()[0]!.style.strokeDasharray).not.toBe('');
    expect(lines()[1]!.style.strokeWidth).toBe('3px');
    const label = (id: string) => [...target.querySelectorAll<SVGElement>('.end-label')].find((t) => t.textContent === id)!;
    expect(label('teal').style.fill).toBe('var(--text-tertiary)');
    expect(label('sky').classList.contains('hl')).toBe(true);
    props.highlight = null;
    flushSync();
    expect(lines().map((l) => l.style.stroke)).toEqual(['red', 'blue']);
    expect(target.querySelector('path.line.context')).toBeNull();
    unmount(c);
  });

  it('paints a dotted series over a solid one even when it is the shorter of the two', () => {
    const solid = { id: 'long', label: 'long', color: 'red', points: [0, 1, 2, 3, 4, 5, 6, 7].map((x) => ({ x, y: x / 10 })) };
    const dotted = { id: 'dot', label: 'dot', color: 'blue', pattern: 'dot' as const, points: [0, 1, 2].map((x) => ({ x, y: x / 10 })) };
    const { target, c } = mountAt({ series: [dotted, solid], label: 'z' });
    const paths = [...target.querySelectorAll('path.line, path.halo')].map((p) => `${p.getAttribute('class')!.split(' ')[0]}:${(p as SVGElement).style.stroke || 'panel'}`);
    expect(paths).toEqual(['line:red', 'halo:panel', 'line:blue']);
    unmount(c);
  });

  it('pulses the best ring when its pulse count goes up, and not before', () => {
    const marker = { x: 2, y: 0.2, label: 'best', quiet: true, color: 'lime' };
    const props = $state({ series, label: 'best', markers: [marker] as (typeof marker & { pulse?: number })[] });
    const { target, c } = mountAt(props);
    expect(target.querySelector('.marker .ring')).toBeTruthy();
    expect(target.querySelector('.marker .pulse')).toBeNull();
    props.markers = [{ ...marker, pulse: 1 }];
    flushSync();
    expect(target.querySelector('.marker .pulse')).toBeTruthy();
    unmount(c);
  });

  it('does not replay a pulse it was mounted with (a tab switch, a finished run)', () => {
    const marker = { x: 2, y: 0.2, label: 'best', quiet: true, color: 'lime', pulse: 3 };
    const { target, c } = mountAt({ series, label: 'best', markers: [marker] });
    expect(target.querySelector('.marker .ring')).toBeTruthy();
    expect(target.querySelector('.marker .pulse')).toBeNull();
    unmount(c);
  });

  it('keeps the axes around an empty plot and states its one line', () => {
    const { target, c } = mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: [] }], label: 'q', xDomain: [0, 120], empty: 'Starts when fasteners-s finishes' });
    expect(target.querySelector('.axis')).toBeTruthy();
    expect(target.querySelectorAll('.grid line').length).toBeGreaterThan(1);
    expect(target.querySelector('.empty')!.textContent).toBe('Starts when fasteners-s finishes');
    unmount(c);
  });

  it('draws legend swatches in the pattern of their lines', () => {
    const { target, c } = mountAt({ series, label: 'cmp' });
    const sw = [...target.querySelectorAll('.legend .swatch line')].map((l) => (l as SVGElement).style.strokeDasharray);
    expect(sw[0]).toBe('');
    expect(sw[1]).toMatch(/^0\.01,?\s*4\.5/);
    unmount(c);
  });

  it('uses the y ticks the caller fitted', () => {
    const { target, c } = mountAt({ series, label: 'fit', yDomain: [0, 0.7], yTickValues: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] });
    const ys = [...target.querySelectorAll('.ticks text[text-anchor="end"]')].map((t) => t.textContent);
    expect(ys).toEqual(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']);
    unmount(c);
  });
});

describe('LineChart round 18', () => {
  const mountAt = (props: Record<string, unknown>, w = 640) => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(LineChart, { target, props: props as never });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    return { target, c };
  };
  const yTicks = (target: Element) => [...target.querySelectorAll('.ticks text[text-anchor="end"]')].map((t) => t.textContent);
  const pts = (f: (x: number) => number) => Array.from({ length: 30 }, (_, x) => ({ x, y: f(x) }));

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

  it('marks a phase change on x: its line, its label above the plot, its span hatched along the axis', () => {
    const { target, c } = mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x) }], label: 't', xMark: { x: 25, label: 'mosaic off' } });
    expect(target.querySelector('.mark-label')?.textContent).toBe('mosaic off');
    expect(target.querySelector('.mark-line')).not.toBeNull();
    expect(target.querySelector('rect[fill^="url("]')).not.toBeNull();
    unmount(c);
    // Outside the domain: nothing drawn.
    const out = mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x) }], label: 't', xMark: { x: 40, label: 'mosaic off' } });
    expect(out.target.querySelector('.mark-line')).toBeNull();
    unmount(out.c);
  });

  it('labels short-named lines at their ends even on a phone, and paints a front line over a dotted one', () => {
    const series = [
      { id: 'box', label: 'box', color: 'orange', front: true, points: pts((x) => 1 - x / 60) },
      { id: 'dfl', label: 'dfl', color: 'teal', pattern: 'dot' as const, points: pts((x) => 1 - x / 61) },
      { id: 'tot', label: 'Total, EMA 0.9', endLabel: 'total', color: 'white', emphasis: true, points: pts((x) => 3 - x / 30) },
      { id: 'raw', label: 'raw', color: 'white', ghost: true, points: pts((x) => 3 - x / 31) },
    ];
    const { target, c } = mountAt({ series, label: 'loss', endLabels: true }, 340);
    const labels = [...target.querySelectorAll('.end-label')].map((t) => t.textContent).sort();
    expect(labels).toEqual(['box', 'dfl', 'total']);
    const strokes = [...target.querySelectorAll('path.line')].map((p) => (p as SVGElement).style.stroke);
    expect(strokes.indexOf('orange')).toBeGreaterThan(strokes.indexOf('teal'));
    expect(strokes.at(-1)).toBe('white');
    unmount(c);
  });

  it('keeps long labels off a narrow plot', () => {
    const series = [{ id: 'a', label: 'fasteners-s', color: 'red', points: pts((x) => x / 30) }];
    const { target, c } = mountAt({ series, label: 'cmp', endLabels: true }, 340);
    expect(target.querySelector('.end-label')).toBeNull();
    unmount(c);
  });

  it('hints at the arrow keys while the plot has keyboard focus, until a value is read out', async () => {
    const props = $state({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x / 30) }], label: 't', legendNote: 'Last', hint: null as string | null });
    const { target, c } = mountAt(props);
    const svg = target.querySelector('svg[role=img]') as SVGSVGElement;
    // happy-dom has no :focus-visible; a keyboard focus is what it would match.
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

  it('shows no hint on a mouse focus or hover', () => {
    const { target, c } = mountAt({ series: [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x / 30) }], label: 't', legendNote: 'Last' });
    const svg = target.querySelector('svg[role=img]') as SVGSVGElement;
    svg.matches = (() => false) as unknown as typeof svg.matches;
    svg.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(target.querySelector('.legend .hint')).toBeNull();
    unmount(c);
  });
});
