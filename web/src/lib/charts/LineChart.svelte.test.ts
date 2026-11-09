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
    const svg = target.querySelector('svg[aria-roledescription=chart]');
    expect(svg).toBeTruthy();
    // Focusable and interactive, so a group with a chart role description, not an image (whose content is opaque).
    expect(svg?.getAttribute('role')).toBe('group');
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
    const svg = target.querySelector('svg[aria-roledescription=chart]')!;
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
    const svg = target.querySelector('svg[aria-roledescription=chart]')!;
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
    // The two lines apart, so each label has a clear spot beside its end (another run's line is an obstacle).
    const apart = [series[0]!, { ...series[1]!, points: series[1]!.points.map((p) => ({ x: p.x, y: 0.05 + p.x / 40 })) }];
    const props = $state({ series: apart, label: 'cmp', highlight: 'sky' as string | null, recede: true, endLabels: true });
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
    const marker = { x: 2, y: 0.2, label: 'best', quiet: true };
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
    const marker = { x: 2, y: 0.2, label: 'best', quiet: true, pulse: 3 };
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

  it('labels no line end on a phone-width plot, short names included, and paints a front line over a dotted one', () => {
    const series = [
      { id: 'box', label: 'box', color: 'orange', front: true, points: pts((x) => 1 - x / 60) },
      { id: 'dfl', label: 'dfl', color: 'teal', pattern: 'dot' as const, points: pts((x) => 1 - x / 61) },
      { id: 'tot', label: 'Total, EMA 0.9', endLabel: 'total', color: 'white', emphasis: true, points: pts((x) => 3 - x / 30) },
      { id: 'raw', label: 'raw', color: 'white', ghost: true, points: pts((x) => 3 - x / 31) },
    ];
    const { target, c } = mountAt({ series, label: 'loss', endLabels: true }, 340);
    expect(target.querySelector('.end-label')).toBeNull();
    const strokes = [...target.querySelectorAll('path.line')].map((p) => (p as SVGElement).style.stroke);
    expect(strokes.indexOf('orange')).toBeGreaterThan(strokes.indexOf('teal'));
    expect(strokes.at(-1)).toBe('white');
    unmount(c);
    const wide = mountAt({ series, label: 'loss', endLabels: true }, 600);
    const labels = [...wide.target.querySelectorAll('.end-label')].map((t) => t.textContent).sort();
    expect(labels).toEqual(['box', 'dfl', 'total']);
    unmount(wide.c);
  });

  it('draws a direct label in its series text ink, the line in its line ink (§2.5)', () => {
    const series = [
      { id: 'box', label: 'box', color: 'var(--chart-2)', points: pts((x) => 1 - x / 60) },
      { id: 'tot', label: 'total', color: 'var(--text-primary)', emphasis: true, points: pts((x) => 3 - x / 30) },
    ];
    const { target, c } = mountAt({ series, label: 'loss', endLabels: true }, 640);
    const fill = (t: string) => ([...target.querySelectorAll<SVGElement>('.end-label')].find((e) => e.textContent === t)!).style.fill;
    expect(fill('box')).toBe('var(--chart-text-2)');
    expect(fill('total')).toBe('var(--text-primary)');
    const strokes = [...target.querySelectorAll<SVGElement>('path.line')].map((p) => p.style.stroke);
    expect(strokes).toContain('var(--chart-2)');
    unmount(c);
  });

  it('ends each line in its status mark and names the status after the label, which it never cuts', () => {
    const series = [
      { id: 'a', label: 'fasteners-s', color: 'red', end: 'live' as const, points: pts((x) => x / 30).slice(0, 12) },
      { id: 'b', label: 'shapes-hard-s', color: 'blue', end: 'stopped' as const, points: pts((x) => x / 40).slice(0, 20) },
      { id: 'c', label: 'a-very-long-run-name-indeed', color: 'green', end: 'failed' as const, points: pts((x) => x / 50).slice(0, 8) },
      { id: 'd', label: 'fasteners-n', color: 'orange', points: pts((x) => x / 35) },
    ];
    const { target, c } = mountAt({ series, label: 'cmp', endLabels: true, endDots: true, xDomain: [0, 29] }, 900);
    expect(target.querySelectorAll('.end-live')).toHaveLength(1);
    expect(target.querySelectorAll('rect.end-dot')).toHaveLength(1);
    expect(target.querySelectorAll('path.end-x')).toHaveLength(1);
    expect(target.querySelectorAll('circle.end-dot')).toHaveLength(2);
    const labels = [...target.querySelectorAll('.end-label')].map((t) => t.textContent).sort();
    expect(labels).toEqual(['a-very-long-run-n… · failed', 'fasteners-n', 'fasteners-s · live', 'shapes-hard-s · stopped']);
    unmount(c);
  });

  it("keys a label the column holds for a line that ended inside the plot: the line's swatch, then its name", () => {
    // A failed dotted run ending inside a bundle of flat lines: no clear spot beside its end, so it goes to the column.
    const flat = (k: number) => ({ id: `l${k}`, label: `l${k}`, color: 'var(--chart-1)', endLabel: false as const, points: pts(() => 0.3 + k * 0.03) });
    const series = [
      ...Array.from({ length: 12 }, (_, k) => flat(k)),
      { id: 'f', label: 'shapes-hard-m', color: 'var(--chart-4)', pattern: 'dot' as const, end: 'failed' as const, points: pts((x) => 0.45 + x / 2000).slice(0, 15) },
    ];
    const { target, c } = mountAt({ series, label: 'cmp', endLabels: true, endDots: true, xDomain: [0, 29], yDomain: [0, 1] }, 900);
    const label = [...target.querySelectorAll<SVGElement>('.end-label')].find((t) => t.textContent === 'shapes-hard-m · failed')!;
    const key = target.querySelector<SVGElement>('line.end-key')!;
    expect(key).toBeTruthy();
    expect(key.style.stroke).toBe('var(--chart-text-4)');
    expect(key.style.strokeDasharray).toBeTruthy();
    expect(Number(label.getAttribute('x'))).toBe(Number(key.getAttribute('x1')) - 1 + 18);
    expect(Number(key.getAttribute('y1'))).toBe(Number(label.getAttribute('y')));
    expect(target.querySelector('line.leader')).toBeNull();
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
    const svg = target.querySelector('svg[aria-roledescription=chart]') as SVGSVGElement;
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
    const svg = target.querySelector('svg[aria-roledescription=chart]') as SVGSVGElement;
    svg.matches = (() => false) as unknown as typeof svg.matches;
    svg.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(target.querySelector('.legend .hint')).toBeNull();
    unmount(c);
  });
});

describe('LineChart end marks (DESIGN_DIRECTION §2.4: one thing per mark)', () => {
  const mountAt = (props: Record<string, unknown>, w = 640) => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(LineChart, { target, props: props as never });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    return { target, c };
  };
  const pts = (n: number) => Array.from({ length: n }, (_, i) => ({ x: i + 1, y: 0.1 + i / 100 }));
  const rings = (target: Element) => target.querySelectorAll('.marker .ring').length;
  const live = (target: Element) => target.querySelector('circle.end-live') as SVGElement | null;

  it('draws the best and the live end as two shapes: a hollow ink ring and a filled lime dot with the heartbeat', () => {
    // Round 25: best at epoch 36, live at 37, two identical lime ringed dots side by side.
    const series = [{ id: 'm', label: 'mAP50-95', color: 'ink', end: 'live' as const, points: pts(37) }];
    const markers = [{ x: 36, y: 0.1 + 35 / 100, label: 'best', quiet: true }];
    const { target, c } = mountAt({ series, label: 'live', endDots: true, markers, xDomain: [0, 37] });
    expect(target.querySelectorAll('.marker')).toHaveLength(1);
    expect(target.querySelector('.marker[data-merged]')).toBeNull();
    expect(rings(target)).toBe(1);
    expect(live(target)!.style.fill).toBe('var(--accent-line)');
    expect(target.querySelector('circle.end-beat')).toBeTruthy();
    // The ring takes no colour of its own: its ink is the stylesheet's text.primary on every run.
    expect((target.querySelector('.marker .ring') as SVGElement).style.stroke).toBe('');
    unmount(c);
  });

  it('draws the ring around the live dot when the best is the latest epoch', () => {
    const series = [{ id: 'm', label: 'mAP50-95', color: 'ink', end: 'live' as const, points: pts(37) }];
    const markers = [{ x: 37, y: 0.1 + 36 / 100, label: 'best', quiet: true }];
    const { target, c } = mountAt({ series, label: 'live', endDots: true, markers, xDomain: [0, 100] });
    expect(target.querySelector('.marker[data-merged="m"]')).toBeTruthy();
    expect(rings(target)).toBe(1);
    expect(live(target)).toBeTruthy();
    // The dot is painted over the ring's centre, never under it.
    const marks = [...target.querySelectorAll('.marker, circle.end-live')];
    expect(marks.map((m) => m.tagName.toLowerCase())).toEqual(['g', 'circle']);
    unmount(c);
  });

  it("never hides a failed run's cross: a best on its last epoch is the ink ring around the danger cross", () => {
    const series = [{ id: 'm', label: 'mAP50-95', color: 'ink', end: 'failed' as const, points: pts(23) }];
    const markers = [{ x: 23, y: 0.1 + 22 / 100, label: 'best', quiet: true }];
    const { target, c } = mountAt({ series, label: 'failed', endDots: true, markers, xDomain: [0, 100] });
    const cross = target.querySelector('path.end-x') as SVGElement | null;
    expect(cross).toBeTruthy();
    expect(cross!.style.stroke).toBe('var(--danger-text)');
    expect(rings(target)).toBe(1);
    expect(live(target)).toBeNull();
    const marks = [...target.querySelectorAll('.marker, path.end-x')];
    expect(marks.map((m) => m.tagName.toLowerCase())).toEqual(['g', 'path']);
    unmount(c);
  });

  it('keeps a stopped run square and a completed run dot in the line ink, with the ring around them', () => {
    const markers = [{ x: 30, y: 0.1 + 29 / 100, label: 'best', quiet: true }];
    const stopped = mountAt({ series: [{ id: 's', label: 's', color: 'ink', end: 'stopped' as const, points: pts(30) }], label: 's', endDots: true, markers, xDomain: [0, 100] });
    expect((stopped.target.querySelector('rect.end-dot') as SVGElement).style.fill).toBe('ink');
    expect(rings(stopped.target)).toBe(1);
    unmount(stopped.c);
    const done = mountAt({ series: [{ id: 'd', label: 'd', color: 'ink', points: pts(30) }], label: 'd', endDots: true, markers, xDomain: [0, 100] });
    expect((done.target.querySelector('circle.end-dot') as SVGElement).style.fill).toBe('ink');
    expect(live(done.target)).toBeNull();
    unmount(done.c);
  });

  it('focuses the best ring once when its focus count goes up, and not for the count it mounted with', () => {
    const series = [{ id: 'd', label: 'd', color: 'ink', points: pts(30) }];
    const props = $state({ series, label: 'd', endDots: true, xDomain: [0, 100], markers: [{ x: 20, y: 0.29, label: 'best', quiet: true, focus: 1 }] });
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(LineChart, { target, props: props as never });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 640 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    expect(target.querySelector('.marker .focus')).toBeNull();
    props.markers = [{ x: 20, y: 0.29, label: 'best', quiet: true, focus: 2 }];
    flushSync();
    expect(target.querySelector('.marker .focus')).toBeTruthy();
    unmount(c);
  });

  it("draws a failed run's cross in its own hue in a comparison: the glyph says it failed, the hue whose run it is", () => {
    // Round 27: Compare's failed teal run ended in a danger-red cross, which on a plot of hues read as another run.
    const series = [
      { id: 'a', label: 'a', color: 'var(--chart-4)', pattern: 'dot' as const, end: 'failed' as const, points: pts(10) },
      { id: 'b', label: 'b', color: 'var(--chart-1)', end: 'stopped' as const, points: pts(20) },
      { id: 'c', label: 'c', color: 'var(--chart-2)', points: pts(30) },
    ];
    const { target, c } = mountAt({ series, label: 'cmp', endDots: true, xDomain: [0, 30] });
    // A dotted line takes its hue's text ink (./ink), and so does its cross; the stopped square its line's hue.
    expect((target.querySelector('path.end-x') as SVGElement).style.stroke).toBe('var(--chart-text-4)');
    expect((target.querySelector('rect.end-dot') as SVGElement).style.fill).toBe('var(--chart-1)');
    unmount(c);
  });

  it('paints failed and stopped runs over the runs that ran on, so none is hidden under the others', () => {
    const series = [
      { id: 'f', label: 'f', color: 'var(--chart-4)', end: 'failed' as const, points: pts(10) },
      { id: 'd', label: 'd', color: 'var(--chart-2)', pattern: 'dash' as const, points: pts(30) },
      { id: 'n', label: 'n', color: 'var(--chart-1)', points: pts(30) },
    ];
    const { target, c } = mountAt({ series, label: 'cmp', endDots: true, xDomain: [0, 30] });
    const order = [...target.querySelectorAll<SVGElement>('path.line')].map((p) => p.style.stroke);
    expect(order).toEqual(['var(--chart-1)', 'var(--chart-text-2)', 'var(--chart-4)']);
    unmount(c);
  });
});

describe('LineChart pick', () => {
  it('picks out the line nearest the pointer where it is drawn, and a click acts on it', () => {
    // Round 26: the PR chart's grey class curves could be told apart only from the table.
    const target = document.createElement('div');
    document.body.appendChild(target);
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
    const c = mount(LineChart, { target, props });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 400 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector('svg[aria-roledescription=chart]')!;
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
