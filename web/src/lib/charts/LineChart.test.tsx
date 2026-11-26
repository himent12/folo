// @vitest-environment happy-dom
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { deep, field, flush as flushSync, signal, tick } from 'fw';
import { describe, expect, it } from 'vitest';
import { gChord } from '$lib/shell/shortcuts';
import { render } from '../../../framework/testing';
import LineChart, { type LineChartProps } from './LineChart';

/** Port of `mount(LineChart, { target, props })`; `unmount(c)` is `c.unmount()`. Props are read live (a `deep` object). */
const mount = (_: typeof LineChart, o: { target: HTMLElement; props: LineChartProps }) => {
  const props = o.props;
  // Every prop through a getter, so a test that reassigns one (a `deep` props object) updates the chart; `picked` is
  // bound, as Svelte's `$state` props object binds it.
  const live = new Proxy({} as LineChartProps, {
    get: (_, k) => (props as unknown as Record<PropertyKey, unknown>)[k],
    has: (_, k) => k in props,
    ownKeys: () => Reflect.ownKeys(props),
    getOwnPropertyDescriptor: (_, k) => (k in props ? { configurable: true, enumerable: true, get: () => (props as unknown as Record<PropertyKey, unknown>)[k] } : undefined),
  });
  return render(() => <LineChart {...live} bind:picked={field(props, 'picked')} />, o.target);
};
const unmount = (c: { unmount(): void }) => c.unmount();

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

  it('starts from the first point on ArrowRight and last point on ArrowLeft, ignores modifier-held arrows, and consumes Escape only while active', async () => {
    const series = [{ id: 'a', label: 'mAP', color: 'red', points: [0, 1, 2, 3].map((x) => ({ x, y: x / 10 })) }];
    const { target, c, svg } = mountAt({ series, label: 'nav', xTitle: (x: number) => `Epoch ${x}`, legendNote: 'Latest' }, 400);
    const live = target.querySelector('[aria-live]')!;

    // Modifier-held ArrowLeft (e.g. Alt+ArrowLeft browser back) is ignored.
    const altBack = new KeyboardEvent('keydown', { key: 'ArrowLeft', altKey: true, bubbles: true, cancelable: true });
    svg.dispatchEvent(altBack);
    flushSync();
    expect(altBack.defaultPrevented).toBe(false);
    expect(target.querySelector('.cursor')).toBeNull();

    // Idle Escape is not consumed so page-level Escape handlers can still run.
    const idleEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    svg.dispatchEvent(idleEsc);
    flushSync();
    expect(idleEsc.defaultPrevented).toBe(false);

    // ArrowRight from idle lands on the first point (Epoch 0).
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
    flushSync();
    await tick();
    await tick();
    expect(live.textContent).toContain('Epoch 0');

    // Escape while active consumes the event and clears cursor + speech.
    const activeEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    svg.dispatchEvent(activeEsc);
    flushSync();
    expect(activeEsc.defaultPrevented).toBe(true);
    expect(target.querySelector('.cursor')).toBeNull();
    expect(live.textContent).toBe('');

    // ArrowLeft from idle lands on the last point (Epoch 3), not skipping it.
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }));
    flushSync();
    await tick();
    await tick();
    expect(live.textContent).toContain('Epoch 3');

    // Rapid ArrowLeft followed immediately by Escape cancels the pending async announcement.
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }));
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    flushSync();
    await tick();
    await tick();
    expect(live.textContent).toBe('');
    unmount(c);
  });

  it('clears spoken text when an external bound cursor resets to null', async () => {
    const cursor = signal<number | null>(null);
    const series = [{ id: 'a', label: 'mAP', color: 'red', points: [0, 1, 2].map((x) => ({ x, y: x / 10 })) }];
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(() => <LineChart series={series} label="synced" xTitle={(x) => `Epoch ${x}`} bind:cursor={cursor} />, target);
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 400 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector<SVGSVGElement>('svg[aria-roledescription=chart]')!;
    const live = target.querySelector('[aria-live]')!;
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
    flushSync();
    await tick();
    await tick();
    expect(live.textContent).toContain('Epoch 2');
    cursor.set(null);
    flushSync();
    expect(live.textContent).toBe('');
    c.unmount();
    target.remove();
  });
});

describe('LineChart touch readout', () => {
  it('pins the cursor on a tap, keeps it when the finger lifts or focus blurs, and lets go on a tap outside', () => {
    const series = [{ id: 'a', label: 'long', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) }];
    const { target, c, svg } = mountAt({ series, label: 'touch', xTitle: (x: number) => `Epoch ${x}` }, 400);
    const touch = (type: string, clientX: number, clientY = 50) => svg.dispatchEvent(new PointerEvent(type, { pointerType: 'touch', clientX, clientY, bubbles: true }));
    touch('pointerdown', 300);
    touch('pointerup', 300);
    touch('pointerleave', 300);
    svg.dispatchEvent(new FocusEvent('blur'));
    flushSync();
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    // The cursor's x reads out on the axis.
    expect(target.querySelector('.pill text')!.textContent).toMatch(/^Epoch \d$/);
    document.body.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });

  it('lets go when the pinned cursor is tapped again, or on Escape (deferring to dialogs, popovers, tooltips, inputs, and gChord)', () => {
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

    // Open dialog, info popover, tooltip, active G-chord, and focused input all take Escape priority.
    const blockers = [
      Object.assign(document.createElement('dialog'), { open: true }),
      Object.assign(document.createElement('div'), { className: 'info-pop open' }),
      (() => {
        const tip = document.createElement('div');
        tip.id = 'folo-tooltip';
        tip.setAttribute('data-show', '');
        return tip;
      })(),
    ];
    for (const el of blockers) {
      document.body.appendChild(el);
      const blockedEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
      document.dispatchEvent(blockedEsc);
      flushSync();
      expect(blockedEsc.defaultPrevented).toBe(false);
      expect(target.querySelector('.cursor.pinned')).toBeTruthy();
      el.remove();
    }
    const input = document.createElement('input');
    document.body.appendChild(input);
    const inputEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    input.dispatchEvent(inputEsc);
    flushSync();
    expect(inputEsc.defaultPrevented).toBe(false);
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    input.remove();

    gChord.set(true);
    try {
      const chordEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
      document.dispatchEvent(chordEsc);
      flushSync();
      expect(chordEsc.defaultPrevented).toBe(false);
      expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    } finally {
      gChord.set(false);
    }

    const esc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(esc);
    flushSync();
    expect(esc.defaultPrevented).toBe(true);
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });

  it('does not arm a phantom touch pin on an empty chart or in a live chart future band, and drops the pin if data clears', () => {
    const { c: emptyC, svg: emptySvg } = mountAt({ series: [], label: 'empty' }, 400);
    emptySvg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX: 200, clientY: 50, bubbles: true }));
    emptySvg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX: 200, clientY: 50, bubbles: true }));
    flushSync();
    const emptyEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(emptyEsc);
    flushSync();
    expect(emptyEsc.defaultPrevented).toBe(false);
    unmount(emptyC);

    // Live run with data on epochs 0..2 inside a planned domain 0..20: tapping the future region must not arm a pin.
    const props = deep({
      series: [{ id: 'a', label: 'mAP', color: 'red', points: [0, 1, 2].map((x) => ({ x, y: x / 10 })) }],
      label: 'live',
      xDomain: [0, 20] as [number, number],
      picked: null as string | null,
    });
    const { target, c, svg } = mountAt(props, 400);
    svg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX: 350, clientY: 50, bubbles: true }));
    svg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX: 350, clientY: 50, bubbles: true }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    const futureEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(futureEsc);
    flushSync();
    expect(futureEsc.defaultPrevented).toBe(false);

    // Pinning on real data then clearing the series drops the pin so Escape is no longer claimed.
    svg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX: 60, clientY: 50, bubbles: true }));
    svg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX: 60, clientY: 50, bubbles: true }));
    flushSync();
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    props.series = [];
    flushSync();
    const goneEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(goneEsc);
    flushSync();
    expect(goneEsc.defaultPrevented).toBe(false);
    unmount(c);
  });
});

describe('LineChart pick', () => {
  it('picks out the line nearest the pointer where it is drawn, and a click acts on it', () => {
    // Round 26: the PR chart's grey class curves could be told apart only from the table.
    const flat = (id: string, y: number) => ({ id, label: id, color: 'grey', context: true, points: [0, 5, 10].map((x) => ({ x, y })) });
    const picks: string[] = [];
    const props = deep({
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

  it("says where a failed line stopped in the event band, at the domain's end, beside the mosaic label or under it", () => {
    const series = [{ id: 'a', label: 'a', color: 'red', points: pts((x) => x), end: 'failed' }];
    const labels = (t: Element) => [...t.querySelectorAll('text.mark-label')].map((l) => ({ text: l.textContent, y: Number(l.getAttribute('y')), anchor: l.getAttribute('text-anchor') }));
    // Alone: a hairline at the last point and its label in the band, before the line (no room after it).
    const alone = mountAt({ series, label: 't', xDomain: [0, 29], xEvent: { x: 29, label: 'failed during epoch 30' } });
    expect(alone.target.querySelectorAll('.mark-line').length).toBe(1);
    const [e] = labels(alone.target);
    expect(e).toMatchObject({ text: 'failed during epoch 30', anchor: 'end' });
    unmount(alone.c);
    // Far from the mosaic label: both in the band, on one baseline.
    const apart = mountAt({ series, label: 't', xDomain: [0, 29], xMark: { x: 3, label: 'mosaic off' }, xEvent: { x: 29, label: 'failed during epoch 30' } });
    const [m1, e1] = labels(apart.target);
    expect(m1!.text).toBe('mosaic off');
    expect(e1!.y).toBe(m1!.y);
    unmount(apart.c);
    // Where the two would touch, the event's label goes inside the plot's top, as the ETA's does.
    const near = mountAt({ series, label: 't', xDomain: [0, 29], xMark: { x: 26, label: 'mosaic off' }, xEvent: { x: 29, label: 'failed during epoch 30' } });
    const [m2, e2] = labels(near.target);
    expect(e2!.y).toBeGreaterThan(m2!.y);
    unmount(near.c);
    // No event outside the domain.
    const out = mountAt({ series, label: 't', xDomain: [0, 29], xEvent: { x: 40, label: 'failed during epoch 41' } });
    expect(labels(out.target)).toEqual([]);
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

  it("never hides a failed run's cross: a best on its last epoch is the ink ring around the danger cross", () => {
    const points = Array.from({ length: 23 }, (_, i) => ({ x: i + 1, y: 0.1 + i / 100 }));
    const series = [{ id: 'm', label: 'mAP50-95', color: 'ink', end: 'failed' as const, points }];
    const markers = [{ x: 23, y: 0.1 + 22 / 100, label: 'best', quiet: true }];
    const { target, c } = mountAt({ series, label: 'failed', endDots: true, markers, xDomain: [0, 100] });
    const cross = target.querySelector('path.end-x') as SVGElement | null;
    expect(cross).toBeTruthy();
    expect(cross!.style.stroke).toBe('var(--danger-text)');
    expect(target.querySelectorAll('.marker .ring')).toHaveLength(1);
    expect(target.querySelector('circle.end-live')).toBeNull();
    // The cross comes after the ring in the markup, so it is drawn over it.
    const marks = [...target.querySelectorAll('.marker, path.end-x')];
    expect(marks.map((m) => m.tagName.toLowerCase())).toEqual(['g', 'path']);
    unmount(c);
  });

  it('pulses the best ring and focuses it when their counts go up, never for the counts it mounted with', () => {
    const series = [{ id: 'd', label: 'd', color: 'ink', points: pts((x) => x / 30) }];
    const marker = { x: 20, y: 0.66, label: 'best', quiet: true };
    const props = deep({ series, label: 'best', markers: [{ ...marker, pulse: 3, focus: 1 }] as (typeof marker & { pulse?: number; focus?: number })[] });
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

  it('positions the live heartbeat ring via a translated wrapper group without transform-box fill-box bbox queries', () => {
    const series = [{ id: 'live', label: 'mAP50-95', color: 'ink', end: 'live' as const, points: pts((x) => x / 30) }];
    const { target, c } = mountAt({ series, label: 'live', endDots: true });
    const wrap = target.querySelector('g.end-beat-wrap');
    const beat = wrap?.querySelector('circle.end-beat');
    const dot = target.querySelector('circle.end-dot.end-live');
    expect(wrap?.getAttribute('transform')).toMatch(/^translate\([^,]+,[^)]+\)$/);
    expect(beat).toBeTruthy();
    expect(beat?.getAttribute('cx')).toBeNull();
    expect(beat?.getAttribute('cy')).toBeNull();
    expect(dot).toBeTruthy();
    expect(wrap?.getAttribute('transform')).toBe(`translate(${dot?.getAttribute('cx')},${dot?.getAttribute('cy')})`);

    const cssText = readFileSync(resolve(import.meta.dirname, 'LineEnds.tsx'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(cssText).not.toContain('transform-box: fill-box');
    expect(cssText).toContain('transform-origin: 0 0;');
    expect(cssText).toContain('will-change: transform, opacity;');
    unmount(c);
  });

  it('self-heals cached SVG rect on out-of-bounds pointermove after an external layout shift and guards empty yTickValues', () => {
    const series = [{ id: 'a', label: 'Loss', color: 'ink', points: [{ x: 1, y: 1 }, { x: 2, y: 0.5 }, { x: 3, y: 0.2 }] }];
    const { target, c, svg } = mountAt({ series, label: 'loss', yLabel: 'Loss', yTickValues: [], legendNote: 'Last' });
    // Empty yTickValues must not produce Infinity/NaN y attribute on yLabel.
    const yLab = target.querySelector('text.x-label');
    expect(yLab?.getAttribute('y')).toBe('10');

    let left = 0;
    let top = 0;
    svg.getBoundingClientRect = () =>
      ({ left, top, width: 640, height: 180, right: left + 640, bottom: top + 180, x: left, y: top, toJSON: () => ({}) }) as DOMRect;
    svg.dispatchEvent(new PointerEvent('pointerenter', { clientX: 60, clientY: 90, pointerType: 'mouse', bubbles: true }));
    svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 60, clientY: 90, pointerType: 'mouse', bubbles: true }));
    flushSync();
    expect(target.querySelector('.legend .note')?.textContent).toBe('1');

    // External banner shifts SVG down by 300px and right by 100px within the 200ms cache TTL: pointer at new position self-heals.
    left = 100;
    top = 300;
    svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 700, clientY: 390, pointerType: 'mouse', bubbles: true }));
    flushSync();
    expect(target.querySelector('.legend .note')?.textContent).toBe('3');
    unmount(c);
  });
});
