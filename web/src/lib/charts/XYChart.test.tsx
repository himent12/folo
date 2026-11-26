// @vitest-environment happy-dom
import { flush, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import XYChart from './XYChart';

const SERIES = [
  {
    id: 'a',
    label: 'model A',
    color: 'var(--chart-1)',
    points: [
      { x: 0, y: 0.2 },
      { x: 0.5, y: 0.6 },
      { x: 1, y: 0.9 },
    ],
  },
];

const roListeners: ResizeObserverCallback[] = [];
const OrigRO = globalThis.ResizeObserver;
globalThis.ResizeObserver = class {
  constructor(cb: ResizeObserverCallback) {
    roListeners.push(cb);
  }
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

describe('XYChart', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  function mount(w = 480) {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(
      () => (
        <XYChart
          label="Precision vs Recall"
          series={SERIES}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xTicks={[0, 0.5, 1]}
          yTicks={[0, 0.5, 1]}
          readout={(cur, hint) => <span class="out">{cur() !== null ? `x=${cur()}` : (hint() ?? 'idle')}</span>}
          table={{ caption: 'PR', head: ['x', 'y'], rows: [['0', '0.2']] }}
        />
      ),
      target,
    );
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flush();
    return { target, c, svg: target.querySelector<SVGSVGElement>('svg[aria-roledescription=chart]')! };
  }

  it('hides the keyboard inspect hint on touch/phone media queries', () => {
    vi.spyOn(window, 'matchMedia').mockImplementation(
      (query: string) =>
        ({
          matches: query.includes('max-width') || query.includes('pointer: coarse'),
          media: query,
          addEventListener: () => {},
          removeEventListener: () => {},
        }) as unknown as MediaQueryList,
    );
    const { target, c, svg } = mount();
    try {
      svg.matches = ((sel: string) => sel === ':focus-visible') as unknown as typeof svg.matches;
      svg.dispatchEvent(new FocusEvent('focus'));
      flush();
      expect(target.querySelector('.out')!.textContent).not.toContain('← → to inspect');
    } finally {
      c.unmount();
    }
  });

  it('caches the SVG DOMRect across pointermove bursts and invalidates when readout height changes', async () => {
    const { target, c, svg } = mount();
    try {
      await Promise.resolve();
      const readout = target.querySelector<HTMLElement>('.readout')!;
      let readoutHeight = 20;
      Object.defineProperty(readout, 'clientHeight', { get: () => readoutHeight, configurable: true });
      const rectSpy = vi.spyOn(svg, 'getBoundingClientRect');
      svg.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
      for (let px = 50; px < 300; px += 25) {
        svg.dispatchEvent(new PointerEvent('pointermove', { clientX: px, clientY: 80, bubbles: true }));
        flush();
      }
      expect(rectSpy).toHaveBeenCalledTimes(1);
      // Readout wraps onto a second line -> ResizeObserver updates readoutH and invalidates cached SVG rect.
      readoutHeight = 40;
      for (const cb of roListeners) cb([{ target: readout } as unknown as ResizeObserverEntry], {} as ResizeObserver);
      flush();
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 100, clientY: 80, bubbles: true }));
      flush();
      expect(rectSpy).toHaveBeenCalledTimes(2);
      svg.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }));
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 120, clientY: 80, bubbles: true }));
      flush();
      expect(rectSpy).toHaveBeenCalledTimes(3);

      // External banner inserted above the chart shifts SVG top down without scroll/resize: lower-plot moves re-measure.
      let top = 80;
      rectSpy.mockImplementation(() => ({ left: 0, top, right: 480, bottom: top + 240, width: 480, height: 240, x: 0, y: top, toJSON: () => ({}) }) as DOMRect);
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 200, clientY: 260, bubbles: true }));
      flush();
      expect(target.querySelector('.out')!.textContent).toContain('x=');
      expect(rectSpy).toHaveBeenCalledTimes(4);
      expect(OrigRO).toBeDefined();
    } finally {
      c.unmount();
    }
  });

  it('ignores modifier-held arrows, consumes Escape only while active, and clears speech on blur/pointermove', () => {
    const { target, c, svg } = mount();
    try {
      const live = target.querySelector('[aria-live]')!;
      const altLeft = new KeyboardEvent('keydown', { key: 'ArrowLeft', altKey: true, bubbles: true, cancelable: true });
      svg.dispatchEvent(altLeft);
      flush();
      expect(altLeft.defaultPrevented).toBe(false);
      expect(target.querySelector('.out')!.textContent).toBe('idle');

      const idleEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
      svg.dispatchEvent(idleEsc);
      flush();
      expect(idleEsc.defaultPrevented).toBe(false);

      svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
      flush();
      expect(target.querySelector('.out')!.textContent).toBe('x=0');
      expect(live.textContent).not.toBe('');

      const activeEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
      svg.dispatchEvent(activeEsc);
      flush();
      expect(activeEsc.defaultPrevented).toBe(true);
      expect(target.querySelector('.out')!.textContent).toBe('idle');
      expect(live.textContent).toBe('');

      svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
      flush();
      expect(live.textContent).not.toBe('');
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 120, clientY: 80, bubbles: true }));
      flush();
      expect(live.textContent).toBe('');

      svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
      flush();
      expect(live.textContent).not.toBe('');
      svg.dispatchEvent(new FocusEvent('blur'));
      flush();
      expect(live.textContent).toBe('');
      expect(target.querySelector('.out')!.textContent).toBe('idle');

      // Touch-pinned cursor clears on Escape with defaultPrevented, but defers to open dialog/popover.
      svg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX: 200, clientY: 80, bubbles: true }));
      svg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX: 200, clientY: 80, bubbles: true }));
      flush();
      expect(target.querySelector('.cursor.pinned')).toBeTruthy();
      const pop = document.createElement('div');
      pop.className = 'info-pop open';
      document.body.appendChild(pop);
      const blockedEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
      document.dispatchEvent(blockedEsc);
      flush();
      expect(blockedEsc.defaultPrevented).toBe(false);
      expect(target.querySelector('.cursor.pinned')).toBeTruthy();
      pop.remove();

      const esc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
      document.dispatchEvent(esc);
      flush();
      expect(esc.defaultPrevented).toBe(true);
      expect(target.querySelector('.cursor')).toBeNull();
    } finally {
      c.unmount();
    }
  });

  it('does not arm a phantom touch pin when stops is empty and clears pinned/speech on external cursor reset', () => {
    const cursor = signal<number | null>(null);
    const target = document.createElement('div');
    document.body.appendChild(target);
    const emptyChart = render(
      () => (
        <XYChart
          label="Empty stops"
          series={SERIES}
          stops={[]}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xTicks={[0, 1]}
          yTicks={[0, 1]}
          readout={() => <span>idle</span>}
          table={{ caption: 'Empty', head: ['x', 'y'], rows: [] }}
        />
      ),
      target,
    );
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 480 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flush();
    const emptySvg = target.querySelector<SVGSVGElement>('svg[aria-roledescription=chart]')!;
    emptySvg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX: 200, clientY: 80, bubbles: true }));
    emptySvg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX: 200, clientY: 80, bubbles: true }));
    flush();
    const emptyEsc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(emptyEsc);
    flush();
    expect(emptyEsc.defaultPrevented).toBe(false);
    emptyChart.unmount();

    const syncedTarget = document.createElement('div');
    document.body.appendChild(syncedTarget);
    const synced = render(
      () => (
        <XYChart
          label="Synced"
          series={SERIES}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xTicks={[0, 0.5, 1]}
          yTicks={[0, 0.5, 1]}
          bind:cursor={cursor}
          readout={(cur) => <span class="out">{cur() ?? 'idle'}</span>}
          table={{ caption: 'PR', head: ['x', 'y'], rows: [['0', '0.2']] }}
        />
      ),
      syncedTarget,
    );
    Object.defineProperty(syncedTarget.querySelector('.plot')!, 'clientWidth', { value: 480 });
    syncedTarget.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flush();
    const syncedSvg = syncedTarget.querySelector<SVGSVGElement>('svg[aria-roledescription=chart]')!;
    const live = syncedTarget.querySelector('[aria-live]')!;
    syncedSvg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX: 200, clientY: 80, bubbles: true }));
    syncedSvg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX: 200, clientY: 80, bubbles: true }));
    syncedSvg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
    flush();
    expect(syncedTarget.querySelector('.cursor.pinned')).toBeTruthy();
    expect(live.textContent).not.toBe('');
    cursor.set(null);
    flush();
    expect(syncedTarget.querySelector('.cursor')).toBeNull();
    expect(live.textContent).toBe('');
    synced.unmount();
  });
});
