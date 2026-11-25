// @vitest-environment happy-dom
import { flush } from 'fw';
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
      expect(OrigRO).toBeDefined();
    } finally {
      c.unmount();
    }
  });
});
