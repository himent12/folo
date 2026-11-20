// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { describe, expect, it, vi } from 'vitest';
import { render } from '../../../../framework/testing';
import MatchView from './MatchView';

describe('MatchView', () => {
  it('stops loading its image when the gallery takes it down (a new sort, a page change; stress s1-datasets-2)', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(() => <MatchView src="/api/bench/b/images/0" width={100} height={100} truth={[]} detections={[]} classes={['a']} />, target);
    flushSync();
    const img = target.querySelector('img')!;
    expect(img.getAttribute('src')).toBe('/api/bench/b/images/0');
    c.unmount();
    // Chrome goes on loading a detached <img> that keeps its src.
    expect(img.hasAttribute('src')).toBe(false);
    target.remove();
  });

  /** Renders a MatchView in a `w × h` frame (happy-dom has no layout); its root and an unmount. */
  function shown(w: number, h: number, view: () => JSX.Element) {
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(w);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(h);
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(view, target);
    flushSync();
    return {
      target,
      done: () => {
        c.unmount();
        target.remove();
        vi.restoreAllMocks();
      },
    };
  }
  const px = (v: string) => parseFloat(v);

  it('draws a whole-image box inside a tile the image fills, where its line is not clipped (issue #382)', () => {
    const truth = [{ class: 0, x1: 0, y1: 0, x2: 640, y2: 480 }];
    const { target, done } = shown(216, 162, () => <MatchView src="/x.jpg" width={640} height={480} truth={truth} detections={[]} classes={['a']} />);
    const box = target.querySelector<HTMLElement>('.box.missed')!;
    // Its 1 px line and halo, 2 px outside the box, fall inside the tile's 0–216 × 0–162.
    const [l, t, w, h] = [box.style.left, box.style.top, box.style.width, box.style.height].map(px) as [number, number, number, number];
    expect(l).toBeGreaterThanOrEqual(2);
    expect(t).toBeGreaterThanOrEqual(2);
    expect(l + w).toBeLessThanOrEqual(214);
    expect(t + h).toBeLessThanOrEqual(160);
    // The image moved in with it: the box stays on its object.
    expect(target.querySelector<HTMLElement>('img')!.style.inset).toBe('2px');
    done();
  });

  it('places full-size chips against the boxes as drawn, never one over another', () => {
    // Two missed boxes on adjacent chip rows at the same x, the first one on the image's top edge.
    const truth = [
      { class: 0, x1: 40, y1: 0, x2: 130, y2: 9 },
      { class: 0, x1: 40, y1: 16, x2: 190, y2: 60 },
    ];
    const { target, done } = shown(216, 162, () => <MatchView src={null} width={216} height={162} truth={truth} detections={[]} classes={['a']} full />);
    const at = [...target.querySelectorAll<HTMLElement>('.box')].map((b) => {
      const chip = b.querySelector<HTMLElement>('.chip')!;
      return { x: px(b.style.left) + px(chip.style.left), y: px(b.style.top) + px(chip.style.top) };
    });
    expect(at).toHaveLength(2);
    // Chips are 14 px high at this size: on the same x they must be a row apart.
    if (Math.abs(at[0]!.x - at[1]!.x) < 1) expect(Math.abs(at[0]!.y - at[1]!.y)).toBeGreaterThanOrEqual(14);
    done();
  });
});
