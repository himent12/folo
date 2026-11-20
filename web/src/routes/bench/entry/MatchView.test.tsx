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
    expect(target.querySelector<HTMLElement>('img')!.style.inset).toBe('2.5px');
    done();
  });

  it('keeps a sliver box at the edge whole, and moves nothing for a box off the image, the large view, or a strip', () => {
    const inset = (t: HTMLElement) => t.querySelector<HTMLElement>('img')!.style.inset;
    // A box over the last two pixels of the image: drawn 1 px wide about its centre, its halo inside the tile.
    const sliver = [{ class: 0, x1: 638, y1: 100, x2: 640, y2: 200 }];
    let v = shown(216, 162, () => <MatchView src="/x.jpg" width={640} height={480} truth={sliver} detections={[]} classes={['a']} />);
    const box = v.target.querySelector<HTMLElement>('.box')!;
    expect(px(box.style.width)).toBe(1);
    expect(px(box.style.left) + 1 + 2).toBeLessThanOrEqual(216);
    v.done();
    // A wrong detection wholly right of the image (the server takes such coordinates): no room asked.
    const off = [{ class: 0, x1: 656, y1: 100, x2: 690, y2: 200, score: 0.9 }];
    v = shown(216, 162, () => <MatchView src="/x.jpg" width={640} height={480} truth={[]} detections={off} classes={['a']} />);
    expect(inset(v.target)).toBe('0px');
    v.done();
    // The large view is drawn as before.
    const whole = [{ class: 0, x1: 0, y1: 0, x2: 640, y2: 480 }];
    v = shown(640, 480, () => <MatchView src="/x.jpg" width={640} height={480} truth={whole} detections={[]} classes={['a']} full />);
    expect(inset(v.target)).toBe('0px');
    v.done();
    // A 4096 × 128 strip in a 320 px tile is 10 px high: no room that would leave it nothing.
    v = shown(320, 10, () => <MatchView src="/x.jpg" width={4096} height={128} truth={[{ class: 0, x1: 0, y1: 0, x2: 50, y2: 128 }]} detections={[]} classes={['a']} />);
    expect(inset(v.target)).toBe('0px');
    expect(v.target.querySelector('.box')).not.toBeNull();
    v.done();
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
