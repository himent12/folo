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

  it('draws a whole-image box inside a tile the image fills, where its line is not clipped (issue #382)', () => {
    // A 216 × 162 tile (happy-dom has no layout), a 640 × 480 image filling it, and a box over the whole image.
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(216);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(162);
    const target = document.createElement('div');
    document.body.appendChild(target);
    const truth = [{ class: 0, x1: 0, y1: 0, x2: 640, y2: 480 }];
    const c = render(() => <MatchView src={null} width={640} height={480} truth={truth} detections={[]} classes={['a']} />, target);
    flushSync();
    const box = target.querySelector<HTMLElement>('.box.missed')!;
    // Its 1 px line and halo, 2 px outside the box, fall inside the tile's 0–216 × 0–162.
    expect([box.style.left, box.style.top, box.style.width, box.style.height]).toEqual(['2px', '2px', '212px', '158px']);
    c.unmount();
    target.remove();
    vi.restoreAllMocks();
  });
});
