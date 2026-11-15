// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { describe, expect, it } from 'vitest';
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
});
