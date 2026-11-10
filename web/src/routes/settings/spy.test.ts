import { describe, expect, it } from 'vitest';
import { sectionAt } from './spy';

const page = (scroll: number) =>
  [
    { id: 'connection', top: 120 },
    { id: 'appearance', top: 600 },
    { id: 'system', top: 2400 },
    { id: 'about', top: 4000 },
  ].map((s) => ({ ...s, top: s.top - scroll }));

describe('sectionAt', () => {
  it('is the first section until another reaches the line under the sticky chrome', () => {
    expect(sectionAt(page(0), 120)).toBe('connection');
    expect(sectionAt(page(500), 120)).toBe('appearance');
  });

  it('is the section under the chrome, not the last one, once the page bottoms out', () => {
    // A phone at the page's end: System's top has passed the line, About's (a short last panel) has not.
    expect(sectionAt(page(2400), 120)).toBe('system');
  });

  it('is undefined with no sections', () => {
    expect(sectionAt([], 120)).toBeUndefined();
  });
});
