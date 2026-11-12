import { describe, expect, it } from 'vitest';
import { highlight, rank, type Rankable } from './commands';

const items: (Rankable & { id: string })[] = [
  { id: 'nav-runs', group: 'Go to', label: 'Runs' },
  { id: 'theme', group: 'Actions', label: 'Switch theme (now dark)', match: ['Toggle theme'] },
  { id: 'act-train-shapes', group: 'Actions', label: 'New run on shapes', match: ['shapes', 'Train shapes'], scoped: true },
  { id: 'act-try-best', group: 'Actions', label: 'Open shapes-s-best in Playground', match: ['shapes-s-best', 'Playground shapes-s-best'], scoped: true },
  { id: 'ds-shapes', group: 'Datasets', label: 'shapes', scoped: true },
  { id: 'model-best', group: 'Models', label: 'shapes-s-best', scoped: true },
];
const ids = (q: string) => rank(items, q).map((i) => i.id);

describe('rank', () => {
  it('leaves actions about one entity (and the entities) out of the empty palette', () => {
    expect(ids('')).toEqual(['nav-runs', 'theme']);
  });

  it('ranks an action on an entity with the entity, just below it', () => {
    const r = ids('shapes');
    expect(r.indexOf('ds-shapes')).toBe(0);
    expect(r).toContain('act-train-shapes');
    expect(r.indexOf('model-best')).toBeLessThan(r.indexOf('act-try-best'));
  });

  it('answers to its other phrasings', () => {
    expect(ids('toggle theme')[0]).toBe('theme');
    expect(ids('playground shapes-s')[0]).toBe('act-try-best');
    expect(ids('new run on sha')[0]).toBe('act-train-shapes');
  });

  it('drops what does not match', () => {
    expect(ids('zzz')).toEqual([]);
  });

  it('matches words, not characters scattered in order (shell F14)', () => {
    const more: (Rankable & { id: string })[] = [
      ...items,
      { id: 'act-try-five', group: 'Actions', label: 'Open shapes-five-n-best in Playground', match: ['shapes-five-n-best'], scoped: true },
      { id: 'act-train-five', group: 'Actions', label: 'New run on shapes-five', match: ['shapes-five'], scoped: true },
      { id: 'ds-five', group: 'Datasets', label: 'shapes-five', scoped: true },
      { id: 'ds-small', group: 'Datasets', label: 'shapes-small', scoped: true },
      { id: 'set-cloud', group: 'Settings', label: 'Cloud', match: ['Cloud compute'], scoped: true },
      { id: 'nav-new', group: 'Actions', label: 'New training run' },
    ];
    const r = (q: string) => rank(more, q).map((i) => i.id);
    // "cloud" is no subsequence soup of "Open … in Playground": the Cloud settings alone.
    expect(r('cloud')).toEqual(['set-cloud']);
    // A word and the start of the next: the dataset, above actions that only hold its letters.
    expect(r('shapes s')[0]).toBe('ds-small');
    // The dataset named by the query ranks above the actions about it.
    expect(r('five').indexOf('ds-five')).toBeLessThan(r('five').indexOf('act-train-five'));
    // Initials still find an action, as a last resort.
    expect(r('ntr')).toEqual(['nav-new']);
  });

  it('highlights what matched', () => {
    const marked = (q: string, t: string) => highlight(q, t).filter((x) => x.m).map((x) => x.s);
    expect(marked('shapes s', 'shapes-small')).toEqual(['shapes', 's']);
    expect(marked('small', 'shapes-small')).toEqual(['small']);
    expect(marked('cloud', 'Open shapes in Playground')).toEqual([]);
  });
});
