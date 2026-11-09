import { describe, expect, it } from 'vitest';
import { rank, type Rankable } from './commands';

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
});
