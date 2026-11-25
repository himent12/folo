import { describe, expect, it } from 'vitest';
import { highlight, rank, type Rankable } from './commands';
import { groupResults, highlightItem } from './commandPalette/scoring';
import type { Item } from './commandPalette/types';

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

  it('finds an entity by its id, whole or by its start or end, but not by a word of it (shell F3)', () => {
    const withIds = [
      ...items,
      { id: 'run-a', group: 'Runs', label: 'shapes long', entityId: 'run_8f5b', scoped: true },
      { id: 'run-b', group: 'Runs', label: 'bccd baseline', entityId: 'run_7b01', scoped: true },
      { id: 'ds-b', group: 'Datasets', label: 'bccd', entityId: 'ds_1d06', scoped: true },
    ];
    const r = (q: string) => rank(withIds, q).map((i) => i.id);
    expect(r('run_8f5b')).toEqual(['run-a']);
    expect(r('RUN_8F5B ')).toEqual(['run-a']);
    expect(r('8f5b')).toEqual(['run-a']);
    expect(r('ds_1d')).toEqual(['ds-b']);
    // A whole id wins over a thing named like it (review of #223).
    expect(rank([...withIds, { id: 'model-x', group: 'Models', label: 'run_8f5b', scoped: true }], 'run_8f5b').map((i) => i.id)).toEqual(['run-a', 'model-x']);
    // Three characters name no id: "7b0" is not every run whose id holds it.
    expect(r('8f5')).toEqual([]);
    expect(r('b01')).toEqual([]);
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
    expect(highlight('', 'shapes-small')).toEqual([{ s: 'shapes-small', m: false }]);
    expect(highlight('shapes', '')).toEqual([]);
  });

  it('re-scores cleanly across repeated keystrokes and mutated items without stale cache hits', () => {
    const dynamic: (Rankable & { id: string })[] = [
      { id: 'a', group: 'Runs', label: 'alpha-baseline', entityId: 'run_1111', scoped: true },
      { id: 'b', group: 'Runs', label: 'beta-baseline', entityId: 'run_2222', scoped: true },
    ];
    expect(rank(dynamic, 'alpha').map((x) => x.id)).toEqual(['a']);
    expect(rank(dynamic, 'beta').map((x) => x.id)).toEqual(['b']);
    // Mutating label/match (including in-place push) invalidates the WeakMap entry automatically.
    dynamic[0]!.label = 'gamma-tuned';
    dynamic[0]!.match = ['special-alias'];
    expect(rank(dynamic, 'alpha').map((x) => x.id)).toEqual([]);
    expect(rank(dynamic, 'gamma').map((x) => x.id)).toEqual(['a']);
    expect(rank(dynamic, 'special').map((x) => x.id)).toEqual(['a']);
    dynamic[0]!.match.push('pushed-alias');
    expect(rank(dynamic, 'pushed').map((x) => x.id)).toEqual(['a']);
  });

  it('never duplicates text in highlight/highlightItem when lowercasing expands UTF-16 length', () => {
    for (const label of ['İa', 'İ a', 'İabc']) {
      const parts = highlight('a', label);
      expect(parts.map((p) => p.s).join('')).toBe(label);
      const itemParts = highlightItem('a', { group: 'Runs', label });
      expect(itemParts.map((p) => p.s).join('')).toBe(label);
    }
  });

  it('groups non-descriptive groups ahead of describes-only groups with sequential visual indices', () => {
    const stub = (id: string, group: string, describes?: boolean) => ({ id, group, label: id, describes, run: () => {} }) as unknown as Item;
    const grouped = groupResults([stub('act-1', 'Actions', true), stub('ds-1', 'Datasets'), stub('act-2', 'Actions', true)]);
    expect(grouped.map((g) => g.group)).toEqual(['Datasets', 'Actions']);
    expect(grouped.flatMap((g) => g.items.map((x) => [x.item.id, x.n]))).toEqual([
      ['ds-1', 0],
      ['act-1', 1],
      ['act-2', 2],
    ]);
  });
});
