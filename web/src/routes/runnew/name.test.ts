import { describe, expect, it } from 'vitest';
import { nameHint, runName, uniqueName } from './name';

describe('run names', () => {
  it('turns what was typed into a tidy name', () => {
    expect(runName('my first run')).toBe('my-first-run');
    expect(runName('  café: v2!  ')).toBe('caf-v2');
    expect(runName('shapes_1k.v2-n')).toBe('shapes_1k.v2-n');
    expect(runName('!!!')).toBe('');
  });

  it('suggests a unique name that fits the length limit', () => {
    expect(uniqueName('shapes-n', new Set())).toBe('shapes-n');
    expect(uniqueName('shapes-n', new Set(['shapes-n', 'shapes-n-2']))).toBe('shapes-n-3');
    const long = 'd'.repeat(70);
    expect(uniqueName(long, new Set())).toBe('d'.repeat(64));
    expect(uniqueName(long, new Set(['d'.repeat(64)]))).toBe(`${'d'.repeat(62)}-2`);
  });

  it('warns about a name that is taken, else says what a typed name is saved as', () => {
    const taken = new Set(['shapes-n']);
    expect(nameHint('shapes-n', taken, true)).toBe('A run called shapes-n already exists. Pick another name to tell them apart.');
    // The suggestion itself is unique; a taken one only comes from typing.
    expect(nameHint('shapes-n', taken, false)).toBeNull();
    expect(nameHint('my run', taken, true)).toBe('Saved as my-run: letters, digits, dot, dash and underscore.');
    expect(nameHint('my-run', taken, true)).toBeNull();
  });
});
