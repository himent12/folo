// @vitest-environment happy-dom
// A note after a value keeps a real space before it: round 23's Compare table read "0.000833(requested 0.01)".
import { mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import Note from './Note.svelte';

describe('Note', () => {
  it('follows its value after a real, breakable space', () => {
    const target = document.createElement('span');
    target.append('0.000833');
    const c = mount(Note, { target, props: { text: '(requested 0.01)', fact: true } });
    expect(target.textContent).toBe('0.000833 (requested 0.01)');
    unmount(c);
  });
});
