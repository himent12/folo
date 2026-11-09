// @vitest-environment happy-dom
// A note after a value keeps a real space before it (DESIGN_DIRECTION §2.3): round 23's Compare table read
// "0.000833(requested 0.01)" and "360 steps(3 epochs)" because its note span followed the value with no space, while
// Run's Config spaced the same note with a margin. Both now render ui/Note; no component hand-rolls one.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import Note from './Note.svelte';

const SRC = join(__dirname, '..', '..');
const svelteFiles = (dir: string): string[] =>
  readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? svelteFiles(p) : n.endsWith('.svelte') ? [p] : [];
  });

describe('Note', () => {
  it('follows its value after a real, breakable space', () => {
    const target = document.createElement('span');
    target.append('0.000833');
    const c = mount(Note, { target, props: { text: '(requested 0.01)', fact: true } });
    expect(target.textContent).toBe('0.000833 (requested 0.01)');
    expect(target.querySelector('.ui-note')?.hasAttribute('data-fact')).toBe(true);
    unmount(c);
  });

  it('is the only way a note follows a value inline', () => {
    const bad: string[] = [];
    for (const f of svelteFiles(SRC)) {
      const markup = readFileSync(f, 'utf-8').split('<style>')[0] ?? '';
      // A note span right after a value (an expression, a closing span or block), through an {#if}, with nothing
      // between: the hand-rolled form. A note on a line of its own (Versus's qualifier) is not one.
      const inline = /(?:\{[^#/:@][^}]*\}|\{\/(?:if|each)\}|<\/span>)(?:\{#if [^}]*\})?<span class="(?:note|req|requested)\b[^"]*"/g;
      for (const m of markup.matchAll(inline)) bad.push(`${relative(SRC, f)}: ${m[0]}`);
    }
    expect(bad).toEqual([]);
  });
});
