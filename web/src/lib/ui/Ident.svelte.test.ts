// @vitest-environment happy-dom
/// <reference types="node" />
// The mono rule for identifiers (DESIGN_DIRECTION §2.3): an architecture, a checkpoint a run started from or a server
// id is set through ui/Ident wherever markup prints it, so no screen sets one in sans while another uses mono (round
// 20: Compare's side-by-side table and Overview's Recent runs did). Strings (palette hints, select labels, titles and
// toasts) are plain text and are not markup, so they are not checked.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import Ident from './Ident.svelte';

const SRC = resolve(process.cwd(), 'src');

function svelteFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? svelteFiles(p) : p.endsWith('.svelte') ? [p] : [];
  });
}

/** The template of a component: no script or style blocks, no comments, no quoted attribute values. */
function markup(src: string): string[] {
  return src
    .replace(/<script[\s\S]*?<\/script>/g, '')
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .split('\n')
    .map((l) => l.replace(/="[^"]*"/g, '=""'));
}

// `{run.config.arch}`, `{m.arch}`, `{from.label}`, `{lineage.label}`, `{parent.label}`, `{run.id}`, `{m.id}` as text.
const ID_TEXT = /(?<![=$])\{\s*[\w?.]*(?:\.arch|(?:from|lineage|parent)\??\.label|(?:run|m)\.id)\s*\}/;

describe('identifiers', () => {
  it('are printed through ui/Ident, never as bare text', () => {
    const bare: string[] = [];
    for (const file of svelteFiles(SRC)) {
      markup(readFileSync(file, 'utf8')).forEach((line, i) => {
        if (ID_TEXT.test(line)) bare.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
      });
    }
    expect(bare).toEqual([]);
  });

  it('render mono, as a link when given an href', () => {
    const target = document.createElement('div');
    const a = mount(Ident, { target, props: { value: 'folo-s', title: 'arch' } });
    flushSync();
    const el = target.firstElementChild as HTMLElement;
    expect(el.tagName).toBe('SPAN');
    expect(el.classList.contains('mono')).toBe(true);
    expect(el.textContent).toBe('folo-s');
    unmount(a);
    const b = mount(Ident, { target, props: { value: 'folo-s-coco80', href: '#/models/m1', class: 'link' } });
    flushSync();
    const link = target.querySelector('a')!;
    expect(link.getAttribute('href')).toBe('#/models/m1');
    expect([...link.classList]).toEqual(expect.arrayContaining(['ident', 'mono', 'link']));
    unmount(b);
  });
});
