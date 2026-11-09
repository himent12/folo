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

  it('never include a named optimizer, which is a name in sans ("AdamW": its struck W reads as "₩" in mono)', () => {
    const OPTIMIZER_AS_IDENT = /<Ident\b[^>]*optimi[sz]/i;
    const bare: string[] = [];
    for (const file of svelteFiles(SRC))
      markup(readFileSync(file, 'utf8')).forEach((line, i) => {
        if (OPTIMIZER_AS_IDENT.test(line)) bare.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
      });
    expect(bare).toEqual([]);
  });

  it('never ride inside a plain-text label: a select option, a palette hint or a meta string', () => {
    // Text that cannot hold markup (a native <option>, a palette hint string) would set an architecture in sans: such
    // labels name the entity only (the select) or pass the identifier as its own part (`{ ident }` in the palette).
    const ARCH_IN_LABEL = /\b(?:label|hint)\s*:\s*`[^`]*\$\{[^}]*\.arch\b[^}]*\}/;
    const bare: string[] = [];
    for (const file of svelteFiles(SRC)) {
      readFileSync(file, 'utf8')
        .split('\n')
        .forEach((line, i) => {
          if (ARCH_IN_LABEL.test(line)) bare.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(bare).toEqual([]);
  });

  it('commands in markup are code: mono, or a ui/CommandLine', () => {
    // `folo serve`, `folo export …` in a sentence sits in a mono span or <code> of its own. Attribute values are left
    // out (a Field hint renders `backticks` as code); only text a reader sees in sans is a failure.
    const CMD = /folo (?:serve|train|predict|export|doctor|dataset|mcp|val)\b/g;
    const bare: string[] = [];
    for (const file of svelteFiles(SRC)) {
      readFileSync(file, 'utf8')
        .replace(/<script[\s\S]*?<\/script>/g, '')
        .replace(/<style[\s\S]*?<\/style>/g, '')
        .replace(/<!--[\s\S]*?-->/g, '')
        .split('\n')
        .forEach((raw, i) => {
          // Keep class attributes (they say mono), drop every other attribute value.
          const line = raw.replace(/\s(?!class=)[\w:-]+=("[^"]*"|\{[^}]*\})/g, '');
          for (const m of line.matchAll(CMD)) {
            const open = [...line.slice(0, m.index).matchAll(/<(\w+)([^>]*)>/g)].at(-1);
            const code = open && (open[1] === 'code' || /class="[^"]*\bmono\b/.test(open[2] ?? ''));
            if (!code) bare.push(`${relative(SRC, file)}:${i + 1}: ${raw.trim()}`);
          }
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
