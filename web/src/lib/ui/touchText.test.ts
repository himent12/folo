/// <reference types="node" />
// Input works the same everywhere (DESIGN_DIRECTION §2.14, touch text): on a coarse pointer every editable control's own
// text is at least 16 px (`type.lg`), or iOS zooms the page into the field when it takes focus. Controls inherit 12–14 px
// text everywhere else, so each component that renders an editable `<input>`, a `<select>` or a `<textarea>` carries an
// `@media (pointer: coarse)` rule that sets that element's `font-size` to `--type-lg-size` (or larger); the labels
// around it keep their sizes. This scan fails on a component that renders one without it.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return files(p);
    return p.endsWith('.svelte') ? [p] : [];
  });
}

/** Input types a person types into or picks text in (the rest are toggles, sliders and pickers with no text). */
const NOT_EDITABLE = /type=["'](checkbox|radio|range|file|color|hidden|button|submit|reset|image)["']/;

/** A tag's attributes, up to its closing `>`: braces skipped, so `oninput={(e) => …}` does not end it early. */
function attrs(markup: string, from: number): string {
  let depth = 0;
  for (let i = from; i < markup.length; i++) {
    const c = markup[i];
    if (c === '{') depth++;
    else if (c === '}') depth--;
    else if (c === '>' && depth === 0) return markup.slice(from, i);
  }
  return markup.slice(from);
}

/** The editable elements a component renders: `input`, `select`, `textarea`. */
export function editables(markup: string): Set<string> {
  const out = new Set<string>();
  for (const m of markup.matchAll(/<(input|select|textarea)\b/g)) {
    if (m[1] === 'input' && NOT_EDITABLE.test(attrs(markup, m.index! + m[0].length))) continue;
    out.add(m[1]!);
  }
  return out;
}

/** The bodies of every `@media (pointer: coarse)` block in a stylesheet (braces matched). */
export function coarseBlocks(css: string): string[] {
  const out: string[] = [];
  for (const m of css.matchAll(/@media\s*\(pointer:\s*coarse\)\s*\{/g)) {
    let depth = 1;
    let i = m.index! + m[0].length;
    const start = i;
    for (; i < css.length && depth > 0; i++) {
      if (css[i] === '{') depth++;
      else if (css[i] === '}') depth--;
    }
    out.push(css.slice(start, i - 1));
  }
  return out;
}

/** Whether a coarse block sets `el`'s text to 16 px or more: a rule naming the element with a type.lg-or-larger size. */
export function coversTouchText(blocks: string[], el: string): boolean {
  const named = new RegExp(`(^|[\\s,>+~(])${el}\\b`);
  return blocks.some((b) =>
    [...b.matchAll(/([^{}]+)\{([^{}]*)\}/g)].some(
      ([, sel, body]) => named.test(sel!.replace(/\/\*[\s\S]*?\*\//g, '')) && /font-size:\s*var\(--type-(lg|xl|2xl)-size\)/.test(body!),
    ),
  );
}

describe('touch text', () => {
  it('every editable control is at least 16 px on a coarse pointer', () => {
    const misses: string[] = [];
    let seen = 0;
    for (const file of files(SRC)) {
      const src = readFileSync(file, 'utf8');
      const style = /<style[^>]*>([\s\S]*?)<\/style>/.exec(src)?.[1] ?? '';
      // Markup only: a script's comment may name a tag ("Native <input type=range>").
      const markup = src.replace(/<(style|script)\b[\s\S]*?<\/\1>/g, '').replace(/<!--[\s\S]*?-->/g, '');
      const blocks = coarseBlocks(style.replace(/\/\*[\s\S]*?\*\//g, ''));
      for (const el of editables(markup)) {
        seen++;
        if (!coversTouchText(blocks, el)) misses.push(`${relative(SRC, file)}: <${el}> has no @media (pointer: coarse) font-size of --type-lg-size`);
      }
    }
    // The scan found the controls it guards (ui/Input, ui/Select, the palette's field, Add dataset's class field, …).
    expect(seen).toBeGreaterThanOrEqual(5);
    expect(misses).toEqual([]);
  });

  it('reads a rule only for the element it names, and only at 16 px or more', () => {
    expect(coversTouchText(coarseBlocks('@media (pointer: coarse) { .sm input, input { font-size: var(--type-lg-size); } }'), 'input')).toBe(true);
    expect(coversTouchText(coarseBlocks('@media (pointer: coarse) { input { font-size: var(--type-md-size); } }'), 'input')).toBe(false);
    expect(coversTouchText(coarseBlocks('@media (pointer: coarse) { .inputs { font-size: var(--type-lg-size); } }'), 'input')).toBe(false);
    expect(coversTouchText(coarseBlocks('@media (pointer: fine) { input { font-size: var(--type-lg-size); } }'), 'input')).toBe(false);
    expect([...editables('<input type="checkbox" /><input bind:value={q} /><select></select>')]).toEqual(['input', 'select']);
    expect([...editables('<input oninput={(e) => go(e)} type="range" />')]).toEqual([]);
  });
});
