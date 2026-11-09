// @vitest-environment happy-dom
/// <reference types="node" />
// Input works the same everywhere (DESIGN_DIRECTION §2.14, touch text): on a coarse pointer every editable control's own
// text is at least 16 px (`type.lg`), or iOS zooms the page into the field when it takes focus. Controls inherit 12–14 px
// text everywhere else, so each component that renders an editable `<input>`, a `<select>` or a `<textarea>` carries an
// `@media (pointer: coarse)` rule that sets that element's `font-size` to `--type-lg-size` (or larger); the labels
// around it keep their sizes. This scan fails on a component that renders one without it.
//
// A rule counts only if its selector matches the control as the component renders it (code review, round 27: a scan of
// selector text alone passed `.unused input { … }`). The component's markup is parsed into a DOM (Svelte's `{…}` blocks
// and expressions reduced to placeholders, `class:x` directives applied as classes) and each editable element there must
// `matches()` one of the coarse 16 px selectors. Computed sizes are out of reach here: in the test run Svelte's
// component CSS is not loaded into the page and happy-dom does not evaluate `(pointer: coarse)`, so the selector is what
// can be proved against what renders.
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

/** The selectors (`:global()` unwrapped) of the coarse rules that set text to 16 px or more. */
export function touchSelectors(blocks: string[]): string[] {
  return blocks.flatMap((b) =>
    [...b.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
      .filter(([, , body]) => /font-size:\s*var\(--type-(lg|xl|2xl)-size\)/.test(body!))
      .flatMap(([, sel]) => sel!.replace(/\/\*[\s\S]*?\*\//g, '').split(','))
      .map((sel) => sel.replace(/:global\(([^()]*(?:\([^()]*\)[^()]*)*)\)/g, '$1').trim())
      .filter(Boolean),
  );
}

/** Every `{…}` group (braces balanced) of a component's markup, replaced by `_`: blocks, expressions, attribute values. */
function placeholders(markup: string): string {
  let out = '';
  let depth = 0;
  for (const c of markup) {
    if (c === '{') {
      if (depth++ === 0) out += '_';
    } else if (c === '}') depth = Math.max(0, depth - 1);
    else if (depth === 0) out += c;
  }
  return out;
}

/**
 * A component's markup as a DOM: Svelte's syntax reduced to placeholders, a component's tag renamed (`<Input>` is not
 * an `<input>`: HTML parsing lowercases it), a self-closing tag closed (an HTML parser would nest its siblings inside
 * it) and every `class:x` directive applied as the class `x`.
 */
export function renderMarkup(markup: string): HTMLElement {
  const html = placeholders(markup)
    .replace(/<(\/?)([A-Z][\w.]*|svelte:[\w-]+)/g, '<$1c-$2')
    .replace(/<([A-Za-z][\w:.-]*)([^<>]*?)\/>/g, '<$1$2></$1>');
  const root = document.createElement('div');
  root.innerHTML = html;
  for (const el of root.querySelectorAll('*'))
    for (const a of [...el.attributes]) if (a.name.startsWith('class:')) el.classList.add(a.name.slice('class:'.length));
  return root;
}

/** The editable controls in a rendered component (the same rule as `editables`). */
function editableEls(root: HTMLElement): Element[] {
  return [...root.querySelectorAll('input, select, textarea')].filter((el) => {
    const type = el.getAttribute('type');
    return !(el.tagName === 'INPUT' && type && NOT_EDITABLE.test(`type="${type}"`));
  });
}

/** Whether a selector matches `el`; one happy-dom cannot parse matches nothing (and so covers nothing). */
function matchesSafe(el: Element, sel: string): boolean {
  try {
    return el.matches(sel);
  } catch {
    return false;
  }
}

describe('touch text', () => {
  it('every editable control is at least 16 px on a coarse pointer, by a rule that matches it as rendered', () => {
    const misses: string[] = [];
    let seen = 0;
    for (const file of files(SRC)) {
      const src = readFileSync(file, 'utf8');
      const style = /<style[^>]*>([\s\S]*?)<\/style>/.exec(src)?.[1] ?? '';
      // Markup only: a script's comment may name a tag ("Native <input type=range>").
      const markup = src.replace(/<(style|script)\b[\s\S]*?<\/\1>/g, '').replace(/<!--[\s\S]*?-->/g, '');
      if (!editables(markup).size) continue;
      const sels = touchSelectors(coarseBlocks(style.replace(/\/\*[\s\S]*?\*\//g, '')));
      for (const el of editableEls(renderMarkup(markup))) {
        seen++;
        if (!sels.some((sel) => matchesSafe(el, sel)))
          misses.push(`${relative(SRC, file)}: <${el.tagName.toLowerCase()}> matches no @media (pointer: coarse) rule with a font-size of --type-lg-size`);
      }
    }
    // The scan found the controls it guards (ui/Input, ui/Select, the palette's field, Add dataset's class field, …).
    expect(seen).toBeGreaterThanOrEqual(5);
    expect(misses).toEqual([]);
  });

  it('counts a rule only when its selector matches the rendered control', () => {
    const covered = (css: string, markup: string) => {
      const sels = touchSelectors(coarseBlocks(css));
      return editableEls(renderMarkup(markup)).every((el) => sels.some((sel) => matchesSafe(el, sel)));
    };
    const field = '<div class="chips" class:invalid={bad} onclick={(e) => e.x > 1}><input bind:value={q} /></div>';
    expect(covered('@media (pointer: coarse) { .chips input { font-size: var(--type-lg-size); } }', field)).toBe(true);
    // Round 27's false pass: the right element named under a class nothing renders.
    expect(covered('@media (pointer: coarse) { .unused input { font-size: var(--type-lg-size); } }', field)).toBe(false);
    // A class a directive adds counts; `:global()` is unwrapped.
    expect(covered('@media (pointer: coarse) { .invalid input { font-size: var(--type-lg-size); } }', field)).toBe(true);
    expect(covered('@media (pointer: coarse) { :global(.chips) input { font-size: var(--type-lg-size); } }', field)).toBe(true);
    // A self-closing component before the control does not swallow it, and a component named Input is not an input.
    expect(covered('@media (pointer: coarse) { .row > input { font-size: var(--type-lg-size); } }', '<div class="row"><Icon size={14} /><input /></div>')).toBe(true);
    expect(editableEls(renderMarkup('<Input bind:value={q} /><Select />'))).toHaveLength(0);
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
