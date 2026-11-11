// The scoping parity guarantee: every <style> block of every .svelte file in web/src is scoped by the Svelte compiler
// and by ours, and every selector Svelte keeps must come out the same (text, modulo whitespace) with the same
// specificity. Selectors Svelte drops as unused (it sees the elements; we cannot) are skipped: we keep them, scoped.
import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import postcss from 'postcss';
import { compile } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';
import { scopeCss, specificity } from './css';

const web = fileURLToPath(new URL('../..', import.meta.url));
const HASH = 'svelte-h4sh';

function svelteFiles(dir: string): string[] {
  const out: string[] = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...svelteFiles(p));
    else if (e.name.endsWith('.svelte')) out.push(p);
  }
  return out;
}

/** Splits a selector list at top-level commas. */
function splitList(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let quote = '';
  let cur = '';
  for (const ch of s) {
    if (quote) {
      if (ch === quote) quote = '';
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') depth--;
    else if (ch === ',' && depth === 0) {
      out.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((x) => x.trim()).filter(Boolean);
}

const norm = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\s+/g, ' ').replace(/\s*([>+~,()])\s*/g, '$1').trim();
/** The selector without any scoping: what both sides started from. */
const skeleton = (s: string) => norm(s).replaceAll(`:where(.${HASH})`, '').replaceAll(`.${HASH}`, '').replace(/^&/, '');

interface RuleInfo {
  selectors: string[];
  decls: string;
}

function rules(css: string, svelte: boolean): RuleInfo[] {
  const out: RuleInfo[] = [];
  postcss.parse(css).walkRules((r) => {
    const decls = (r.nodes ?? []).filter((n) => n.type === 'decl').map((n) => n.toString().replace(/\s+/g, ' ')).join(';');
    // Svelte comments out unused selectors inside a list, comma included.
    const sel = svelte ? r.selector.replace(/\/\*\s*\(unused\)[\s\S]*?\*\//g, ',') : r.selector;
    out.push({ selectors: splitList(sel).filter((x) => norm(x) !== ''), decls });
  });
  return out;
}

describe('scoped CSS matches Svelte 5', () => {
  const files = svelteFiles(join(web, 'src')).filter((f) => /<style[\s>]/.test(readFileSync(f, 'utf8')));
  it(`covers the app (${files.length} files with styles)`, () => {
    expect(files.length).toBeGreaterThan(100);
  });

  const stats = { selectors: 0, exact: 0, files: 0 };
  const textDiffs: string[] = [];
  for (const file of files) {
    const name = relative(web, file).replace(/\\/g, '/');
    it(name, () => {
      const source = readFileSync(file, 'utf8');
      const style = source.match(/<style[^>]*>([\s\S]*?)<\/style>/)![1]!;
      const theirs = compile(source, { filename: file, css: 'external', cssHash: () => HASH, generate: 'client' }).css!.code;
      const ours = scopeCss(style, HASH);
      const a = rules(theirs, true);
      const b = rules(ours, false);
      let j = 0;
      for (const ra of a) {
        if (ra.selectors.length === 0) continue;
        while (j < b.length && (b[j]!.decls !== ra.decls || !ra.selectors.every((s) => b[j]!.selectors.some((t) => skeleton(t) === skeleton(s))))) j++;
        expect(j, `${name}: no rule of ours for "${ra.selectors.join(', ')}"`).toBeLessThan(b.length);
        const rb = b[j++]!;
        for (const s of ra.selectors) {
          const t = rb.selectors.find((x) => skeleton(x) === skeleton(s))!;
          stats.selectors++;
          expect(specificity(norm(t)), `${name}: specificity of "${t}" vs Svelte's "${s}"`).toEqual(specificity(norm(s)));
          if (norm(t) === norm(s)) stats.exact++;
          else textDiffs.push(`${name}: svelte "${norm(s)}" / fw "${norm(t)}"`);
        }
      }
      // @keyframes names and animation references.
      const kf = (css: string) => [...css.matchAll(/@keyframes\s+([\w-]+)/g)].map((m) => m[1]).sort();
      expect(kf(ours)).toEqual(kf(theirs));
      const anim = (css: string) => [...css.matchAll(/animation(?:-name)?\s*:\s*([^;}]+)/g)].map((m) => m[1]!.trim()).sort();
      expect(anim(ours)).toEqual(anim(theirs));
      stats.files++;
    });
  }

  it('selectors matched exactly (report)', () => {
    console.log(`scoped CSS parity: ${stats.files} files, ${stats.selectors} selectors compared, ${stats.exact} identical, ${textDiffs.length} differ in text only (same specificity)`);
    for (const d of textDiffs) console.log('  ' + d);
    expect(textDiffs).toEqual([]);
  });
});

describe('scoping edge cases (verbatim Svelte 5 output)', () => {
  const cases: [string, string][] = [
    ['.a { color: red }', '.a.svelte-h4sh { color: red }'],
    ['.a .b > c::after {}', '.a.svelte-h4sh .b:where(.svelte-h4sh) > c:where(.svelte-h4sh)::after {}'],
    ['* {}', '.svelte-h4sh {}'],
    [':hover {}', '.svelte-h4sh:hover {}'],
    ['.danger :global(svg) {}', '.danger.svelte-h4sh svg {}'],
    [":global([data-theme='light']) .primary {}", "[data-theme='light'] .primary.svelte-h4sh {}"],
    ['div :global.x {}', 'div.svelte-h4sh.x {}'],
    ['.a :global { .b { c: d } }', '.a.svelte-h4sh { .b { c: d } }'],
    [':global { .x { c: d } }', '.x { c: d }'],
    ['.p { .q { c: d } &:hover { e: f } }', '.p.svelte-h4sh { .q:where(.svelte-h4sh) { c: d } &:hover { e: f } }'],
    ['.btn:disabled:not(.ghost, [aria-busy=\'true\']) {}', ".btn.svelte-h4sh:disabled:not(.ghost, [aria-busy='true']) {}"],
    ['.x:has(.y) {}', '.x.svelte-h4sh:has(.y:where(.svelte-h4sh)) {}'],
    ['[role=menuitem].danger:is(:hover, :focus-visible) :global(svg) {}', '[role=menuitem].danger.svelte-h4sh:is(:where(.svelte-h4sh):hover, :where(.svelte-h4sh):focus-visible) svg {}'],
    ['@keyframes in { from { opacity: 0 } } .l { animation: in 1s }', '@keyframes svelte-h4sh-in { from { opacity: 0 } } .l.svelte-h4sh { animation: svelte-h4sh-in 1s }'],
    ['@keyframes -global-spin {} .l { animation: spin 1s }', '@keyframes spin {} .l.svelte-h4sh { animation: spin 1s }'],
    ['@media (max-width: 767px) { :global(.cta-row) .btn { a: b } }', '@media (max-width: 767px) { .cta-row .btn.svelte-h4sh { a: b } }'],
    [':root .x {}', ':root .x.svelte-h4sh {}'],
  ];
  for (const [input, expected] of cases) {
    it(input, () => {
      expect(norm(scopeCss(input, HASH))).toBe(norm(expected));
    });
  }
});
