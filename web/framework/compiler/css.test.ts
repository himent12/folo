// Scoping: fw's CSS scoping mirrors Svelte 5's analyzer and transformer. Until Studio left Svelte, every <style> of
// the app was also compiled by the Svelte compiler and compared selector for selector (171 files, 3,950 selectors, all
// identical in text and specificity); what stays is each rule of that scoping, as Svelte 5 printed it.
import { describe, expect, it } from 'vitest';
import { scopeCss } from './css';

/** The scope class Svelte printed in these cases (the hash is the only part that is the compiler's). */
const HASH = 'svelte-h4sh';

const norm = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\s+/g, ' ').replace(/\s*([>+~,()])\s*/g, '$1').trim();

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
