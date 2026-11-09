/// <reference types="node" />
// A panel at 768–1279 px uses its width, and a dialog or frame is sized to its content and the viewport
// (DESIGN_DIRECTION §2.2, round 28). Round 27 at 820 and 1024 px: Health's four findings in a 350 px column of a 715 px
// panel, Model's metric help in one column on half its panel, the image inspector ~480 px tall in an 1180 px window
// with its Classes cut, the Playground's empty frame ~650 px tall, and a run name cut to "shapes-har…". Layout is not
// computed in the test DOM, so this reads the rules that make each of them hold.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

/** The component's style, comments stripped. */
function style(file: string): string {
  const src = readFileSync(resolve(SRC, file), 'utf8');
  return src.split('<style>')[1]!.split('</style>')[0]!.replace(/\/\*[\s\S]*?\*\//g, '');
}

/** The rules at the top level of `css`, or inside the `@media` whose query is exactly `query`. */
function rulesIn(css: string, query: string | null): { sel: string; body: string }[] {
  let scope = css;
  if (query !== null) {
    const at = css.indexOf(`@media ${query} {`);
    if (at < 0) return [];
    let depth = 0;
    let i = css.indexOf('{', at);
    const start = i + 1;
    for (; i < css.length; i++) {
      if (css[i] === '{') depth++;
      else if (css[i] === '}' && --depth === 0) break;
    }
    scope = css.slice(start, i);
  } else {
    // Drop every at-rule block, keeping the top-level rules.
    scope = css.replace(/@[^{]+\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, '');
  }
  return [...scope.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ sel: m[1]!.trim(), body: m[2]! }));
}

const body = (css: string, query: string | null, sel: string) =>
  rulesIn(css, query)
    .filter((r) => r.sel.split(',').map((s) => s.trim()).includes(sel))
    .map((r) => r.body)
    .join('\n');

describe('a panel at 768–1279 px uses its width', () => {
  it("Dataset › Health: the findings are two columns from 768 px, one on phones", () => {
    const css = style('routes/dataset/StatsTab.svelte');
    expect(body(css, '(min-width: 768px)', '.hitems')).toMatch(/grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
    // Open evidence keeps its own full-width row in the grid.
    expect(body(css, null, '.hitems li.evidence')).toMatch(/grid-column: 1 \/ -1/);
  });

  it("Model's metric help keeps its two columns down to 768 px", () => {
    const css = style('routes/Model.svelte');
    expect(body(css, null, '.k-body')).toMatch(/grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
    for (const q of ['(max-width: 1279px)', '(max-width: 1023px)']) expect(body(css, q, '.k-body'), q).not.toMatch(/grid-template-columns/);
  });

  it('a Source run name is never cut with an ellipsis', () => {
    expect(body(style('routes/Models.svelte'), null, 'a.src')).not.toMatch(/text-overflow|overflow: hidden/);
  });
});

describe('a dialog or a frame is sized to its content and the viewport', () => {
  it('the image inspector is the viewport less 24 px above and below from 768 to 1279 px, its labels scrolling', () => {
    const css = style('routes/dataset/Inspector.svelte');
    expect(body(css, null, '.insp')).toMatch(/--box-h: min\(960px, calc\(100dvh - 48px\)\)/);
    expect(body(css, '(min-width: 768px) and (max-width: 1279px)', '.insp')).toMatch(/height: var\(--box-h\)/);
    // The label list scrolls in the rail; Classes and the image's facts stay pinned under it.
    expect(body(css, null, '.labels-scroll')).toMatch(/overflow-y: auto/);
    expect(body(css, null, 'aside > section')).toMatch(/flex: none/);
    const src = readFileSync(resolve(SRC, 'routes/dataset/Inspector.svelte'), 'utf8');
    expect(src).toMatch(/<div class="labels-scroll scroll-shade">\s*<ul class="labels"/);
  });

  it("the Playground's empty frame on a tablet is as tall as its prompt", () => {
    const css = style('routes/playground/DropZone.svelte');
    expect(body(css, '(min-width: 768px) and (max-width: 1023px)', '.finder')).toMatch(/height: auto/);
  });

  it("a stretched panel's foot and a card's strip sit on the bottom edge", () => {
    expect(body(style('routes/Overview.svelte'), null, '.unranked')).toMatch(/margin-top: auto/);
    expect(body(style('routes/RunNew.svelte'), null, '.card')).toMatch(/grid-template-rows: 1fr auto/);
  });
});
