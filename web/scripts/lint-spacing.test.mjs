import { describe, expect, it } from 'vitest';
import { lintCss, lintMedia, lintScriptMedia, lintSmallText } from './lint-spacing.mjs';

describe('lintCss', () => {
  it('flags off-grid px values and accepts grid ones', () => {
    expect(lintCss('a { padding: 8px 12px; }')).toEqual([]);
    expect(lintCss('a { padding: 3px; }')).toHaveLength(1);
  });

  it('reads declarations whose value continues on the next line', () => {
    const out = lintCss('a {\n  padding:\n    3px;\n}');
    expect(out).toHaveLength(1);
    expect(out[0]).toContain('line 2');
  });

  it('ignores commented-out code, also across lines', () => {
    expect(lintCss('a {\n  /* padding: 3px;\n     margin: 5px; */\n  gap: 8px;\n}')).toEqual([]);
  });

  it('skips var() math and grid-ok exemptions', () => {
    expect(lintCss('a { top: calc(var(--x) - 3px); }')).toEqual([]);
    expect(lintCss('a { margin: 7px; /* grid-ok */ }')).toEqual([]);
  });

  it('flags px type sizes and accepts tokens and em', () => {
    expect(lintCss('a { font-size: 15px; }')[0]).toContain('--type-*-size');
    expect(lintCss('a { font-size: var(--type-sm-size); }')).toEqual([]);
    expect(lintCss('a { font-size: 0.7em; }')).toEqual([]);
    expect(lintCss('a { font-size: 9px; /* grid-ok */ }')).toEqual([]);
    // The shorthand sets a size too.
    expect(lintCss('a { font: 10px Geist; }')[0]).toContain('--type-*-size');
    expect(lintCss('a { font: inherit; }')).toEqual([]);
  });
});

describe('lintMedia: media widths are tier boundaries', () => {
  it('accepts the tiers, written as min-width N / max-width N - 1', () => {
    for (const q of ['(max-width: 767px)', '(min-width: 768px)', '(max-width: 1023px)', '(min-width: 1024px)', '(max-width: 1279px)', '(min-width: 1280px)'])
      expect(lintMedia(`@media ${q} { a { color: red; } }`), q).toEqual([]);
    expect(lintMedia('@media (min-width: 768px) and (max-width: 1023px) { a { color: red; } }')).toEqual([]);
  });

  it('rejects any other width, an off-by-one boundary, other units and the range syntax', () => {
    for (const q of ['(max-width: 959px)', '(max-width: 768px)', '(min-width: 767px)', '(max-width: 640px)', '(max-width: 48em)', '(width < 768px)', '(min-width: 768px) and (max-width: 1199px)'])
      expect(lintMedia(`@media ${q} { a { color: red; } }`), q).toHaveLength(1);
  });

  it('leaves non-width features, container queries and comments alone, and reports the line', () => {
    expect(lintMedia('@media (pointer: coarse) { a {} }\n@media (prefers-reduced-motion: reduce) { a {} }')).toEqual([]);
    expect(lintMedia('@container (max-width: 640px) { a {} }')).toEqual([]);
    expect(lintMedia('/* @media (max-width: 959px) */ a {}')).toEqual([]);
    expect(lintMedia('a {}\n\n@media (max-width: 1099px) { a {} }', null, 10)[0]).toContain('line 12');
  });

  it('in script, asks for the tokens instead of a spelled-out width query', () => {
    expect(lintScriptMedia("const q = matchMedia('(max-width: 959px)');")[0]).toContain('not a tier boundary');
    expect(lintScriptMedia("const q = new MediaQuery('max-width: 767px', false);")[0]).toContain('use `media`');
    expect(lintScriptMedia('const q = new MediaQuery(media.phone, false);')).toEqual([]);
    expect(lintScriptMedia("const m = matchMedia('(prefers-reduced-motion: reduce)');")).toEqual([]);
    // Inline styles that set a width are not media queries.
    expect(lintScriptMedia('<div style="max-width: 320px"></div>')).toEqual([]);
  });
});

describe('lintSmallText: the 12 px text floor', () => {
  it('flags unannotated xs and micro sizes and line heights', () => {
    expect(lintSmallText('a { font-size: var(--type-xs-size); }', null, { css: true })).toHaveLength(1);
    expect(lintSmallText('a { line-height: var(--type-micro-line); }', null, { css: true })).toHaveLength(1);
    expect(lintSmallText('a { font-size: var(--type-sm-size); }', null, { css: true })).toEqual([]);
  });

  it('accepts an annotation on the same line or the line above', () => {
    expect(lintSmallText('a { font-size: var(--type-xs-size); /* small: chart tick */ }', null, { css: true })).toEqual([]);
    expect(lintSmallText('a {\n  /* small: image chip */\n  font-size: var(--type-micro-size);\n}', null, { css: true })).toEqual([]);
    // Two lines above is too far.
    expect(lintSmallText('/* small: image chip */\na {\n  font-size: var(--type-micro-size);\n}', null, { css: true })).toHaveLength(1);
  });

  it('only lets each reason use the steps it allows', () => {
    expect(lintSmallText('a { font-size: var(--type-micro-size); /* small: chart tick */ }', null, { css: true })[0]).toContain('does not allow micro');
    expect(lintSmallText('a { font-size: var(--type-xs-size); /* small: keycap mark */ }', null, { css: true })[0]).toContain('does not allow xs');
    expect(lintSmallText('a { font-size: var(--type-xs-size); /* small: hint */ }', null, { css: true })).toHaveLength(1);
    expect(lintSmallText('a { font-size: var(--type-xs-size); /* small: image chip */ }', null, { css: true })).toEqual([]);
  });

  it('checks markup and TS too, and a px font size outside a stylesheet', () => {
    expect(lintSmallText('<span style="font-size: var(--type-xs-size)">x</span>')).toHaveLength(1);
    expect(lintSmallText('<!-- small: chart tick -->\n<text style="font-size: var(--type-xs-size)">1</text>')).toEqual([]);
    expect(lintSmallText("ctx.font = '10px Geist';")[0]).toContain('outside a stylesheet');
    expect(lintSmallText('<text font-size="10px">1</text>')[0]).toContain('outside a stylesheet');
    // A commented-out use is not a use.
    expect(lintSmallText('// font-size: var(--type-xs-size)')).toEqual([]);
    // A token in another declaration on the line exempts nothing.
    expect(lintSmallText('<span style="font-size:10px; line-height:var(--type-sm-line)">x</span>')[0]).toContain('font-size:10px');
    expect(lintSmallText('<text style="font-size: var(--type-sm-size)" x="10px">1</text>')).toEqual([]);
  });
});
