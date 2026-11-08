import { describe, expect, it } from 'vitest';
import { lintCss } from './lint-spacing.mjs';

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
  });
});
