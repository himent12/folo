import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

// DetectionView's drawing rules live in its stylesheet; these read it as written, so a later edit cannot quietly fade
// the labels again or shrink a chip below the floor.
const css = readFileSync(new URL('./DetectionView.svelte', import.meta.url), 'utf8').split('<style>')[1]!.replace(/\/\*[\s\S]*?\*\//g, '');

/** The declarations of the rule whose selector list is exactly `selector`. */
function rule(selector: string): string {
  const m = new RegExp(`(?:^|})\\s*${selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*{([^}]*)}`).exec(css);
  if (!m) throw new Error(`no rule ${selector}`);
  return m[1]!;
}

describe('DetectionView styles', () => {
  it('draws full-size boxes as a 2 px stroke in a 1.5 px outer halo', () => {
    const box = rule('.box');
    expect(box).toMatch(/border:\s*2px solid var\(--c\)/);
    expect(box).toMatch(/0 0 0 1\.5px var\(--class-halo\)/);
  });

  it('keeps the thumbnails’ own stroke and halo', () => {
    const thumb = rule('.compact .box');
    expect(thumb).toMatch(/border-width:\s*1\.5px/);
    expect(thumb).toMatch(/0 0 0 2px var\(--class-halo\)/);
  });

  it('fades the stroke of the boxes around a selection, never their labels', () => {
    const recede = rule('.box.recede');
    expect(recede).not.toMatch(/opacity/);
    expect(recede).toMatch(/border-color:\s*color-mix\(in srgb, var\(--c\) 30%, transparent\)/);
    for (const sel of ['.chip', '.leader', '.box.dim']) expect(rule(sel)).not.toMatch(/(^|[^-])opacity/);
  });

  it('sets chip text at 10 px or more (xs or micro), whatever the view size', () => {
    const sizes = [...css.matchAll(/font-size:\s*([^;]+);/g)].map((m) => m[1]!.trim());
    expect(sizes.length).toBeGreaterThan(0);
    for (const s of sizes) expect(['var(--type-xs-size)', 'var(--type-micro-size)']).toContain(s);
  });
});
