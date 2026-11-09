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

  it('draws thumbnails in the compact style: a 1 px stroke in a 1 px halo, full weight only when hovered or selected', () => {
    const thumb = rule('.compact .box');
    expect(thumb).toMatch(/border-width:\s*1px/);
    expect(thumb).toMatch(/box-shadow:\s*0 0 0 1px var\(--class-halo\)/);
    expect(rule('.compact .box.active,\n  .compact .box.selected')).toMatch(/border-width:\s*2px/);
    // Truth stays dashed (predictions are solid), just outside the prediction's halo.
    expect(rule('.compact .box.truth')).toMatch(/outline:\s*1px dashed/);
  });

  it('fades everything around a selection to 30 %, chips included, and edges the selected chip in the accent', () => {
    expect(rule('.box.recede')).toMatch(/border-color:\s*color-mix\(in srgb, var\(--c\) 30%, transparent\)/);
    expect(rule('.box.recede .chip,\n  .box.recede .leader')).toMatch(/opacity:\s*0\.3/);
    expect(rule('.box.truth.fade')).toMatch(/opacity:\s*0\.3/);
    expect(rule('.box.selected .chip')).toMatch(/var\(--accent-solid\)/);
    // Without a selection nothing fades a label: hovering one box dims the other strokes only.
    for (const sel of ['.chip', '.leader', '.box.dim']) expect(rule(sel)).not.toMatch(/(^|[^-])opacity/);
  });

  it('frames a small selected box with lime viewfinder corners', () => {
    expect(rule('.vf')).toMatch(/var\(--accent-solid\)/);
    expect(readFileSync(new URL('./DetectionView.svelte', import.meta.url), 'utf8')).toMatch(/selected === i && small32\(b\)/);
  });

  it('sets chip text at 10 px or more (xs or micro), whatever the view size', () => {
    const sizes = [...css.matchAll(/font-size:\s*([^;]+);/g)].map((m) => m[1]!.trim());
    expect(sizes.length).toBeGreaterThan(0);
    for (const s of sizes) expect(['var(--type-xs-size)', 'var(--type-micro-size)']).toContain(s);
  });
});
