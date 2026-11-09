/// <reference types="node" />
// Ways out and banners (DESIGN_DIRECTION §12, round 27). A block of ways out on a phone is one button and a row of text
// links (`.cta-links`, ui/Button), so run/Callout, ui/ErrorState and the Playground's notice all carry the class; and a
// banner never tints under the lime primary: no surface fills with a status's muted tint where a primary can sit.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const read = (p: string) => readFileSync(resolve(process.cwd(), 'src', p), 'utf8');

describe('ways out', () => {
  it('are one button and a row of links wherever a block of them leads a phone screen', () => {
    expect(read('routes/run/Callout.svelte')).toMatch(/class="c-acts cta-row cta-links"/);
    expect(read('lib/ui/ErrorState.svelte')).toMatch(/class="acts cta-links"/);
    expect(read('routes/Playground.svelte')).toMatch(/class="nf-actions cta-links"/);
    expect(read('lib/ui/Button.svelte')).toMatch(/\.cta-links:has\(> \.primary\) > \.btn:not\(\.primary\)/);
  });

  it('leave no empty row when the actions render nothing', () => {
    expect(read('routes/run/Callout.svelte')).toMatch(/\.c-acts:not\(:has\(\*\)\)\s*\{\s*display: none;/);
  });
});

describe('banners', () => {
  it('never fill with a status tint (the lime primary sits on the panel surface)', () => {
    const css = read('routes/run/Callout.svelte').split('<style>')[1]!.replace(/\/\*[\s\S]*?\*\//g, '');
    expect(css).not.toMatch(/background:[^;]*--(danger|warning)-muted/);
    expect(css).toMatch(/\.callout\.danger\s*\{[^}]*border-left-color: var\(--danger-solid\)/);
  });
});
