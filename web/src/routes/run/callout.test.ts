/// <reference types="node" />
// Ways out and banners (DESIGN_DIRECTION §12, round 28). A block of ways out on a phone is the one remedy layout
// (`.cta-stack`, ui/Button): the primary first across the block, every other way out a full-width secondary under it,
// so run/Callout, ui/ErrorState and the Playground's notice all carry the class and nothing turns a fix into a text
// link; and a banner never tints under the lime primary: no surface fills with a status's muted tint where a primary
// can sit.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const read = (p: string) => readFileSync(resolve(process.cwd(), 'src', p), 'utf8');

describe('ways out', () => {
  it('are one remedy layout wherever a block of them leads a phone screen', () => {
    expect(read('routes/run/Callout.svelte')).toMatch(/class="c-acts cta-row cta-stack"/);
    expect(read('lib/ui/ErrorState.svelte')).toMatch(/class="acts cta-stack"/);
    expect(read('routes/Playground.svelte')).toMatch(/class="nf-actions cta-stack"/);
    const button = read('lib/ui/Button.svelte');
    expect(button).toMatch(/:global\(\.cta-stack\) > \.btn \{[^}]*flex: 1 0 100%;/);
    expect(button).toMatch(/:global\(\.cta-stack\) > \.primary \{\s*order: -1;/);
  });

  it('never turn a way out into a text link (round 27: an underlined "Use shapes-n-best" beside a lime button)', () => {
    for (const f of ['lib/ui/Button.svelte', 'lib/ui/ErrorState.svelte', 'routes/run/Callout.svelte', 'routes/Playground.svelte']) {
      expect(read(f), f).not.toMatch(/cta-links/);
    }
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
