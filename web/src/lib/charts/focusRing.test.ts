// A focusable chart (role="group", aria-roledescription="chart") takes the global :focus-visible ring (base.css), never
// a halo of its own (DESIGN_DIRECTION §2.9): round 22 found Histogram's 2 px ring 4 px out, unlike every other control.
// A proxy ring (a menu option drawn around its focused radio, `:has(input:focus-visible)`) is not the element's own.
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('chart focus', () => {
  it('no chart draws its own focus ring', () => {
    const bad: string[] = [];
    for (const n of readdirSync(__dirname).filter((n) => n.endsWith('.svelte'))) {
      const style = readFileSync(join(__dirname, n), 'utf-8').split('<style>')[1] ?? '';
      for (const m of style.matchAll(/([^{}]*:focus(?:-visible)?)\s*\{([^}]*)\}/g)) {
        const selector = (m[1] ?? '').trim();
        if (/:has\(|[+~]/.test(selector)) continue;
        if (/outline(?!:\s*none)/.test(m[2] ?? '')) bad.push(`${n}: ${selector}`);
      }
    }
    expect(bad).toEqual([]);
  });
});
