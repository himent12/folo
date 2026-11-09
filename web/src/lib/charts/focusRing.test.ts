// A focusable chart (role="group", aria-roledescription="chart") takes the global :focus-visible ring (base.css), never
// a halo of its own (DESIGN_DIRECTION §2.9): round 22 found Histogram's 2 px ring 4 px out, unlike every other control.
// A proxy ring (a menu option drawn around its focused radio, `:has(input:focus-visible)`) is not the element's own.
//
// Every other control too: round 23 found Input, Select and Add dataset's class field drawing an 18 % lime halo and a
// lime edge on `:focus-within`, a second focus style that showed on mouse clicks and was under 3:1 in light. A field's
// frame takes the global ring through `:has(input:focus-visible)`; a proxy (a slider's thumb) keeps its width and colour.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
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

  it('no control draws a focus style of its own: no halo, no lime edge, nothing on mouse focus', () => {
    const src = join(__dirname, '..', '..');
    const files = (dir: string): string[] =>
      readdirSync(dir).flatMap((n) => {
        const p = join(dir, n);
        return statSync(p).isDirectory() ? files(p) : n.endsWith('.svelte') ? [p] : [];
      });
    const RING = /^0 0 0 (2px|var\(--size-focus-ring\)) var\(--focus\)$/;
    const bad: string[] = [];
    for (const f of files(src)) {
      const style = readFileSync(f, 'utf-8').split('<style>')[1] ?? '';
      for (const m of style.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/([^{}]*:focus[^{}]*)\{([^}]*)\}/g)) {
        const selector = (m[1] ?? '').trim().replace(/\s+/g, ' ');
        // A state shown while not focused (`.cell.pin:not(:focus-visible)`) is not a focus style.
        if (!/:focus/.test(selector.replace(/:not\([^)]*\)/g, ''))) continue;
        const body = m[2] ?? '';
        const where = `${relative(src, f)}: ${selector}`;
        const draws = /outline(?!-offset)(?!:\s*none)|box-shadow|border-color/.test(body);
        // Mouse focus: :focus-within and a bare :focus never draw (the ring is :focus-visible only).
        if (/:focus-within|:focus(?![-\w])/.test(selector) && draws) bad.push(`${where} (draws on mouse focus)`);
        // A lime (or danger) edge as the focus mark.
        if (/border-color:\s*var\(--focus\)/.test(body)) bad.push(`${where} (focus edge)`);
        // A halo: any shadow on focus but a proxy ring of the ring's own width and colour.
        for (const sh of body.matchAll(/box-shadow:\s*([^;]+);/g))
          if (!RING.test((sh[1] ?? '').trim()) && (sh[1] ?? '').trim() !== 'none') bad.push(`${where} (halo: ${sh[1]})`);
      }
    }
    expect(bad).toEqual([]);
  });
});
