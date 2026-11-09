/// <reference types="node" />
// One panel edge (base.css `--line-panel`): every panel-like surface (a bg.panel box with the panel radius and a
// border) draws its outer edge with it, so no panel's edge is darker or lighter than its neighbours' (round 20: the
// Dataset Stats KPI strip). A strip that paints its 1 px dividers with a line-coloured background clips it to the
// padding box, or the translucent edge composites over it.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? files(p) : /\.(svelte|css)$/.test(p) ? [p] : [];
  });
}

/** Every rule block (`selector { declarations }`) without nested blocks, comments stripped. */
function rules(src: string): { sel: string; body: string }[] {
  const css = (src.includes('<style>') ? src.split('<style>')[1]!.split('</style>')[0]! : src).replace(/\/\*[\s\S]*?\*\//g, '');
  return [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ sel: m[1]!.trim(), body: m[2]! }));
}

describe('panel edges', () => {
  const all = files(SRC).flatMap((f) => rules(readFileSync(f, 'utf8')).map((r) => ({ ...r, file: relative(SRC, f) })));

  it('use --line-panel on every panel-like surface', () => {
    const off = all
      .filter((r) => /background: var\(--bg-panel\)/.test(r.body) && /border-radius: var\(--radius-lg\)/.test(r.body))
      .filter((r) => /border: 1px solid var\(--line-(?!panel)/.test(r.body))
      .map((r) => `${r.file}: ${r.sel}`);
    expect(off).toEqual([]);
  });

  it('never theme-switch a panel edge by hand', () => {
    const off = all.filter((r) => /\[data-theme=/.test(r.sel) && /^\s*border-color: var\(--line-(default|subtle)\);\s*$/.test(r.body)).map((r) => `${r.file}: ${r.sel}`);
    expect(off).toEqual([]);
  });

  it('clip a line-coloured divider background inside the edge', () => {
    const off = all
      .filter((r) => /background: var\(--line-/.test(r.body) && /border: 1px solid/.test(r.body))
      .filter((r) => !/background-clip: padding-box/.test(r.body))
      .map((r) => `${r.file}: ${r.sel}`);
    expect(off).toEqual([]);
  });
});
