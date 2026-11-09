/// <reference types="node" />
// Lime means the primary action, focus and live state, nothing else (DESIGN_DIRECTION §2.6). Navigation marks where
// you are with text.primary on bg.active: the sidebar, the icon rail, the phone tab bar, the palette's highlighted row
// and Settings' section rail never paint an item, its icon or its label in an accent token.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const NAV = ['lib/shell/Sidebar.svelte', 'lib/shell/TabBar.svelte', 'lib/shell/CommandPalette.svelte', 'lib/shell/Topbar.svelte'];

describe('navigation', () => {
  it('uses no accent colour', () => {
    const hits: string[] = [];
    for (const f of NAV) {
      readFileSync(resolve(process.cwd(), 'src', f), 'utf8')
        .split('\n')
        .forEach((l, i) => {
          if (/--accent-/.test(l)) hits.push(`${f}:${i + 1}: ${l.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });

  it("marks Settings' current section without the accent", () => {
    const css = readFileSync(resolve(process.cwd(), 'src/routes/Settings.svelte'), 'utf8').match(/\.toc a\.on \{[^}]*\}/)?.[0] ?? '';
    expect(css).toContain('--bg-active');
    expect(css).not.toContain('--accent');
  });

  it('marks a selected list row with the ink bar, never the accent (the shared selection rule in base.css)', () => {
    const base = readFileSync(resolve(process.cwd(), 'src/lib/design/base.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    const rule = base.match(/:where\(\.table\) tbody tr:is\(\.on, \[aria-selected='true'\]\) > :first-child \{[^}]*\}/)?.[0] ?? '';
    expect(rule).toContain('var(--select-bar)');
    expect(rule).not.toContain('--accent');
  });
});
