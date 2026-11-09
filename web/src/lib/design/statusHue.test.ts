/// <reference types="node" />
// Status colour marks status; lime is for the one action (DESIGN_DIRECTION §2.6). Round 22 broke it four ways: a KPI's
// number in danger red, a toast's "Open run" as a lime link, demo data with the warning dot, and the meter and the line
// of one out-of-memory forecast in two hues. These scans keep the first three out of every component's styles; the
// fourth is RunNew.svelte.test.ts's.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function svelteFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? svelteFiles(p) : p.endsWith('.svelte') ? [p] : [];
  });
}

/** Every CSS rule in a component's style block: its selector and its declarations, comments removed. */
function rules(src: string): { selector: string; body: string }[] {
  const css = (src.match(/<style[^>]*>([\s\S]*?)<\/style>/)?.[1] ?? '').replace(/\/\*[\s\S]*?\*\//g, '');
  return [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ selector: m[1]!.trim(), body: m[2]! }));
}

const HUE = /(?:^|;|\s)color:\s*var\(--(?:accent|danger|warning|success|info)-(?:text|solid|line)\)/;

function scan(match: (selector: string) => boolean, hue = HUE): string[] {
  const hits: string[] = [];
  for (const file of svelteFiles(SRC))
    for (const r of rules(readFileSync(file, 'utf8'))) if (match(r.selector) && hue.test(r.body)) hits.push(`${relative(SRC, file)}: ${r.selector}`);
  return hits;
}

describe('status hue', () => {
  it('never colours a metric: KPI and stat values stay in ink, the status is a mark beside its words', () => {
    expect(scan((s) => /(?:^|[\s.,])(?:k-value|value|kpi-v|stat-v)\b/.test(s) && !/:global\(svg\)|>\s*:global/.test(s))).toEqual([]);
  });

  it("never colours an action: a toast's action is ink, not lime and not its toast's hue", () => {
    expect(scan((s) => /\.action\b/.test(s))).toEqual([]);
  });

  it('treats demo data as a mode: no warning hue on it anywhere', () => {
    const warn = /var\(--warning-(?:solid|text|muted)\)/;
    const hits: string[] = [];
    for (const file of svelteFiles(SRC))
      for (const r of rules(readFileSync(file, 'utf8'))) if (/\.mock\b|data-mode='mock'|\.demo\b/.test(r.selector) && warn.test(r.body)) hits.push(`${relative(SRC, file)}: ${r.selector}`);
    expect(hits).toEqual([]);
  });
});
