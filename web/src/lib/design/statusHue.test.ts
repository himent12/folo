/// <reference types="node" />
// Status colour marks status; lime is for the one action (DESIGN_DIRECTION §2.6). Round 22 broke it four ways: a KPI's
// number in danger red, a toast's "Open run" as a lime link, demo data with the warning dot, and the meter and the line
// of one out-of-memory forecast in two hues. These scans keep the first three out of every component's styles; the
// fourth is RunNew.svelte.test.ts's. Round 23 found two more: Compare's best value as lime text in a lime chip ("0.604
// best"), and New run's Summary showing a device it cannot find ("gpu:1") in red; and an inline action ("Lower the
// threshold") as a lime link.
// Round 24 found three more: New run's switches solid lime when on (a toggle competing with Start training), a
// callout's "Too short to learn:" and "Why?" in amber text (weak in light), and "Ran out of GPU memory" and
// "Prediction failed" as red titles. A status shows through its icon and its callout's tint; titles and links are ink.
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

  it('never colours a value: a table value, a winner or a summary value stays in ink, its status a mark beside it', () => {
    // Value slots: a table's number cell (`.num`), a winner, New run's summary value (`.rv-v`), a `.v` / `-v` readout.
    const value = /(?:^|[\s.,])(?:num|winner|best|rv-v|v|val|[\w-]+-v)(?![\w-])/;
    expect(scan((s) => value.test(s) && !/:global\(svg\)|>\s*:global|\.(?:sw|dot|mark|glyph|icon)\b/.test(s))).toEqual([]);
  });

  it('marks a best value with a word, never an accent chip: an accent tint is a badge, not a tag on a value', () => {
    // A success tint behind a status glyph (Export's check) is a mark; lime behind text is what this rules out.
    const tint = /background:\s*var\(--accent-muted\)/;
    const hits: string[] = [];
    for (const file of svelteFiles(SRC)) {
      if (/[\\/]ui[\\/](?:Badge|StatusBadge)\.svelte$/.test(file)) continue;
      for (const r of rules(readFileSync(file, 'utf8'))) if (tint.test(r.body)) hits.push(`${relative(SRC, file)}: ${r.selector}`);
    }
    expect(hits).toEqual([]);
  });

  it('never colours an inline link or action in lime: links are ink', () => {
    const lime = /(?:^|;|\s)color:\s*var\(--accent-(?:text|solid|line)\)/;
    expect(scan((s) => /\b(?:inline|link|more|go)\b|(?:^|\s)a\b/.test(s), lime)).toEqual([]);
  });

  it('never sets a title or a link in a status hue: the status shows through the icon and the callout tint', () => {
    const status = /(?:^|;|\s)color:\s*var\(--(?:danger|warning|success|info|accent)-(?:text|solid|line)\)/;
    const words = /(?:^|[\s.,])(?:title|c-title|n-title|heading|more|link|why|dock-warn)(?![\w-])|(?:^|\s)a\b/;
    // Each selector of a list on its own: ".danger > :global(svg), .danger .c-title" is a mark and a title.
    const part = (p: string) => words.test(p) && !/:global\(svg\)|\.(?:icon|mark|glyph|sw|dot)\b/.test(p);
    expect(scan((s) => s.split(',').some(part), status)).toEqual([]);
  });

  it("never colours a control's state in lime: a switch, checkbox or pressed toggle that is on is ink", () => {
    const lime = /(?:background(?:-color)?|accent-color|border-color):\s*var\(--accent-(?:solid|line|text)\)/;
    expect(scan((s) => /:checked|aria-(?:checked|pressed)=.true.|\.on\b/.test(s), lime)).toEqual([]);
  });
});
