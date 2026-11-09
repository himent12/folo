// Series hues in text take their text ink (DESIGN_DIRECTION §2.5): `--chart-N` and `--chart-fill-N` are marks (3:1),
// `--chart-text-N` is held to 4.5:1 by design/contrast.mjs. Round 23 found Playground's JSON (amber, teal and magenta
// at 3.7-4.0:1 on the inset) and Compare's and Loss's end labels (sky at 3.55:1 on white) in line ink.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';
import { seriesText } from './ink';

const SRC = join(__dirname, '..', '..');
const files = (dir: string): string[] =>
  readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    if (statSync(p).isDirectory()) return files(p);
    return /\.(svelte|css)$/.test(n) ? [p] : [];
  });

describe('series text ink', () => {
  it('maps a series line or fill to its text ink and keeps anything else', () => {
    expect(seriesText('var(--chart-1)')).toBe('var(--chart-text-1)');
    expect(seriesText('var(--chart-fill-4)')).toBe('var(--chart-text-4)');
    expect(seriesText('var(--text-primary)')).toBe('var(--text-primary)');
    expect(seriesText('var(--chart-context)')).toBe('var(--chart-context)');
  });

  it('no text is coloured with a series line or fill token', () => {
    const bad: string[] = [];
    for (const f of files(SRC)) {
      if (f.endsWith('tokens.css')) continue;
      readFileSync(f, 'utf-8')
        .split('\n')
        .forEach((line, i) => {
          // CSS `color:` (text) and Svelte `style:color`, in a series' line or fill ink.
          if (/(^|[\s;{])color:\s*var\(--chart-(fill-)?[1-6]\)|style:color=\{?['"`]?var\(--chart-(fill-)?[1-6]\)/.test(line))
            bad.push(`${relative(SRC, f)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(bad).toEqual([]);
  });
});
