// The status pill is the title form (DESIGN_DIRECTION 6): only beside the name of the one run a screen is about. Every
// other status (tables, lists, Compare) is dot + word (`plain`), so one job never has two looks.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');
const TITLE_FORM = new Set(['routes/Run.svelte', 'routes/overview/LiveRunCard.svelte']);

function files(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files(p, out);
    else if (name.endsWith('.svelte')) out.push(p);
  }
  return out;
}

describe('status pills', () => {
  it('appear only as the title form; every other status is dot + word', () => {
    const hits: string[] = [];
    for (const f of files(SRC)) {
      const rel = relative(SRC, f).split(sep).join('/');
      for (const m of readFileSync(f, 'utf8').matchAll(/<StatusBadge\b[^>]*\/>/g)) {
        if (!/\bplain\b/.test(m[0]) && !TITLE_FORM.has(rel)) hits.push(`${rel}: ${m[0]}`);
      }
    }
    expect(hits).toEqual([]);
  });
});
