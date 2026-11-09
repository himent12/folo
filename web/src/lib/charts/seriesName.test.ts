/// <reference types="node" />
// A series has one name (DESIGN_DIRECTION §2.23). Round 24's Training loss named its components three ways: the legend
// "Box / Classification / Distribution focal", the end labels "box / dfl / cls" and the phone legend "box / cls / dfl".
// A chart's legend, its direct labels and its phone legend all read a series' `label`; these scans keep a second name
// out: no direct label that differs from the legend's (`endLabel` is only ever `false`), and no label that changes
// with the viewport.
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

function scan(re: RegExp): string[] {
  const hits: string[] = [];
  for (const file of svelteFiles(SRC)) {
    if (file.endsWith('LineChart.svelte')) continue;
    readFileSync(file, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        if (re.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}`);
      });
  }
  return hits;
}

describe('one name per series', () => {
  it('never labels a line end with a name other than its legend label', () => {
    expect(scan(/\bendLabel:\s*(?!false\b)/)).toEqual([]);
  });

  it("never renames a series by the viewport (a phone legend reads the desktop legend's names)", () => {
    expect(scan(/\blabel:\s*(?:narrow|phone|compact)\s*\?/)).toEqual([]);
  });
});
