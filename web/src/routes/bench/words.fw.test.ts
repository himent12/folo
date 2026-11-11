/// <reference types="node" />
// words.fw.ts is words.ts for the fw app (no StatusBadge.svelte import, label's words.fw): the same code line for line
// apart from those, so the tests of words.ts (words, loops, reading, sentences) hold for it too. This keeps them in step.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import * as legacy from './words';
import * as twin from './words.fw';

const read = (f: string) => readFileSync(resolve(process.cwd(), 'src/routes/bench', f), 'utf8').replace(/\r\n/g, '\n');

describe('words.fw', () => {
  it('is words.ts but for its imports of Svelte-side modules', () => {
    const fw = read('words.fw.ts')
      // Its header, the restated OwnStatus and the type it needs.
      .replace(/^\/\/ fw twin of words\.ts[^]*?Keep the two in step\.\n/, '')
      .replace('  RunStatus,\n', '')
      .replace(/\n\/\*\* StatusBadge's `OwnStatus`[^\n]*\ninterface OwnStatus \{\n {2}status: RunStatus;\n {2}label: string;\n\}\n/, '')
      .replace("from '../label/words.fw';", "from '../label/words';");
    const old = read('words.ts').replace("import type { OwnStatus } from '$lib/ui/StatusBadge.svelte';\n", '');
    expect(fw).toBe(old);
  });

  it('exports the same names', () => {
    expect(Object.keys(twin).sort()).toEqual(Object.keys(legacy).sort());
  });
});
