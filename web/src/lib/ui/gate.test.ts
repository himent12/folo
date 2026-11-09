/// <reference types="node" />
// Prerequisites are visible (DESIGN_DIRECTION §2.8): `ui/Gate` gives a disabled control its reason as a tooltip and an
// accessible description, which a touch screen and a keyboard on a disabled button never show. So a reason the person
// can act on ("Select 2–4 runs to compare", "Enter a folder path") is also said in words on screen, in an element
// marked `data-gate-why` beside or under the control. "Server offline" is the one exception: the page's error state and
// the connection status say it already.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return files(p);
    return p.endsWith('.svelte') ? [p] : [];
  });
}

/** A reason that is only ever the offline one: `x ? OFFLINE_REASON : null`. */
const OFFLINE_ONLY = /^[\w.!]+\s*\?\s*OFFLINE_REASON\s*:\s*null$/;

it('says every reason a person can act on in words, not only in a disabled control\'s tooltip', () => {
  const hits: string[] = [];
  for (const file of files(SRC)) {
    const src = readFileSync(file, 'utf8').replace(/<!--[\s\S]*?-->/g, '');
    for (const m of src.matchAll(/<Gate\s+(?:reason=\{([^}]*)\}|\{reason\})/g)) {
      const expr = (m[1] ?? 'reason').trim();
      if (OFFLINE_ONLY.test(expr)) continue;
      if (!/data-gate-why/.test(src)) hits.push(`${relative(SRC, file)}: <Gate reason={${expr}}> has no visible reason (data-gate-why)`);
    }
  }
  expect(hits).toEqual([]);
});
