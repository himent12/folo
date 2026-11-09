import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// A thumbnail that draws boxes shows the whole image (`fit="contain"`, letterboxed on bg.inset): `cover` crops it, and
// a box crossing the crop is cut in half at the edge, reading as a smaller object or a wrong label (DetectionView.svelte,
// web/DESIGN_DIRECTION.md §2.20). This scans every compact DatasetImage / DetectionView that is given boxes; the one
// deliberate crop carries a `crop-ok` comment just above its tag, saying why.
const src = fileURLToPath(new URL('../..', import.meta.url));

function* walk(dir: string): Generator<string> {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (name.endsWith('.svelte')) yield p;
  }
}

/**
 * Every `<DatasetImage … />` / `<DetectionView … />` tag (always self-closing; an arrow function in an attribute holds
 * a `>`) with its attributes and the text just before it.
 */
function tags(text: string) {
  const out: { attrs: string; before: string; line: number }[] = [];
  for (const m of text.matchAll(/<(?:DatasetImage|DetectionView)\b([\s\S]*?)\/>/g)) {
    const at = m.index!;
    out.push({ attrs: m[1]!, before: text.slice(Math.max(0, at - 240), at), line: text.slice(0, at).split('\n').length });
  }
  return out;
}

describe('thumbnails with boxes', () => {
  it('letterbox their image instead of cropping boxes at the edge', () => {
    const bad: string[] = [];
    for (const file of walk(src)) {
      for (const t of tags(readFileSync(file, 'utf8'))) {
        const compact = /(^|\s)compact(\s|$|=\{true\})/.test(t.attrs);
        const boxes = /\b(boxes|truth)=/.test(t.attrs);
        const cover = /fit="cover"/.test(t.attrs);
        if (compact && boxes && cover && !/crop-ok/.test(t.before)) bad.push(`${relative(src, file)}:${t.line}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it('finds the thumbnails it guards', () => {
    const all = [...walk(src)].flatMap((f) => tags(readFileSync(f, 'utf8')).filter((t) => /(^|\s)compact(\s|$)/.test(t.attrs) && /\bboxes=/.test(t.attrs)));
    // The dataset grid, the run's validation samples and the augmentation preview.
    expect(all.length).toBeGreaterThanOrEqual(3);
  });
});
