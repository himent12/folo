/// <reference types="node" />
// No handler is a lint escape (DESIGN_DIRECTION §2.9): an element that reacts to the pointer either has real keys (a
// grid's arrows, a combobox's activedescendant) or says, in a `svelte-ignore` with its reason, where the keys live.
// An empty `onkeydown={() => {}}` hides that question instead of answering it. Motion signals a change and never
// decorates (§2.9): an animation runs forever only where its element exists for exactly as long as the thing it shows
// (a spinner, a skeleton, a live run's heartbeat, the Playground's scan while it detects); every other loop is capped.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function files(dir: string, ext = '.svelte'): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? files(p, ext) : p.endsWith(ext) ? [p] : [];
  });
}

/** The loops that end with what they show: mounted only while it lasts. */
const ENDLESS = new Set([
  'lib/ui/Spinner.svelte',
  'lib/ui/Skeleton.svelte',
  'lib/ui/StatusBadge.svelte',
  'lib/shell/SidebarLive.svelte',
  // A live run's end dot: its heartbeat is drawn only while the run trains (r26-t).
  'lib/charts/LineChart.svelte',
  'routes/Playground.svelte',
  // A labeling tile's live dot: drawn only while the agent has that image.
  'routes/label/LabelTile.svelte',
]);

describe('event handlers', () => {
  it('are never empty (a no-op key handler is a lint escape, not keyboard support)', () => {
    const hits: string[] = [];
    for (const file of files(SRC)) {
      readFileSync(file, 'utf8')
        .split('\n')
        .forEach((line, i) => {
          if (/\bon\w+=\{\s*\(\w*\)\s*=>\s*\{\s*\}\s*\}/.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });

  it('loop forever only in the parts mounted exactly while their state lasts; loading shimmers elsewhere are capped', () => {
    const hits: string[] = [];
    for (const file of [...files(SRC), ...files(SRC, '.css')]) {
      const rel = relative(SRC, file).replaceAll('\\', '/');
      if (ENDLESS.has(rel)) continue;
      readFileSync(file, 'utf8')
        .split('\n')
        .forEach((line, i) => {
          if (/animation(-iteration-count)?:[^;]*\binfinite\b/.test(line)) hits.push(`${rel}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });
});
