/// <reference types="node" />
// Copy is for users (DESIGN_DIRECTION §2.3, "Copy" and "Measure"): what Studio prints never names a developer document or a source
// path ("docs/BENCHMARKS.md's figures"), and a space at the edge of a `{#if}` block is never lost (Svelte trims the
// whitespace a block's content starts or ends with, so "served from{#if origin} (" rendered "served from(").
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function files(dir: string, ext: RegExp): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return p.endsWith('mock') ? [] : files(p, ext);
    return ext.test(p) && !/\.test\.ts$/.test(p) ? [p] : [];
  });
}

/** Source without comments (markup, script and style), so the scan reads only what can reach the screen. */
function uncommented(src: string): string {
  return src
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:'"`\w])\/\/.*$/gm, '$1');
}

/** A developer document or a source path: `docs/API.md`, `design/README.md`, `crates/…`, `src/lib/…`, `lib/oom.ts`. */
const DEV_PATH = /\b(?:docs|design|crates|src)\/[\w./-]+|\b[A-Z][A-Z_]+\.md\b|\blib\/\w+\.ts\b/;

describe('copy', () => {
  it('never names a developer document or a source path', () => {
    const hits: string[] = [];
    for (const file of files(SRC, /\.(svelte|ts)$/)) {
      uncommented(readFileSync(file, 'utf8'))
        .split('\n')
        .forEach((line, i) => {
          // Imports name modules, not copy.
          if (/^\s*(import|export \* from|export \{[^}]*\} from)\b/.test(line)) return;
          if (DEV_PATH.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });

  it('caps explanatory text at the one measure, never a ch width of its own', () => {
    // `--measure` (base.css) is ~60 characters in Geist; a local "60ch" ran ~90 (a ch is Geist's wide tabular 0).
    const hits: string[] = [];
    for (const file of files(SRC, /\.svelte$/)) {
      readFileSync(file, 'utf8')
        .split('\n')
        .forEach((line, i) => {
          if (/max-(?:width|inline-size):[^;]*\d+ch\b/.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });

  it("keeps the space at a block's edge as {' '}, where Svelte would trim it", () => {
    const hits: string[] = [];
    // Text that starts a block's content with a space, or ends it with one: `{#if x} (…`, `… and {/if}`.
    const lead = /(?:\{(?:#if|:else if)\s[^{}]*\}|\{:else\}) +[^\s<{]/;
    const trail = /[^\s>}] +\{(?:\/if|:else if\s[^{}]*|:else|\/each|\/key)\}/;
    for (const file of files(SRC, /\.svelte$/)) {
      readFileSync(file, 'utf8')
        .replace(/<script[\s\S]*?<\/script>/g, (m) => m.replace(/[^\n]/g, ''))
        .split('\n')
        .forEach((line, i) => {
          if (lead.test(line) || trail.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });

  it('writes a percentage one way, the number against its sign ("89%", `pct` in lib/format), never "89 %"', () => {
    const hits: string[] = [];
    for (const file of files(SRC, /\.(svelte|ts)$/)) {
      uncommented(readFileSync(file, 'utf8'))
        .split('\n')
        .forEach((line, i) => {
          // A figure or an interpolation, a space, then the sign, inside a string or as markup text.
          if (/(\d|\}) %(?![\w-])/.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });

  it('names the score it prints: "mAP50-95 0.604" or "mAP50 0.812", never a bare "mAP 0.604"', () => {
    // DESIGN_DIRECTION §2.23, one name per measure: the sidebar's live card, the palette, the Playground and New run's
    // Start from all printed "mAP 0.6…" beside screens that say mAP50-95 for the same figure. Prose about the metric
    // ("expect an mAP near 0") names no figure, so only "mAP" before a value is caught.
    const hits: string[] = [];
    for (const file of files(SRC, /\.(svelte|ts)$/)) {
      uncommented(readFileSync(file, 'utf8'))
        .split('\n')
        .forEach((line, i) => {
          // "mAP" then a space and a figure, an interpolation (`${`) or a Svelte expression (`{`).
          if (/\bmAP (?:\d|\$\{|\{)/.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(hits).toEqual([]);
  });
});
