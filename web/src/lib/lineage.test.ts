/// <reference types="node" />
import { describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import { origin, originTip } from './lineage';

const model = (id: string, name: string) => ({ id, name }) as Model;
const models = [model('m_coco_s', 'folo-s-coco80'), model('m_7d1e', 'fasteners-s-best')];

describe('origin', () => {
  it('is scratch without weights', () => {
    expect(origin(null, models)).toEqual({ kind: 'scratch', label: 'scratch' });
    expect(origin('  ', models).kind).toBe('scratch');
  });

  it('resolves a model id to its name', () => {
    expect(origin('m_coco_s', models)).toEqual({ kind: 'model', id: 'm_coco_s', label: 'folo-s-coco80' });
    expect(origin('m_7d1e', new Map(models.map((m) => [m.id, m]))).label).toBe('fasteners-s-best');
  });

  it('names a weights file by its base name', () => {
    expect(origin('/data/weights/folo-l-coco.folo', models)).toEqual({ kind: 'file', path: '/data/weights/folo-l-coco.folo', label: 'folo-l-coco.folo' });
    expect(origin('C:\\w\\last.folo', models).label).toBe('last.folo');
    expect(origin('m_shapes_n').kind).toBe('model'); // no list: still an id
  });

  it('keeps an unknown id (the models list may not have loaded yet)', () => {
    expect(origin('m_gone', [])).toEqual({ kind: 'model', id: 'm_gone', label: 'm_gone' });
  });

  it('puts the id or path in the tooltip beside the name', () => {
    expect(originTip(origin('m_coco_s', models))).toBe('Model m_coco_s');
    expect(originTip(origin('m_gone', []))).toBeUndefined();
    expect(originTip(origin('/w/last.folo'))).toBe('/w/last.folo');
    expect(originTip(origin(null))).toBeUndefined();
  });
});

describe('one wording for "from scratch"', () => {
  it('is a label spelled only in lib/lineage (FROM_SCRATCH as a value, FROM_SCRATCH_META in a meta line)', async () => {
    const { readdirSync, readFileSync, statSync } = await import('node:fs');
    const { join, relative, resolve } = await import('node:path');
    const SRC = resolve(process.cwd(), 'src');
    const walk = (dir: string): string[] =>
      readdirSync(dir).flatMap((f) => {
        const p = join(dir, f);
        if (statSync(p).isDirectory()) return p.endsWith('mock') ? [] : walk(p);
        return /\.(svelte|ts)$/.test(p) && !/\.test\.ts$/.test(p) && !p.endsWith('lineage.ts') ? [p] : [];
      });
    const hits: string[] = [];
    for (const file of walk(SRC)) {
      const src = readFileSync(file, 'utf8')
        .replace(/<!--[\s\S]*?-->/g, '')
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/(^|[^:'"`\w])\/\/.*$/gm, '$1');
      src.split('\n').forEach((line, i) => {
        // The label on its own (a string, or markup text alone between tags or blocks); prose that says it in a sentence
        // ("another size starts from scratch") is not the label.
        if (/(['"`])[Ff]rom scratch\1|[>}]\s*[Ff]rom scratch\s*[<{]/.test(line)) hits.push(`${relative(SRC, file)}:${i + 1}: ${line.trim()}`);
      });
    }
    expect(hits).toEqual([]);
  });
});
