import { describe, expect, it } from 'vitest';
import type { Bench, BenchCatalog, BenchEntry } from '$lib/api/types';
import { catalog } from '$lib/api/mock/benchCatalog';
import { contestantFacts, costFacts, coverageFacts, entryStatus, progressFacts, shortFingerprint, sourceKind } from './words';

const bench = { images: 360, classes: ['a', 'b', 'c', 'd'], plan: catalog().default_plan } as unknown as Pick<Bench, 'images' | 'classes' | 'plan'>;
const entry = (o: Partial<BenchEntry>) =>
  ({ id: 'e', name: 'x', images: 360, conditions: true, scored: true, missing_classes: [], status: 'done', score: { total: 60, partial: false, axes: [] }, ...o }) as BenchEntry;

describe('coverageFacts', () => {
  it('says what an entry left out, in the order the leaderboard reads them, and nothing for one that ran everything', () => {
    const partial = { total: 50, partial: true, axes: [] };
    expect(coverageFacts(entry({}), bench)).toEqual([]);
    expect(coverageFacts(entry({ images: 48, conditions: false, scored: false, score: partial }), bench)).toEqual(['48 of 360 images', 'unchanged only', 'no confidences']);
    // An external entry: the conditions it answered when the result says, else that some were left out.
    expect(coverageFacts(entry({ score: partial }), bench, 4)).toEqual(['4 of 12 conditions']);
    expect(coverageFacts(entry({ score: partial }), bench)).toEqual(['some conditions left out']);
    // The classes a model lacks, which is no partial.
    expect(coverageFacts(entry({ missing_classes: ['c', 'd'] }), bench)).toEqual(['knows 2 of 4 classes']);
  });
});

describe('words', () => {
  it('writes a fingerprint short, and a dataset source without a split as its validation split', () => {
    expect(shortFingerprint('9c41aa00bb11e2a0')).toBe('9c41…e2a0');
    expect(sourceKind({ kind: 'dataset', dataset_id: 'd' })).toBe(sourceKind({ kind: 'dataset', dataset_id: 'd', split: 'val' }));
  });

  it('says who an entry is without repeating its name', () => {
    expect(contestantFacts(entry({ name: 'Claude Code · Sonnet', contestant: { kind: 'agent', agent: 'claude', model: 'sonnet', effort: 'medium' } }))).toEqual({ facts: ['Agent', 'medium effort'] });
    expect(contestantFacts(entry({ name: 'my agent', contestant: { kind: 'agent', agent: 'codex', model: null, effort: 'low' } }))).toEqual({ facts: ['Agent', 'Codex', 'Default model', 'low effort'] });
    expect(contestantFacts(entry({ contestant: { kind: 'model', model_id: 'm', arch: 'folo-s', params: 9_100_000, imgsz: 640 } }))).toEqual({ arch: 'folo-s', facts: ['9.1M params'] });
    expect(contestantFacts(entry({ contestant: { kind: 'external' }, scored: false })).facts).toEqual(['External', 'no confidences']);
  });

  it('words where a running entry is and what it cost', () => {
    const cat = catalog() as BenchCatalog;
    expect(progressFacts(entry({ status: 'running', progress: { done: 7196, total: 13320, condition: 'haze', severity: 3 } }), cat)).toEqual({ where: 'Haze, severity 3', pct: '54%' });
    expect(progressFacts(entry({ status: 'running', progress: { done: 10, total: 13320, condition: null, severity: null } }), cat).where).toBe('Unchanged images');
    expect(progressFacts(entry({ status: 'queued' }), cat)).toEqual({ where: 'Waiting for the GPU', pct: null });
    expect(costFacts(entry({ contestant: { kind: 'model', model_id: 'm', arch: 'folo-n', params: 1, imgsz: 640 }, speed: { device: 'gpu:0', ms_per_image: 2.35 } }))).toEqual({ value: '2.35 ms', sub: 'per image · gpu:0' });
    expect(costFacts(entry({ contestant: { kind: 'external' } }))).toBeNull();
    const codex = costFacts(entry({ contestant: { kind: 'agent', agent: 'codex', model: null, effort: 'low' }, usage: { calls: 6, input_tokens: 100_000, cached_input_tokens: 0, output_tokens: 16_000, cost_usd: null } }));
    expect(codex).toEqual({ value: '116k tokens', sub: '6 calls' });
  });

  it('words an open external entry as receiving predictions, never as a model running a condition', () => {
    const open = { status: 'running', contestant: { kind: 'external' }, progress: { done: 1200, total: 6000, condition: null, severity: null } } as unknown as BenchEntry;
    expect(progressFacts(open)).toEqual({ where: 'Receiving predictions', pct: '1,200 of 6,000 records' });
    expect(entryStatus(open)).toEqual({ status: 'running', label: 'Receiving' });
    const model = { ...open, contestant: { kind: 'model' }, progress: { done: 1200, total: 6000, condition: 'haze', severity: 3 } } as unknown as BenchEntry;
    expect(progressFacts(model).where).toBe('haze, severity 3');
    expect(entryStatus(model).label).toBe('Running');
  });
});
