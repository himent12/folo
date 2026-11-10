import { describe, expect, it } from 'vitest';
import type { Bench, BenchCatalog, BenchEntry } from '$lib/api/types';
import { catalog } from '$lib/api/mock/benchCatalog';
import { contestantFacts, costFacts, coverageFacts, entryStatus, loopStatus, progressFacts, scoreText, shortFingerprint, sourceKind } from './words';

const bench = { images: 360, classes: ['a', 'b', 'c', 'd'], plan: catalog().default_plan } as unknown as Pick<Bench, 'images' | 'classes' | 'plan'>;
const entry = (o: Partial<BenchEntry>) =>
  ({ id: 'e', name: 'x', images: 360, conditions: true, scored: true, missing_classes: [], status: 'done', score: { total: 60, partial: false, axes: [] }, ...o }) as BenchEntry;

describe('coverageFacts', () => {
  it('says nothing for an entry that ran everything', () => {
    expect(coverageFacts(entry({}), bench)).toEqual([]);
  });

  it('says what an agent entry left out, in the order the leaderboard reads them', () => {
    const agent = entry({ images: 48, conditions: false, scored: false, score: { total: 50, partial: true, axes: [] } });
    expect(coverageFacts(agent, bench)).toEqual(['48 of 360 images', 'unchanged only', 'no confidences']);
  });

  it('counts the conditions an external entry answered when the result says, else says some were left out', () => {
    const ext = entry({ score: { total: 50, partial: true, axes: [] } });
    expect(coverageFacts(ext, bench, 4)).toEqual(['4 of 12 conditions']);
    expect(coverageFacts(ext, bench)).toEqual(['some conditions left out']);
  });

  it('names the classes a model lacks, which is no partial', () => {
    expect(coverageFacts(entry({ missing_classes: ['c', 'd'] }), bench)).toEqual(['knows 2 of 4 classes']);
  });
});

describe('words', () => {
  it('writes a score with one decimal, a fingerprint short, a source by its kind', () => {
    expect(scoreText(71.924)).toBe('71.9');
    expect(shortFingerprint('9c41aa00bb11e2a0')).toBe('9c41…e2a0');
    expect(sourceKind({ kind: 'dataset', dataset_id: 'd', split: 'val' })).toBe('validation split');
    expect(sourceKind({ kind: 'dataset', dataset_id: 'd' })).toBe('validation split');
    expect(sourceKind({ kind: 'label_job', job_id: 'j' })).toBe('labeling job');
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

  it('gives entries and loops a status mark and a word', () => {
    expect(entryStatus({ status: 'done' })).toEqual({ status: 'completed', label: 'Done' });
    expect(entryStatus({ status: 'queued' }).label).toBe('Queued');
    expect(loopStatus({ status: 'running', phase: 'training' })).toEqual({ status: 'running', label: 'Training' });
    expect(loopStatus({ status: 'done', phase: null }).label).toBe('Done');
  });
});
