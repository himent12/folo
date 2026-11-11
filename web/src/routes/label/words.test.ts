import { describe, expect, it } from 'vitest';
import type { LabelAgent, LabelImage, LabelJob, LabelRun } from '$lib/api/types';
import { classChangeError, classesError } from '$lib/label';
import { askOutcome, imageState, jobStatus, modelLabel, runAgent, tokens, usageFacts } from './words';

const job = (status: LabelJob['status'], outcome: LabelRun['outcome'] = null) => ({ status, run: outcome ? { outcome } : null }) as LabelJob;
const claude = { id: 'claude', models: [{ id: 'sonnet', label: 'Sonnet' }] } as LabelAgent;

describe('usage', () => {
  it('says what the agent reported, one fact each', () => {
    expect(usageFacts({ calls: 4, input_tokens: 18_400, cached_input_tokens: 3_252, output_tokens: 5_100, cost_usd: 0.021 })).toEqual([
      '4 calls',
      '18.4k tokens in (3.3k cached)',
      '5.1k out',
      '$0.02',
    ]);
    expect(usageFacts({ calls: 1, input_tokens: 10_430, cached_input_tokens: 0, output_tokens: 312, cost_usd: null })).toEqual(['1 call', '10.4k tokens in', '312 out']);
    expect(usageFacts({ calls: 0, input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, cost_usd: null })).toEqual([]);
  });

  it('writes token counts short', () => {
    expect([940, 1000, 3252, 99_949, 126_400, 1_420_000].map(tokens)).toEqual(['940', '1.0k', '3.3k', '99.9k', '126k', '1.4M']);
  });
});

describe('jobStatus', () => {
  it('says what the agent is doing, and when its last run failed', () => {
    expect(jobStatus(job('running'))).toEqual({ status: 'running', label: 'Labeling' });
    expect(jobStatus(job('stopping'))).toEqual({ status: 'stopping', label: 'Stopping' });
    expect(jobStatus(job('idle', 'failed'))).toEqual({ status: 'failed', label: 'Agent failed' });
    expect(jobStatus(job('idle', 'completed')).label).toBe('Idle');
    expect(jobStatus(job('idle')).label).toBe('Idle');
  });
});

describe('agents and images', () => {
  it("names a model by the agent's label, else by its id, and the CLI's own default as such", () => {
    expect(modelLabel('sonnet', claude)).toBe('Sonnet');
    expect(modelLabel('claude-sonnet-5', claude)).toBe('claude-sonnet-5');
    expect(modelLabel(null, claude)).toBe('Default model');
    expect(runAgent({ agent: 'codex', model: null } as LabelRun, [claude])).toEqual(['Codex', 'Default model']);
  });

  it("says an image's state", () => {
    const img = (status: LabelImage['status'], boxes = 0, source: LabelImage['source'] = 'agent') =>
      ({ status, boxes: Array.from({ length: boxes }), source }) as LabelImage;
    expect(imageState(img('labeled', 3))).toBe('3 boxes, drawn by the agent');
    expect(imageState(img('labeled', 0, 'manual'))).toBe('No objects, edited by hand');
    expect(imageState(img('failed'))).toBe('The agent could not label it');
    expect(imageState({ ...img('failed'), error: 'the reply left this image out' })).toBe('The agent could not label it: the reply left this image out');
  });
});

describe('askOutcome', () => {
  const run = (outcome: LabelRun['outcome'], error: string | null = null, hint: string | null = null) => ({ agent: 'claude', outcome, error, hint }) as LabelRun;
  const img = (status: LabelImage['status'], boxes = 0, error: string | null = null) => ({ status, boxes: Array.from({ length: boxes }), source: 'agent', error }) as LabelImage;

  it("turns the editor's line into the run's error, the image's own error or its new boxes once the run ends", () => {
    expect(askOutcome(run('failed', '`claude` printed no result', 'Run `claude -p hi`.'), img('pending'))).toEqual({
      tone: 'error',
      title: 'Claude Code could not label it again',
      detail: '`claude` printed no result Run `claude -p hi`.',
    });
    expect(askOutcome(run('completed'), img('failed', 0, 'the reply left this image out'))).toMatchObject({ tone: 'error', detail: 'the reply left this image out' });
    expect(askOutcome(run('completed'), img('labeled', 2))).toEqual({ tone: 'success', title: 'New boxes', detail: '2 boxes, drawn by the agent' });
    expect(askOutcome(run('stopped'), img('labeled', 2))?.tone).toBe('info');
  });

  it('waits while the image is still with the agent', () => {
    expect(askOutcome(run('completed'), img('labeling'))).toBeNull();
  });
});

describe('class rules', () => {
  it('takes 1 to 80 distinct names without commas, colons or line breaks', () => {
    expect(classesError(['car', 'person'])).toBeNull();
    expect(classesError([])).toMatch(/at least one/);
    expect(classesError(['car', 'Car'])).toMatch(/twice/);
    expect(classesError(['a:b'])).toMatch(/colon/);
    expect(classesError(Array.from({ length: 81 }, (_, i) => `c${i}`))).toMatch(/80/);
  });

  it('lets a job with boxes rename or add classes, never remove one', () => {
    expect(classChangeError(['a', 'b'], ['a'], true)).toMatch(/not removed/);
    expect(classChangeError(['a', 'b'], ['x', 'b', 'c'], true)).toBeNull();
    expect(classChangeError(['a', 'b'], ['a'], false)).toBeNull();
  });
});
