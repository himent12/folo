import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LabelEventMap, LabelJob, StartLabeling } from '../types';
import { MockApi } from './index';
import { BATCH_MS } from './label';

const NOW = Date.parse('2026-11-09T12:00:00Z');

beforeEach(() => {
  vi.useFakeTimers({ now: NOW });
});
afterEach(() => vi.useRealTimers());

const balanced = (j: LabelJob) => j.counts.pending + j.counts.labeled + j.counts.failed === j.counts.images;

describe('MockApi agent labeling: seeded jobs', () => {
  it('has a fresh, a running, a labeled and a failed job, newest first', async () => {
    const jobs = await new MockApi('normal', 0).labelJobs();
    expect(jobs.map((j) => j.name)).toEqual(['warehouse-bins', 'shapes-sorting', 'fasteners-review', 'parts-bench']);
    const [fresh, live, done, failed] = jobs as [LabelJob, LabelJob, LabelJob, LabelJob];
    expect(fresh.run).toBeNull();
    expect(fresh.counts).toMatchObject({ images: 40, pending: 40, labeled: 0 });
    expect(live.status).toBe('running');
    expect(live.run?.done).toBeGreaterThan(0);
    expect(live.run!.done).toBeLessThan(live.run!.total);
    expect(done.status).toBe('idle');
    expect(done.counts).toMatchObject({ pending: 0, failed: 0, labeled: done.counts.images });
    expect(done.usage.calls).toBeGreaterThan(done.run!.usage.calls);
    expect(failed.run).toMatchObject({ outcome: 'failed', done: 0 });
    expect(failed.run?.error).toMatch(/not logged in/i);
    expect(failed.run?.hint).toBeTruthy();
    for (const j of jobs) expect(balanced(j)).toBe(true);
  });

  it('marks images corrected by hand, a background image among them', async () => {
    const api = new MockApi('normal', 0);
    const page = await api.labelImages('lb_51d0', { limit: 1000 });
    const manual = page.images.filter((i) => i.source === 'manual');
    expect(manual.length).toBeGreaterThan(2);
    expect(manual.some((i) => i.boxes.length === 0)).toBe(true);
    expect((await api.labelImages('lb_51d0', { status: 'labeled' })).total).toBe(page.total);
  });

  it('has none in the empty scenario', async () => {
    expect(await new MockApi('empty', 0).labelJobs()).toEqual([]);
  });
});

describe('MockApi agent labeling: a run', () => {
  it('labels a batch every 1.5 s, saying so on the stream, and adds up the usage', async () => {
    const api = new MockApi('normal', 0);
    const seen: { [K in keyof LabelEventMap]?: LabelEventMap[K][] } = {};
    const push = <K extends keyof LabelEventMap>(k: K) => (d: LabelEventMap[K]) => void ((seen[k] ??= []) as LabelEventMap[K][]).push(d);
    const close = api.labelEvents('lb_7c21', { job: push('job'), image: push('image'), log: push('log') });
    const started = await api.startLabeling('lb_7c21', { agent: 'claude' });
    expect(started.status).toBe('running');
    expect(started.run).toMatchObject({ agent: 'claude', model: 'haiku', effort: 'medium', total: 40, done: 0 });
    // Three calls of four images go out at once.
    expect((await api.labelImages('lb_7c21', { status: 'labeling' })).total).toBe(12);

    await vi.advanceTimersByTimeAsync(BATCH_MS + 250);
    const after = await api.labelJob('lb_7c21');
    expect(after.run!.done + after.run!.failed).toBe(4);
    expect(after.run!.usage.calls).toBe(1);
    expect(after.run!.usage.input_tokens).toBeGreaterThan(4000);
    expect(after.run!.usage.cost_usd).toBeGreaterThan(0);
    expect(balanced(after)).toBe(true);
    const first = seen.image!.filter((i) => i.id === '1').map((i) => i.status);
    expect(first[0]).toBe('labeling');
    expect(['labeled', 'failed']).toContain(first.at(-1));
    // The server's lines: the run's plan, then each call's.
    expect(seen.log![0]?.message).toBe('Claude Code (haiku, medium effort) starts on 40 images: 10 calls of up to 4');
    expect(seen.log!.map((l) => l.message)).toContainEqual(expect.stringMatching(/^call 1: \d+ images? labeled, \d+ box(es)? · \d+ tokens in, \d+ out$/));

    await vi.advanceTimersByTimeAsync(10 * BATCH_MS);
    const end = await api.labelJob('lb_7c21');
    expect(end.status).toBe('idle');
    expect(end.run).toMatchObject({ outcome: 'completed', total: 40 });
    expect(end.run!.done + end.run!.failed).toBe(40);
    expect(end.run!.failed).toBeGreaterThan(0);
    expect(end.counts.boxes).toBeGreaterThan(40);
    expect(seen.job!.at(-1)?.status).toBe('idle');
    const messages = seen.log!.map((l) => l.message);
    expect(messages).toContainEqual(expect.stringMatching(/^call \d+: the agent's reply left this image out$/));
    expect(messages.at(-1)).toMatch(/^Claude Code is done: \d+ of 40 images labeled, \d+ failed · 10 calls, \d+ tokens in, \d+ out$/);
    close();
  });

  it('stops: the calls in flight end and their images keep the state they had', async () => {
    const api = new MockApi('normal', 0);
    await api.startLabeling('lb_7c21', { agent: 'claude' });
    await vi.advanceTimersByTimeAsync(BATCH_MS);
    expect((await api.stopLabeling('lb_7c21')).status).toBe('stopping');
    await vi.advanceTimersByTimeAsync(1000);
    const j = await api.labelJob('lb_7c21');
    expect(j).toMatchObject({ status: 'idle', run: { outcome: 'stopped' } });
    const lines = (await api.labelLogs('lb_7c21')).lines.map((l) => l.message);
    expect(lines).toContain('stop requested');
    expect(lines.at(-1)).toMatch(/^Claude Code was stopped: \d+ of 40 images labeled, \d+ failed · \d+ calls?, \d+ tokens in, \d+ out$/);
    expect((await api.labelImages('lb_7c21', { status: 'labeling' })).total).toBe(0);
    expect(j.counts.pending).toBe(40 - j.run!.done - j.run!.failed);
    await expect(api.stopLabeling('lb_7c21')).rejects.toMatchObject({ status: 409, code: 'label_job_not_active' });
  });

  it('refuses Codex where it is not installed, with its hint, and runs it where it is', async () => {
    await expect(new MockApi('normal', 0).startLabeling('lb_7c21', { agent: 'codex' })).rejects.toMatchObject({ status: 501, code: 'not_supported' });
    const agents = await new MockApi('normal', 0).labelAgents();
    expect(agents.map((a) => [a.id, a.available, !!a.hint])).toEqual([
      ['claude', true, false],
      ['codex', false, true],
    ]);
    const j = await new MockApi('all-agents', 0).startLabeling('lb_7c21', { agent: 'codex' });
    expect(j.run).toMatchObject({ agent: 'codex', model: null, effort: 'low' });
    expect(agents.map((a) => [a.default_model, a.default_effort])).toEqual([
      ['haiku', 'medium'],
      [null, 'low'],
    ]);
  });

  it('answers 409 to a second start and 400 when nothing is left to label', async () => {
    const api = new MockApi('normal', 0);
    await expect(api.startLabeling('lb_3f9a', { agent: 'claude' })).rejects.toMatchObject({ status: 409, code: 'label_job_active' });
    await expect(api.startLabeling('lb_51d0', { agent: 'claude' })).rejects.toMatchObject({ status: 400 });
    expect((await api.startLabeling('lb_51d0', { agent: 'claude', image_ids: ['3'], note: 'Tighter boxes.' })).run?.total).toBe(1);
  });

  it("refuses what the server refuses, in its words, before it looks at the agent or the job", async () => {
    const api = new MockApi('normal', 0);
    const refused = async (body: Partial<StartLabeling>, message: string) =>
      expect(api.startLabeling('lb_7c21', { agent: 'claude', ...body })).rejects.toMatchObject({ status: 400, code: 'invalid_body', message });
    await refused({ batch: 0 }, 'batch must be in 1..=16');
    await refused({ batch: 17 }, 'batch must be in 1..=16');
    await refused({ batch: 2.5 }, 'batch must be in 1..=16');
    await refused({ parallel: 0 }, 'parallel must be in 1..=8');
    await refused({ parallel: 9 }, 'parallel must be in 1..=8');
    const model = 'model must be at most 100 characters of A-Z a-z 0-9 . _ : / - [ ]';
    await refused({ model: 'claude sonnet' }, model);
    await refused({ model: 'a'.repeat(101) }, model);
    await refused({ note: 'n'.repeat(2001) }, 'note must be at most 2000 characters');
    await refused({ image_ids: ['1', 'nope'] }, "image_ids: the job has no image 'nope'");
    // Codex is not installed here: a bad field is still said first, as the server checks the body before the agent.
    await expect(api.startLabeling('lb_7c21', { agent: 'codex', batch: 0 })).rejects.toMatchObject({ status: 400 });
    // At the limits it starts; a model name with the characters the CLIs use is taken.
    const j = await api.startLabeling('lb_7c21', { agent: 'claude', model: 'claude-sonnet-4.5[1m]', batch: 16, parallel: 8, note: 'n'.repeat(2000) });
    expect(j.run).toMatchObject({ model: 'claude-sonnet-4.5[1m]', total: 40 });
  });
});

describe('MockApi agent labeling: edits and datasets', () => {
  it('saves hand-drawn boxes clipped to the image, but not on an image the agent has', async () => {
    const api = new MockApi('normal', 0);
    const img = (await api.labelImages('lb_7c21', { limit: 1 })).images[0]!;
    const saved = await api.updateLabelImage('lb_7c21', img.id, [{ class: 1, x1: -20, y1: 10, x2: 50, y2: img.height + 40 }]);
    expect(saved).toMatchObject({ status: 'labeled', source: 'manual', boxes: [{ class: 1, x1: 0, y1: 10, x2: 50, y2: img.height }] });
    await expect(api.updateLabelImage('lb_7c21', img.id, [{ class: 9, x1: 0, y1: 0, x2: 9, y2: 9 }])).rejects.toMatchObject({ status: 400 });
    const busy = (await api.labelImages('lb_3f9a', { status: 'labeling', limit: 1 })).images[0]!;
    await expect(api.updateLabelImage('lb_3f9a', busy.id, [])).rejects.toMatchObject({ status: 409, code: 'label_job_active' });
  });

  it('takes an image that waits for its call out of the run when it is labeled by hand', async () => {
    const api = new MockApi('normal', 0);
    const before = (await api.labelJob('lb_3f9a')).run!;
    const waiting = (await api.labelImages('lb_3f9a', { status: 'pending' })).images.at(-1)!;
    const saved = await api.updateLabelImage('lb_3f9a', waiting.id, []);
    expect(saved).toMatchObject({ status: 'labeled', source: 'manual' });
    expect((await api.labelJob('lb_3f9a')).run!.total).toBe(before.total - 1);
    // The run goes on without it: the agent never has it, and its labels stay the hand-drawn ones.
    vi.advanceTimersByTime(10 * 60_000);
    const job = await api.labelJob('lb_3f9a');
    expect(job.run).toMatchObject({ outcome: 'completed', total: before.total - 1 });
    expect(job.run!.done + job.run!.failed).toBe(before.total - 1);
    const after = (await api.labelImages('lb_3f9a', { limit: 1000 })).images.find((i) => i.id === waiting.id)!;
    expect(after).toMatchObject({ status: 'labeled', source: 'manual', boxes: [] });
  });

  it('lets classes be renamed or added once there are boxes, never removed', async () => {
    const api = new MockApi('normal', 0);
    await expect(api.updateLabelJob('lb_51d0', { classes: ['bolt', 'nut'] })).rejects.toMatchObject({ status: 400 });
    const j = await api.updateLabelJob('lb_51d0', { classes: ['bolt', 'nut', 'washer', 'wood screw', 'rivet'] });
    expect(j.classes).toHaveLength(5);
    await expect(api.updateLabelJob('lb_3f9a', { name: 'x' })).rejects.toMatchObject({ status: 409 });
  });

  it('makes a dataset the Datasets list then shows, named after the job', async () => {
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', { val_fraction: 0.2 });
    expect(ds.name).toBe('fasteners-review');
    expect(ds.classes).toEqual(['bolt', 'nut', 'washer', 'screw']);
    expect(ds.splits.train + ds.splits.val).toBe(36);
    expect(ds.splits.val).toBe(7);
    expect((await api.datasets()).map((d) => d.id)).toContain(ds.id);
    expect((await api.labelJob('lb_51d0')).dataset_id).toBe(ds.id);
    expect((await api.createLabelDataset('lb_51d0', {})).name).toBe('fasteners-review-2');
    await expect(api.createLabelDataset('lb_7c21', {})).rejects.toMatchObject({ status: 400, code: 'invalid_dataset' });
    await expect(api.createLabelDataset('lb_3f9a', {})).rejects.toMatchObject({ status: 409 });
  });

  it('adds a folder the demo knows, skipping what the job has, and refuses one it does not', async () => {
    const api = new MockApi('normal', 0);
    const job = await api.createLabelJob({ name: 'mini', classes: ['bolt', 'nut'] });
    const first = await api.importLabelImages(job.id, 'D:\\data\\fasteners-mini\\');
    expect(first.added).toHaveLength(24);
    expect((await api.importLabelImages(job.id, 'D:/data/fasteners-mini')).skipped).toHaveLength(24);
    await expect(api.importLabelImages(job.id, 'E:/nowhere')).rejects.toMatchObject({ status: 404, code: 'path_not_found' });
    // More than 100 at once: the stream says reload instead of an image event each.
    const said: string[] = [];
    const close = api.labelEvents(job.id, { image: () => said.push('image'), reload: () => said.push('reload') });
    expect((await api.importLabelImages(job.id, 'D:/data/shapes')).added).toHaveLength(1000);
    expect(said).toEqual(['reload']);
    close();
    await expect(api.createLabelJob({ name: 'x', classes: ['a', 'A'] })).rejects.toMatchObject({ status: 400 });
  });
});

describe('MockApi agent labeling: uploads', () => {
  beforeEach(() => vi.stubGlobal('createImageBitmap', async () => ({ width: 64, height: 48, close() {} })));
  afterEach(() => vi.unstubAllGlobals());
  const file = (name: string, bytes: string, lastModified = 1) => new File([bytes], name, { type: 'image/jpeg', lastModified });

  it('knows an image by its bytes, as the server does, whatever its name or date', async () => {
    const api = new MockApi('normal', 0);
    const job = await api.createLabelJob({ name: 'up', classes: ['a'] });
    expect((await api.addLabelImages(job.id, [file('a.jpg', 'AAAA')])).added.map((i) => i.file)).toEqual(['a.jpg']);
    const again = await api.addLabelImages(job.id, [file('copy.jpg', 'AAAA', 99), file('b.jpg', 'BBBB'), file('b-copy.jpg', 'BBBB')]);
    expect(again.added.map((i) => i.file)).toEqual(['b.jpg']);
    expect(again.skipped).toEqual([
      { file: 'copy.jpg', reason: 'the job already has this image (a.jpg)' },
      { file: 'b-copy.jpg', reason: 'the job already has this image (b.jpg)' },
    ]);
  });

  it('adds other bytes under a name the job has as a name of their own', async () => {
    const api = new MockApi('normal', 0);
    const job = await api.createLabelJob({ name: 'up', classes: ['a'] });
    // The same name, size and date: a file name is not what makes an image the same one.
    const r = await api.addLabelImages(job.id, [file('a.jpg', 'AAAA'), file('a.jpg', 'ZZZZ')]);
    expect(r.added.map((i) => i.file)).toEqual(['a.jpg', 'a-2.jpg']);
    expect(r.skipped).toEqual([]);
  });
});

describe('MockApi agent labeling: servers without it', () => {
  it('folo serve --mock lists no agent_labeling and answers 501', async () => {
    const api = new MockApi('no-labeling', 0);
    expect((await api.system()).features).not.toContain('agent_labeling');
    await expect(api.labelJobs()).rejects.toMatchObject({ status: 501, code: 'not_implemented' });
    expect((await new MockApi('normal', 0).system()).features).toContain('agent_labeling');
  });
});
