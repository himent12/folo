import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { image, specs, stats, total } from './datasets';
import { MockApi } from './index';
import { epochAt, progressAt, stepAt } from './runs';

const NOW = Date.parse('2026-11-08T14:30:00Z');

describe('mock datasets', () => {
  it('generates the same image for the same index', () => {
    const s = specs[0]!;
    expect(image(s, 42)).toEqual(image(s, 42));
    expect(image(s, 42)).not.toEqual(image(s, 43));
  });

  it('keeps boxes inside the image', () => {
    for (const s of specs) {
      for (let i = 0; i < 200; i++) {
        for (const b of image(s, i).boxes) {
          expect(b.x1).toBeGreaterThanOrEqual(0);
          expect(b.y1).toBeGreaterThanOrEqual(0);
          expect(b.x2).toBeLessThanOrEqual(s.width);
          expect(b.y2).toBeLessThanOrEqual(s.height);
          expect(b.x2).toBeGreaterThan(b.x1);
          expect(b.class).toBeLessThan(s.classes.length);
        }
      }
    }
  });

  it('computes stats that agree with the images', () => {
    const s = specs[1]!;
    const st = stats(s);
    expect(st.class_counts.reduce((a, b) => a + b, 0)).toBe(st.boxes);
    expect(st.boxes_per_image_hist.counts.reduce((a, b) => a + b, 0)).toBe(total(s));
    expect(st.box_size_hist.edges.length).toBe(st.box_size_hist.counts.length + 1);
  });
});

describe('mock runs', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it('lists runs newest first with one live run part way through', async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    expect(runs.map((r) => r.created)).toEqual([...runs.map((r) => r.created)].sort().reverse());
    const live = runs.filter((r) => r.status === 'running');
    expect(live).toHaveLength(1);
    expect(live[0]!.epoch).toBeGreaterThan(20);
    expect(live[0]!.epoch).toBeLessThan(60);
  });

  it('serves a metric history consistent with the run', async () => {
    const api = new MockApi('normal', 0);
    const run = await api.run('run_7d1e');
    const m = await api.runMetrics('run_7d1e');
    expect(m.epochs).toHaveLength(run.epoch - 1);
    expect(m.steps.at(-1)!.step).toBeLessThanOrEqual(run.step);
    expect(m.epochs.at(-1)!.val.map50_95).toBeGreaterThan(m.epochs[0]!.val.map50_95);
  });

  it('streams steps and epochs as time passes', async () => {
    const api = new MockApi('normal', 0);
    const steps: number[] = [];
    const epochs: number[] = [];
    const off = api.runEvents('run_7d1e', { step: (s) => steps.push(s.step), epoch: (e) => epochs.push(e.epoch) });
    vi.advanceTimersByTime(7000);
    off();
    expect(steps.length).toBeGreaterThan(50);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
    expect(epochs.length).toBeGreaterThanOrEqual(1);
  });

  it('stops a running run', async () => {
    const api = new MockApi('normal', 0);
    const r = await api.stopRun('run_7d1e');
    expect(r.status).toBe('stopping');
    vi.advanceTimersByTime(2000);
    expect((await api.run('run_7d1e')).status).toBe('stopped');
  });

  it('does not report reconnects for a queued run', () => {
    const api = new MockApi('normal', 0);
    let reconnects = 0;
    const statuses: string[] = [];
    const off = api.runEvents('run_q4n2', { reconnect: () => reconnects++, status: (s) => statuses.push(s.status) });
    vi.advanceTimersByTime(1000);
    off();
    expect(reconnects).toBe(0);
    expect(statuses).toEqual(['queued']);
  });

  it('emits the final epoch before the completed status', async () => {
    const api = new MockApi('normal', 0);
    const ds = await api.createSyntheticDataset({ name: 'tiny', images: 20, classes: ['a', 'b'], imgsz: 320, seed: 0 });
    const cfg = (await api.run('run_31aa')).config;
    const run = await api.createRun({ ...cfg, name: 'tiny-n', dataset_id: ds.id, epochs: 2 });
    const events: string[] = [];
    const off = api.runEvents(run.id, { epoch: (e) => events.push(`epoch ${e.epoch}`), status: (s) => events.push(s.status) });
    vi.advanceTimersByTime(1500 + 2 * 300 * 1000 + 500);
    off();
    // Epochs are 1-based, as in folo-api.
    expect(events.filter((e) => e.startsWith('epoch'))).toEqual(['epoch 1', 'epoch 2']);
    expect(events.indexOf('epoch 2')).toBeLessThan(events.indexOf('completed'));
    const model = (await api.models()).find((m) => m.run_id === run.id);
    expect(model?.classes).toEqual(['a', 'b']);
  });

  it('keeps synthetic classes, weights and samples consistent', async () => {
    const api = new MockApi('normal', 0);
    const ds = await api.createSyntheticDataset({ name: 'many', images: 10, classes: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], imgsz: 320, seed: 0 });
    const full = await api.dataset(ds.id);
    expect(full.classes).toHaveLength(6);
    expect(full.stats!.class_counts).toHaveLength(6);
    const cfg = (await api.run('run_31aa')).config;
    const run = await api.createRun({ ...cfg, name: 'many-n', dataset_id: ds.id, epochs: 1 });
    vi.advanceTimersByTime(1500 + 700 * 1000);
    const s = await api.runSamples(run.id);
    expect(s.samples.length).toBe(full.splits.val);
    for (const x of s.samples) expect(Number(x.image_id.slice(4))).toBeLessThan(10);
  });

  it('keeps model classes when the dataset is unregistered', async () => {
    const api = new MockApi('normal', 0);
    await api.deleteDataset('ds_shapes');
    const m = await api.model('m_shapes_n');
    expect(m.classes).toEqual(['circle', 'square', 'triangle']);
  });

  it('predicts in the model class space', async () => {
    const api = new MockApi('normal', 0);
    const p = await api.predict({ kind: 'dataset', dataset_id: 'ds_fasteners', image_id: 'img_000000' }, { model_id: 'm_shapes_n', conf: 0 });
    for (const b of p.boxes) {
      expect(b.class).toBeLessThan(3);
      expect(['circle', 'square', 'triangle']).toContain(b.name);
    }
  });

  it('derives synthetic images from the requested seed', async () => {
    const body = { name: 'seeded', images: 20, classes: ['a', 'b'], imgsz: 320 };
    const a1 = await new MockApi('normal', 0).createSyntheticDataset({ ...body, seed: 1 });
    const a2 = new MockApi('normal', 0);
    await a2.createSyntheticDataset({ ...body, name: 'other', seed: 2 });
    const b1 = await a2.createSyntheticDataset({ ...body, seed: 1 });
    const imgs = async (api: MockApi, id: string) => (await api.datasetImages(id, { limit: 5 })).images;
    const api1 = new MockApi('normal', 0);
    const c1 = await api1.createSyntheticDataset({ ...body, seed: 1 });
    const c9 = await api1.createSyntheticDataset({ ...body, name: 'nine', seed: 999 });
    expect(await imgs(api1, c1.id)).not.toEqual(await imgs(api1, c9.id));
    expect((await imgs(a2, b1.id)).map((i) => i.boxes)).toEqual((await imgs(api1, c1.id)).map((i) => i.boxes));
    void a1;
  });

  it('fails every call in the error scenario', async () => {
    const api = new MockApi('error', 0);
    await expect(api.runs()).rejects.toMatchObject({ code: 'network' });
  });

  it('produces plausible curves', () => {
    const api = new MockApi('normal', 0);
    void api;
    const prof = {
      id: 'x',
      name: 'x',
      config: {
        name: 'x', dataset_id: 'ds_shapes', arch: 'folo-n' as const, epochs: 50, batch: 16, imgsz: 640, device: 'auto' as const,
        optimizer: 'auto' as const, lr0: 0.01, weights: null, seed: 0,
        augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4] as [number, number, number], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
      },
      classes: 3, classNames: ['a', 'b', 'c'], classEase: [1, 1, 1], stepsPerEpoch: 10, epochSeconds: 1, mapMax: 0.7, tau: 8, imgsPerS: 300,
      createdAt: NOW, startedAt: NOW, device: 'cpu',
    };
    expect(epochAt(prof, 49).val.map50_95).toBeGreaterThan(epochAt(prof, 2).val.map50_95);
    expect(stepAt(prof, 499).loss.total).toBeLessThan(stepAt(prof, 5).loss.total);
    // 1-based epochs: steps 1-10 are epoch 1, step 10 is its last, step 11 starts epoch 2.
    expect([1, 10, 11, 500].map((n) => stepAt(prof, n).epoch)).toEqual([1, 1, 2, 50]);
    expect(epochAt(prof, 0).epoch).toBe(1);
    expect(progressAt(prof, NOW + 25_000).status).toBe('running');
    expect(progressAt(prof, NOW + 60_000).status).toBe('completed');
  });
});
