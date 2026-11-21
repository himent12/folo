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

  /** Frees the one training slot: stops the live run and the two waiting ones. */
  async function idle(api: MockApi) {
    for (const id of ['run_7d1e', 'run_q4n2', 'run_q7c5']) await api.stopRun(id);
    vi.advanceTimersByTime(1300);
  }

  it('emits the final epoch before the completed status', async () => {
    const api = new MockApi('normal', 0);
    await idle(api);
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

  it("refuses a synthetic count the real server refuses, as it does, and makes the counts at its limits (issue #382)", async () => {
    const api = new MockApi('normal', 0);
    await idle(api);
    const body = { name: 'n', classes: ['a'], imgsz: 320, seed: 0 };
    await expect(api.createSyntheticDataset({ ...body, images: 10.5 })).rejects.toMatchObject({ status: 422, code: 'invalid_body' });
    await expect(api.createSyntheticDataset({ ...body, images: -1 })).rejects.toMatchObject({ status: 422, code: 'invalid_body' });
    for (const images of [0, 100_001]) await expect(api.createSyntheticDataset({ ...body, images })).rejects.toMatchObject({ status: 400, code: 'invalid_body' });
    for (const images of [1, 100_000]) {
      const ds = await api.createSyntheticDataset({ ...body, name: `n${images}`, images });
      expect(Object.values(ds.splits).reduce((x, y) => x + y, 0)).toBe(images);
    }
  });

  it('keeps synthetic classes, weights and samples consistent', async () => {
    const api = new MockApi('normal', 0);
    await idle(api);
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

describe('mock API: PR #69 behaviour', () => {
  const input = { kind: 'dataset' as const, dataset_id: 'ds_shapes', image_id: '0' };

  it('runs out of GPU memory on the server in the oom scenario, never in the page or for folo-n', async () => {
    const api = new MockApi('oom', 0);
    await expect(api.predict(input, { model_id: 'm_shapes_s' })).rejects.toMatchObject({ status: 507, code: 'gpu_out_of_memory' });
    await expect(api.predictInPage(input, { model_id: 'm_shapes_s' })).resolves.toMatchObject({ model_id: 'm_shapes_s' });
    await expect(api.predict(input, { model_id: 'm_shapes_n' })).resolves.toMatchObject({ model_id: 'm_shapes_n' });
  });

  it('answers a GPU the machine does not have with device_not_found', async () => {
    const api = new MockApi('gpu-unplugged', 0);
    expect((await api.system()).gpus.map((g) => g.ordinal)).toEqual([0, 1]);
    const c = (await api.run('run_c19a')).config;
    await expect(api.createRun({ ...c, name: 'x', device: 'gpu:1' })).rejects.toMatchObject({ status: 404, code: 'device_not_found' });
    expect((await api.system()).gpus.map((g) => g.ordinal)).toEqual([0]);
  });

  it('refuses a run name over 64 characters, as the server does', async () => {
    const api = new MockApi('normal', 0);
    const c = (await api.run('run_c19a')).config;
    await expect(api.createRun({ ...c, name: 'x'.repeat(65) })).rejects.toMatchObject({ status: 400, code: 'invalid_config' });
    await expect(api.createRun({ ...c, name: 'x'.repeat(64) })).resolves.toMatchObject({ name: 'x'.repeat(64) });
  });

  it('returns the registered dataset for a folder that is already registered', async () => {
    const api = new MockApi('normal', 0);
    const before = (await api.datasets()).length;
    expect((await api.createDataset({ name: 'shapes', path: 'd:\\data\\shapes\\', format: 'auto' })).id).toBe('ds_shapes');
    expect(await api.datasets()).toHaveLength(before);
  });

  it('counts every issue whatever the filter, and says what a cleanup would do', async () => {
    const api = new MockApi('normal', 0);
    const all = await api.datasetIssues('ds_shapes_hard', { limit: 1000 });
    const one = await api.datasetIssues('ds_shapes_hard', { severity: 'error' });
    expect(one.counts).toEqual(all.counts);
    expect(one.counts?.error).toBe(3);
    expect(one.issues.every((i) => i.fix?.startsWith('delete line'))).toBe(true);
    expect(all.issues.filter((i) => i.severity === 'info').every((i) => i.fix === undefined)).toBe(true);
  });
});

describe('mock queue (docs/API.md, POST /api/runs/{id}/queue)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  const places = async (api: MockApi) =>
    Object.fromEntries((await api.runs()).filter((r) => r.queue_position != null).map((r) => [r.id, r.queue_position]));

  it('has two runs waiting behind the live one, front first, and lists queue control and the ONNX download', async () => {
    const api = new MockApi('normal', 0);
    expect(await places(api)).toEqual({ run_q4n2: 1, run_q7c5: 2 });
    expect((await api.run('run_7d1e')).queue_position).toBeNull();
    expect((await api.system()).features).toEqual(expect.arrayContaining(['queue_control', 'weights_onnx']));
  });

  it('moves a waiting run, logs it, and tells its subscribers where it is now', async () => {
    const api = new MockApi('normal', 0);
    const queue: { position: number; waiting: number }[] = [];
    const logs: string[] = [];
    const off = api.runEvents('run_q7c5', { queue: (q) => queue.push(q), log: (l) => logs.push(l.message) });
    const r = await api.moveRun('run_q7c5', { action: 'up' });
    expect(r.queue_position).toBe(1);
    expect(await places(api)).toEqual({ run_q7c5: 1, run_q4n2: 2 });
    vi.advanceTimersByTime(250);
    off();
    expect(queue).toEqual([{ position: 1, waiting: 2 }]);
    expect(logs).toHaveLength(1);
    expect((await api.runLogs('run_q7c5')).lines.map((l) => l.message)).toEqual(logs);
  });

  it('start never interrupts the live run: the run goes to the front and waits', async () => {
    const api = new MockApi('normal', 0);
    const r = await api.moveRun('run_q7c5', { action: 'start' });
    expect(r).toMatchObject({ status: 'queued', queue_position: 1 });
    expect((await api.run('run_7d1e')).status).toBe('running');
  });

  it('starts the front run once the slot is free, with the resolved config, and refuses to move it then', async () => {
    const api = new MockApi('normal', 0);
    expect((await api.run('run_q4n2')).resolved).toBeNull();
    await api.stopRun('run_7d1e');
    vi.advanceTimersByTime(1300);
    expect((await api.run('run_q4n2')).queue_position).toBeNull();
    // Taken by the worker, still queued while its trainer is built.
    await expect(api.moveRun('run_q4n2', { action: 'down' })).rejects.toMatchObject({ status: 409, code: 'run_not_queued' });
    vi.advanceTimersByTime(2000);
    const r = await api.run('run_q4n2');
    expect(r.status).toBe('running');
    // shapes-hard: auto (rule 5) picks MuSGD at 0.01 for every run.
    expect(r.resolved).toMatchObject({ optimizer: 'musgd', lr0: 0.01, batch: 32, accumulate: 2 });
    expect(await places(api)).toEqual({ run_q7c5: 1 });
    await expect(api.moveRun('run_31aa', { action: 'up' })).rejects.toMatchObject({ status: 409, code: 'run_not_queued' });
    await expect(api.moveRun('run_q7c5', { position: 0 })).rejects.toMatchObject({ status: 400, code: 'invalid_body' });
  });

  it('leaves a run stopped while it waited without a resolved config, and queues new runs at the back', async () => {
    const api = new MockApi('normal', 0);
    const stopped = await api.stopRun('run_q4n2');
    expect(stopped).toMatchObject({ status: 'stopped', queue_position: null, resolved: null });
    expect(await places(api)).toEqual({ run_q7c5: 1 });
    const cfg = (await api.run('run_31aa')).config;
    const r = await api.createRun({ ...cfg, name: 'shapes-n-2' });
    expect(r).toMatchObject({ status: 'queued', queue_position: 2 });
  });

  it('deletes a run that waits and never started (the one behind moves up), never the live one', async () => {
    const api = new MockApi('normal', 0);
    await api.deleteRun('run_q4n2');
    expect(await places(api)).toEqual({ run_q7c5: 1 });
    await expect(api.run('run_q4n2')).rejects.toMatchObject({ status: 404 });
    await expect(api.deleteRun('run_7d1e')).rejects.toMatchObject({ status: 409, code: 'run_active' });
  });
});

describe('mock operating point', () => {
  it('reports best_f1_conf on every epoch and run model, the same in /eval, and none for the imported COCO file', async () => {
    const api = new MockApi('normal', 0);
    const m = await api.model('m_fast_n');
    const c = m.metrics?.best_f1_conf ?? 0;
    expect(c).toBeGreaterThan(0.1);
    expect(c).toBeLessThan(0.6);
    expect(Math.abs(c * 1000 - Math.round(c * 1000))).toBeLessThan(1e-6);
    expect((await api.modelEval('m_fast_n')).best_f1_conf).toBe(m.metrics?.best_f1_conf);
    const { epochs } = await api.runMetrics('run_c19a');
    expect(epochs.every((e) => typeof e.val.best_f1_conf === 'number')).toBe(true);
    expect((await api.model('m_coco_s')).metrics?.best_f1_conf).toBeNull();
    expect((await api.modelEval('m_coco_s')).best_f1_conf).toBeNull();
  });

  it("says what /eval ran on: the run's dataset and val split, a train-only dataset's training split, the matrix's thresholds", async () => {
    const api = new MockApi('normal', 0);
    const run = await api.modelEval('m_fast_n');
    expect(run).toMatchObject({ dataset_id: 'ds_fasteners', dataset_name: 'fasteners', split: 'val', split_fallback: null, images: 360 });
    expect(run).toMatchObject({ confusion_conf: 0.25, confusion_iou: 0.45 });
    // fasteners-mini has no val images: the imported model with its classes is evaluated on its training split.
    expect((await api.dataset('ds_fasteners_mini')).splits.val).toBe(0);
    const mini = await api.modelEval('m_mini_n');
    expect(mini).toMatchObject({ dataset_id: 'ds_fasteners_mini', dataset_name: 'fasteners-mini', split: 'train', images: 24 });
    expect(mini.split_fallback).toEqual({
      requested: 'val',
      reason: 'empty_split',
      message: "dataset 'fasteners-mini' has no val images, so this is its train split (24 images): training-set numbers, not held-out ones",
    });
    // The matrix's rows count the training split's boxes.
    const boxes = (await api.dataset('ds_fasteners_mini')).stats!.class_counts;
    expect(mini.confusion.slice(0, 2).map((r) => r.reduce((a, b) => a + b, 0))).toEqual(boxes);
    // No registered dataset has folo-s-coco80's classes: the demo answers as an older server, without provenance.
    const coco = await api.modelEval('m_coco_s');
    for (const k of ['dataset_id', 'dataset_name', 'split', 'split_fallback', 'confusion_conf', 'confusion_iou'] as const) expect(coco[k]).toBeUndefined();
  });

});

// mAP is the mean over classes (docs/METRICS.md), as folo-server's mock reports it
// (`mock_model_metrics_match_their_run_epoch`): the headline, the per-class table, the PR curves' "All classes" and
// the run's per-class at its best epoch are one set of numbers on every demo model and run.
describe('mock metrics reconcile', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

  it('every model: the class means are its metrics, and each PR curve’s AP is its class AP50', async () => {
    for (const scenario of ['normal', 'no-throughput'] as const) {
      const api = new MockApi(scenario, 0);
      const models = await api.models();
      expect(models.length).toBeGreaterThan(3);
      for (const m of models) {
        if (!m.metrics) continue;
        const ev = await api.modelEval(m.id);
        expect(ev.per_class).toHaveLength(m.classes.length);
        expect(Math.abs(mean(ev.per_class.map((c) => c.map50)) - m.metrics.map50), `${m.name} AP50`).toBeLessThan(0.001);
        expect(Math.abs(mean(ev.per_class.map((c) => c.map50_95)) - m.metrics.map50_95), `${m.name} AP50-95`).toBeLessThan(0.001);
        for (const c of ev.per_class) {
          const pr = ev.pr[c.class]!;
          expect(Math.abs(mean(pr.precision) - pr.ap50)).toBeLessThan(0.0001);
          expect(pr.ap50).toBe(c.map50);
        }
        // The "All classes" curve (the mean over classes at each recall) has the model's mAP50 as its AP.
        const all = mean(ev.pr[0]!.recall.map((_, i) => mean(ev.pr.map((p) => p.precision[i]!))));
        expect(Math.abs(all - m.metrics.map50), `${m.name} all classes`).toBeLessThan(0.001);
      }
    }
  });

  it('every run epoch: the overall metrics are the means of its per-class metrics', async () => {
    const api = new MockApi('normal', 0);
    for (const r of await api.runs()) {
      const { epochs } = await api.runMetrics(r.id);
      for (const e of epochs) {
        if (!e.val.per_class.length) continue;
        expect(Math.abs(mean(e.val.per_class.map((c) => c.map50)) - e.val.map50), `${r.name} epoch ${e.epoch}`).toBeLessThan(0.001);
        expect(Math.abs(mean(e.val.per_class.map((c) => c.map50_95)) - e.val.map50_95), `${r.name} epoch ${e.epoch}`).toBeLessThan(0.001);
      }
    }
  });
});

describe('mock: the final fields of server PR #140', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it('fine-tunes from published weights in their architecture, whatever `arch` says', async () => {
    const api = new MockApi('normal', 0);
    const c = { ...(await api.run('run_31aa')).config, name: 'coco-ft', arch: 'folo-s' as const, weights: 'folo-n-coco' };
    const est = await api.estimateRun(c);
    expect(est.basis.arch).toBe('folo-n');
    const run = await api.createRun(c);
    expect(run.config.arch).toBe('folo-n');
    await expect(api.createRun({ ...c, weights: 'folo-x-nope' })).rejects.toMatchObject({ status: 404, code: 'model_not_found' });
  });

  it("serves a run's CLI command (run_cli): quoted for PowerShell, its argv exact", async () => {
    const api = new MockApi('normal', 0);
    expect((await api.system()).features).toContain('run_cli');
    const { cli, argv } = await api.runCli('run_c19a');
    expect(argv.slice(0, 2)).toEqual(['folo', 'train']);
    expect(argv.join(' ')).toBe(cli);
    await expect(api.runCli('run_nope')).rejects.toMatchObject({ status: 404, code: 'run_not_found' });
    // A server from before the field has no such route.
    const old = new MockApi('no-throughput', 0);
    expect((await old.system()).features).not.toContain('run_cli');
    await expect(old.runCli('run_c19a')).rejects.toMatchObject({ status: 404 });
  });

  it('estimates close mosaic after the cap, as the run will report it', async () => {
    const api = new MockApi('normal', 0);
    const c = (await api.run('run_f15c')).config;
    expect((await api.estimateRun(c)).close_mosaic).toBe(7);
    expect((await api.run('run_f15c')).resolved?.close_mosaic).toBe(7);
    expect((await api.estimateRun({ ...c, epochs: 100 })).close_mosaic).toBe(10);
  });

  it('refuses an amp run the server cannot train with 501 not_supported, before anything is queued', async () => {
    const api = new MockApi('tensor-cores-off', 0);
    const c = { ...(await api.run('run_31aa')).config, name: 'amp', device: 'auto' as const, amp: true };
    const before = (await api.runs()).length;
    await expect(api.createRun(c)).rejects.toMatchObject({ status: 501, code: 'not_supported', message: (await api.system()).capabilities?.amp.reason });
    expect((await api.runs()).length).toBe(before);
    await expect(new MockApi('normal', 0).createRun({ ...c, device: 'cpu' })).rejects.toMatchObject({ status: 501, code: 'not_supported' });
    await expect(new MockApi('normal', 0).createRun(c)).resolves.toMatchObject({ config: { amp: true } });
  });

  it('refuses an estimate as it refuses the run, without the simulated restart', async () => {
    const normal = new MockApi('normal', 0);
    const c = { ...(await normal.run('run_31aa')).config, name: 'amp', device: 'cpu' as const, amp: true };
    await expect(normal.estimateRun(c)).rejects.toMatchObject({ status: 501, code: 'not_supported' });
    const restarted = new MockApi('amp-refused', 0);
    await expect(restarted.estimateRun({ ...c, device: 'auto' })).resolves.toBeTruthy();
    expect((await restarted.system()).capabilities?.amp.available).toBe(true);
  });

  it('amp-refused: says amp is available, refuses it at POST, then says it is not', async () => {
    const api = new MockApi('amp-refused', 0);
    expect((await api.system()).capabilities?.amp.available).toBe(true);
    const c = { ...(await api.run('run_31aa')).config, name: 'amp', device: 'auto' as const, amp: true };
    await expect(api.createRun(c)).rejects.toMatchObject({ status: 501, code: 'not_supported' });
    const caps = (await api.system()).capabilities!;
    expect(caps.amp.available).toBe(false);
    expect(caps.tensor_cores).toMatchObject({ available: true, enabled: false });
    await expect(api.createRun({ ...c, amp: false })).resolves.toMatchObject({ config: { amp: false } });
  });
});
