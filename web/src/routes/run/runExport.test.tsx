// @vitest-environment happy-dom
// Run detail metrics export (CSV/JSON + Blob URL lifecycle cleanup) and checkpoint CLI quick-copy (`folo val` / `folo predict`).
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Dataset, Epoch, Model, Run } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../framework/testing';
import RunPage from '../Run';
import CheckpointCli from './CheckpointCli';
import {
  buildCheckpointCli,
  buildLatestPerClass,
  buildRunMetricsCsvRows,
  buildRunMetricsJson,
  downloadRunMetricsCsv,
  downloadRunMetricsJson,
  downloadRunPerClassCsv,
} from './runExport';

const SAMPLE_RUN: Run = {
  id: 'run_exp1',
  name: 'fasteners-s',
  status: 'completed',
  config: {
    name: 'fasteners-s',
    dataset_id: 'ds_fasteners',
    arch: 'folo-s',
    epochs: 2,
    batch: 16,
    imgsz: 640,
    device: 'gpu:0',
    optimizer: 'auto',
    lr0: 0.01,
    weights: null,
    augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 1 },
    seed: 42,
  },
  created: '2026-11-20T00:00:00Z',
  started: '2026-11-20T00:00:05Z',
  finished: '2026-11-20T00:05:00Z',
  epoch: 2,
  epochs: 2,
  step: 20,
  steps_per_epoch: 10,
  eta_s: 0,
  best: { epoch: 2, map50_95: 0.512 },
  device: 'gpu:0',
  model_id: 'm_fast_s',
  error: null,
};

const SAMPLE_EPOCHS: Epoch[] = [
  {
    epoch: 1,
    duration_s: 12.4,
    train_loss: { box: 1.2, cls: 0.8, dfl: 1.0, total: 3.0 },
    val: {
      map50: 0.55,
      map50_95: 0.31,
      precision: 0.62,
      recall: 0.58,
      best_f1_conf: 0.35,
      per_class: [
        { class: 0, map50: 0.6, map50_95: 0.35, precision: 0.7, recall: 0.6 },
        { class: 1, map50: 0.5, map50_95: 0.27, precision: 0.54, recall: 0.56 },
      ],
    },
  },
  {
    epoch: 2,
    duration_s: 11.8,
    train_loss: { box: 0.9, cls: 0.5, dfl: 0.8, total: 2.2 },
    val: {
      map50: 0.78,
      map50_95: 0.512,
      precision: 0.81,
      recall: 0.74,
      best_f1_conf: 0.42,
      per_class: [
        { class: 0, map50: 0.75, map50_95: 0.48, precision: 0.79, recall: 0.71 },
        { class: 1, map50: 0.81, map50_95: 0.544, precision: 0.83, recall: 0.77 },
      ],
    },
  },
];

const SAMPLE_MODEL: Model = {
  id: 'm_fast_s',
  name: 'fasteners-s-best',
  arch: 'folo-s',
  classes: ['bolt', '=nut'],
  imgsz: 640,
  path: "D:/O'Brien/models/fasteners-s best.folo",
  params: 9_000_000,
  flops_g: 21.4,
  bytes: 18_000_000,
  created: '2026-11-20T00:05:00Z',
  run_id: 'run_exp1',
  metrics: { map50: 0.78, map50_95: 0.512, precision: 0.81, recall: 0.74 },
};

const SAMPLE_DATASET: Dataset = {
  id: 'ds_fasteners',
  name: 'fasteners',
  path: 'D:/my data/fasteners',
  format: 'yolo',
  classes: ['bolt', '=nut'],
  created: '2026-11-01T00:00:00Z',
  splits: { train: 100, val: 20, test: 0 },
};

let target: HTMLElement;
let mounted: Rendered | undefined;

beforeEach(() => {
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  vi.useRealTimers();
  vi.restoreAllMocks();
  toast.items = [];
});

describe('runExport pure builders and Blob lifecycle', () => {
  it('builds per-epoch rows, sorted per-class rows with delta, and full JSON payload (omitting inexact seeds)', () => {
    const rows = buildRunMetricsCsvRows(SAMPLE_EPOCHS);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toEqual([1, 12.4, 1.2, 0.8, 1.0, 3.0, 0.55, 0.31, 0.62, 0.58, 0.35]);

    const perClass = buildLatestPerClass(SAMPLE_EPOCHS, ['bolt', '=nut']);
    expect(perClass.map((r) => r.name)).toEqual(['=nut', 'bolt']);
    expect(perClass[0]!.delta_prev).toBeCloseTo(0.274, 5);
    expect(perClass[1]!.delta_prev).toBeCloseTo(0.13, 5);

    const payload = buildRunMetricsJson(
      { ...SAMPLE_RUN, config: { ...SAMPLE_RUN.config, seed: 2 ** 53 } },
      SAMPLE_EPOCHS,
      ['bolt', '=nut'],
      'fasteners',
      '2026-11-27T00:00:00.000Z',
    );
    expect(payload.exported_at).toBe('2026-11-27T00:00:00.000Z');
    expect('seed' in payload.config).toBe(false);
    expect(payload.epochs).toHaveLength(2);
    expect(payload.latest_per_class).toHaveLength(2);
    expect(payload.run.dataset_name).toBe('fasteners');
  });

  it('downloads CSV, per-class CSV (formula-safe), and JSON and revokes every object URL without leaking timers', async () => {
    vi.useFakeTimers();
    const created: Array<{ blob: Blob; url: string }> = [];
    const revoked: string[] = [];
    const clicked: Array<{ href: string; download: string }> = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      const url = `blob:run-export/${created.length}`;
      created.push({ blob: b as Blob, url });
      return url;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((u) => revoked.push(u));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push({ href: this.href, download: this.download });
    });

    expect(downloadRunMetricsCsv(SAMPLE_RUN, SAMPLE_EPOCHS)).toBe(true);
    expect(downloadRunPerClassCsv(SAMPLE_RUN, SAMPLE_EPOCHS, ['bolt', '=nut'])).toBe(true);
    expect(downloadRunMetricsJson(SAMPLE_RUN, SAMPLE_EPOCHS, ['bolt', '=nut'], 'fasteners')).toBe(true);

    expect(clicked.map((c) => c.download)).toEqual([
      'fasteners-s-metrics.csv',
      'fasteners-s-per-class.csv',
      'fasteners-s-metrics.json',
    ]);
    const csvText = await created[0]!.blob.text();
    expect(csvText).toContain('epoch,duration_s,train_box_loss');
    expect(csvText).toContain('2,11.8,0.9,0.5,0.8,2.2,0.78,0.512');

    // Spreadsheet formula neutralization on class name `=nut`
    const perClassCsv = await created[1]!.blob.text();
    expect(perClassCsv).toContain("1,'=nut,0.544");

    const jsonObj = JSON.parse(await created[2]!.blob.text()) as { run: { id: string }; epochs: unknown[] };
    expect(jsonObj.run.id).toBe('run_exp1');
    expect(jsonObj.epochs).toHaveLength(2);

    expect(revoked).toEqual([]);
    vi.advanceTimersByTime(1000);
    expect(revoked).toEqual(['blob:run-export/0', 'blob:run-export/1', 'blob:run-export/2']);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('builds shell-safe folo val and folo predict checkpoint CLI commands with placeholders and split fallback', () => {
    expect(buildCheckpointCli({ ...SAMPLE_RUN, model_id: null }, undefined, SAMPLE_DATASET)).toBeNull();

    const withPaths = buildCheckpointCli(SAMPLE_RUN, SAMPLE_MODEL, SAMPLE_DATASET, null)!;
    expect(withPaths.val).toBe("folo val --model 'D:/O''Brien/models/fasteners-s best.folo' --data 'D:/my data/fasteners'");
    expect(withPaths.predict).toBe("folo predict --model 'D:/O''Brien/models/fasteners-s best.folo' <images>");
    expect(withPaths.modelPlaceholder).toBeNull();
    expect(withPaths.dataPlaceholder).toBeNull();

    // Train-only fallback adds --split train; leading-dash paths are attached safely via --flag=...
    const dashPaths = buildCheckpointCli(
      SAMPLE_RUN,
      { ...SAMPLE_MODEL, path: '-weird.folo' },
      { ...SAMPLE_DATASET, path: '-data dir' },
      { datasetId: 'ds_fasteners', datasetName: 'fasteners', split: 'train', fallback: null },
    )!;
    expect(dashPaths.val).toBe("folo val --model=-weird.folo '--data=-data dir' --split train");

    // Missing paths emit readable placeholders, never bare server ids
    const noPaths = buildCheckpointCli(SAMPLE_RUN, undefined, undefined, null)!;
    expect(noPaths.val).toContain('--model <path to fasteners-s-best.folo>');
    expect(noPaths.val).toContain('--data <path to ds_fasteners>');
    expect(noPaths.val).not.toContain('m_fast_s');
    expect(noPaths.modelPlaceholder).toBe('fasteners-s-best.folo');
    expect(noPaths.dataPlaceholder).toBe('ds_fasteners');
  });
});

describe('CheckpointCli and Run page export integration', () => {
  it('toggles between Validate and Predict commands and copies to clipboard', async () => {
    const bundle = buildCheckpointCli(SAMPLE_RUN, SAMPLE_MODEL, SAMPLE_DATASET, null)!;
    mounted = render(() => <CheckpointCli bundle={bundle} />, target);
    flushSync();

    const toggle = target.querySelector<HTMLButtonElement>('.ckpt-toggle')!;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(target.querySelector('pre.line')?.textContent).toBe(bundle.val);

    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    target.querySelector<HTMLButtonElement>('button[aria-label="Copy command"]')!.click();
    await Promise.resolve();
    expect(write).toHaveBeenCalledWith(bundle.val);

    // Switch to Predict
    const predictBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.trim() === 'Predict')!;
    predictBtn.click();
    flushSync();
    expect(target.querySelector('pre.line')?.textContent).toBe(bundle.predict);
  });

  it('wires Export metrics menu and Checkpoint CLI into a loaded completed Run page', async () => {
    const api = new MockApi('normal', 0);
    const run = (await api.runs()).find((r) => r.status === 'completed' && r.model_id)!;
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#/runs/${run.id}`);
    mounted = render(() => <RunPage />, target);
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('#per-class table') || !target.querySelector('[data-checkpoint-cli]')) {
        throw new Error('waiting for completed run');
      }
    });

    // Per-class panel header has Export menu
    const exportBtn = target.querySelector<HTMLButtonElement>('#per-class button[aria-label="Export run metrics"]')!;
    exportBtn.click();
    flushSync();
    const labels = [...target.querySelectorAll('#per-class [role="menuitem"]')].map((e) => e.textContent?.trim());
    expect(labels).toEqual(['Export metrics (CSV)', 'Export per-class AP (CSV)', 'Export metrics (JSON)']);

    // Config panel includes collapsible Checkpoint CLI
    const ckptToggle = target.querySelector<HTMLButtonElement>('[data-checkpoint-cli] .ckpt-toggle')!;
    ckptToggle.click();
    flushSync();
    expect(target.querySelector('[data-checkpoint-cli] pre.line')?.textContent).toContain('folo val --model');
  });

  it('reads current run metadata when exporting after a live run transitions from running to completed', async () => {
    const api = new MockApi('normal', 0);
    const baseRun = (await api.runs()).find((r) => r.status === 'running')!;
    let currentRun: Run = { ...baseRun, status: 'running', epoch: 1, finished: null, model_id: null };
    let emitStatus: ((ev: { status: Run['status']; error: string | null }) => void) | undefined;
    const realRun = api.run.bind(api);
    vi.spyOn(api, 'run').mockImplementation(async (id) => (id === baseRun.id ? { ...currentRun } : realRun(id)));
    const realEvents = api.runEvents.bind(api);
    vi.spyOn(api, 'runEvents').mockImplementation((id, handlers) => {
      if (id === baseRun.id) emitStatus = handlers.status;
      return realEvents(id, handlers);
    });

    const createdBlobs: Blob[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      createdBlobs.push(b as Blob);
      return `blob:live-transition/${createdBlobs.length}`;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#/runs/${baseRun.id}`);
    mounted = render(() => <RunPage />, target);
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('#per-class button[aria-label="Export run metrics"]')) {
        throw new Error('waiting for live run export button');
      }
    });

    // Transition the live run to completed; sameItems retains menu structure, callback must read latest Run.
    currentRun = {
      ...currentRun,
      status: 'completed',
      epoch: 2,
      finished: '2026-11-27T01:00:00Z',
      model_id: 'm_completed_best',
      best: { epoch: 2, map50_95: 0.654 },
    };
    emitStatus?.({ status: 'completed', error: null });
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('a[href="#/models/m_completed_best"]')) throw new Error('waiting for completed run refresh');
    });

    target.querySelector<HTMLButtonElement>('#per-class button[aria-label="Export run metrics"]')!.click();
    flushSync();
    const jsonItem = [...target.querySelectorAll<HTMLButtonElement>('#per-class [role="menuitem"]')].find(
      (b) => b.textContent?.trim() === 'Export metrics (JSON)',
    )!;
    jsonItem.click();
    expect(createdBlobs).toHaveLength(1);
    const exported = JSON.parse(await createdBlobs[0]!.text()) as {
      run: { status: string; finished: string | null; model_id: string | null; best: { map50_95: number } | null };
    };
    expect(exported.run.status).toBe('completed');
    expect(exported.run.finished).toBe('2026-11-27T01:00:00Z');
    expect(exported.run.model_id).toBe('m_completed_best');
    expect(exported.run.best?.map50_95).toBe(0.654);
  });

  it('keeps the <images> placeholder hint visible in Predict mode even when model path is also a placeholder', () => {
    const bundle = buildCheckpointCli(SAMPLE_RUN, undefined, undefined, null)!;
    mounted = render(() => <CheckpointCli bundle={bundle} />, target);
    flushSync();
    target.querySelector<HTMLButtonElement>('.ckpt-toggle')!.click();
    flushSync();
    const predictBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.trim() === 'Predict')!;
    predictBtn.click();
    flushSync();
    const hint = target.querySelector('.ckpt-hint')?.textContent ?? '';
    expect(hint).toContain('<path to fasteners-s-best.folo>');
    expect(hint).toContain('<images>');
  });
});
