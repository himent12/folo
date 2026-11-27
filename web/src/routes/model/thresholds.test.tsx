// @vitest-environment happy-dom
// Tests for measured operating-point thresholds on Model Detail, ThresholdCalibration JSON export &
// Playground handoff (`Apply in Playground`, Back/Forward/reload history semantics, handoff TTL/scope,
// session origin attribution), plus `folo val --save-thresholds` & `folo predict --thresholds` CLI builders.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { fromModelEval, type Evaluation } from '$lib/api/evaluation';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Model } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import ModelPage from '../Model';
import Playground from '../Playground';
import { parseThresholdsJson } from '../playground/perClassConf';
import {
  consumeThresholdHandoff,
  HANDOFF_TTL_MS,
  perClassFromUrl,
  playgroundPath,
  serializePerClassParam,
  session,
  stageThresholdHandoff,
} from '../playground/session';
import CheckpointCli from '../run/CheckpointCli';
import { buildCheckpointCli } from '../run/runExport';
import ExportDialog from './ExportDialog';
import { buildPredictCli, buildValCli } from './modelCli';
import { buildModelThresholds } from './thresholds';

let target: HTMLElement;
let view: Rendered | undefined;
let downloads: Array<{ filename: string; text: string }> = [];

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  session.set(null);
  consumeThresholdHandoff(null);
  downloads = [];
  target = document.createElement('div');
  document.body.appendChild(target);

  const blobs = new Map<string, Blob>();
  vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
    const url = `blob:thr/${blobs.size}`;
    blobs.set(url, b as Blob);
    return url;
  });
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    const blob = blobs.get(this.getAttribute('href') ?? '');
    if (blob) {
      void blob.text().then((text) => downloads.push({ filename: this.download, text }));
    }
  });
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  session.set(null);
  consumeThresholdHandoff(null);
  toast.items = [];
  vi.restoreAllMocks();
});

describe('buildModelThresholds', () => {
  it('exports only the evaluator-measured best_f1_conf operating point and measured per-class metrics without fabricating PR-curve confidences', () => {
    const model: Pick<Model, 'classes' | 'metrics'> = {
      classes: ['bolt', 'nut', 'washer'],
      metrics: { map50: 0.85, map50_95: 0.62, precision: 0.84, recall: 0.80, best_f1_conf: 0.40 },
    };
    const ev: Evaluation = {
      perClass: [
        // Out-of-order + duplicate class 0 + zero-instance class 2 to verify dedup/sort/filter.
        { class: 1, ap50: 0.82, ap50_95: 0.58, precision: 0.90, recall: 0.50, instances: 10 },
        { class: 0, ap50: 0.90, ap50_95: 0.70, precision: 0.75, recall: 0.90, instances: 20 },
        { class: 0, ap50: 0.10, ap50_95: 0.10, precision: 0.10, recall: 0.10, instances: 20 },
        { class: 2, ap50: 0, ap50_95: 0, precision: 0, recall: 0, instances: 0 },
      ],
      curves: [
        [
          { x: 0.5, y: 0.99 },
          { x: 0.7, y: 0.98 },
          { x: 0.9, y: 0.75 },
        ],
        [
          { x: 0.5, y: 0.90 },
          { x: 0.85, y: 0.86 },
        ],
        [],
      ],
      mean: [],
      // Separate confusion-matrix cutoff (0.25) with inflated counts that MUST NOT leak into `preds`.
      confusion: [
        [19, 0, 0, 1],
        [0, 9, 0, 1],
        [0, 0, 0, 0],
        [50, 40, 0, 0],
      ],
      images: 16,
      bestF1Conf: 0.40,
      source: { datasetId: 'ds_1', datasetName: 'fasteners', split: 'val', fallback: null },
      matrixAt: { conf: 0.25, iou: 0.45 },
    };

    const cal = buildModelThresholds(model, ev)!;
    expect(cal).not.toBeNull();
    expect(cal.hasMeasuredPerClass).toBe(false);
    expect(cal.perClassConf).toEqual({});
    expect(cal.payload.global.best_conf).toBe(0.4);
    expect(cal.payload.per_class.map((c) => c.class)).toEqual([0, 1]);
    expect(cal.payload.per_class.map((c) => c.name)).toEqual(['bolt', 'nut']);
    // Every class carries the exact measured global operating point and exact measured precision/recall at 0.40.
    expect(cal.payload.per_class[0]!.best_conf).toBe(0.4);
    expect(cal.payload.per_class[0]!.precision).toBe(0.75);
    expect(cal.payload.per_class[0]!.recall).toBe(0.9);
    expect(cal.payload.per_class[0]!.truths).toBe(20);
    // tp = 18, precision = 0.75 -> preds at best_f1_conf = 24 (NOT confusion column sum 69).
    expect(cal.payload.per_class[0]!.preds).toBe(24);
    expect(cal.payload.per_class[1]!.best_conf).toBe(0.4);
    expect(cal.payload.per_class[1]!.precision).toBe(0.9);
    expect(cal.payload.per_class[1]!.recall).toBe(0.5);
    expect(cal.payload.macro_f1).toBe(cal.payload.global.best_f1);

    // Every field required by Rust ThresholdCalibration::validate() is finite and in range.
    for (const c of cal.payload.per_class) {
      expect(c.best_conf).toBeGreaterThanOrEqual(0);
      expect(c.best_conf).toBeLessThanOrEqual(1);
      expect(Number.isInteger(c.truths)).toBe(true);
      expect(Number.isInteger(c.preds)).toBe(true);
    }

    // Round-trips cleanly through Playground's JSON parser.
    const parsed = parseThresholdsJson(JSON.stringify(cal.payload), model.classes);
    expect(parsed.ok).toBe(true);
    if (parsed.ok) {
      expect(parsed.globalConf).toBe(0.4);
      expect(parsed.perClassConf[0]).toBe(0.4);
      expect(parsed.perClassConf[1]).toBe(0.4);
    }
  });

  it('returns null when no evaluated classes have ground truth or when best_f1_conf was not measured', () => {
    const evNoGt: Evaluation = {
      perClass: [{ class: 0, ap50: 0, ap50_95: 0, precision: 0, recall: 0, instances: 0 }],
      curves: [[]],
      mean: [],
      confusion: [
        [0, 0],
        [0, 0],
      ],
      images: 0,
      bestF1Conf: 0.4,
      source: { datasetId: null, datasetName: null, split: null, fallback: null },
      matrixAt: null,
    };
    expect(buildModelThresholds({ classes: ['a'], metrics: null }, evNoGt)).toBeNull();

    const evNoConf: Evaluation = {
      ...evNoGt,
      perClass: [{ class: 0, ap50: 0.8, ap50_95: 0.6, precision: 0.8, recall: 0.8, instances: 5 }],
      bestF1Conf: null,
      matrixAt: { conf: 0.25, iou: 0.45 },
    };
    expect(buildModelThresholds({ classes: ['a'], metrics: null }, evNoConf)).toBeNull();
  });
});

describe('Calibrated thresholds CLI builders (--save-thresholds / --thresholds)', () => {
  const model: Model = {
    id: 'm_fast_n',
    name: 'fasteners-n-best',
    arch: 'folo-n',
    imgsz: 640,
    classes: ['bolt', 'nut'],
    params: 2_624_064,
    flops_g: 6.5,
    bytes: 5_400_000,
    created: '2026-11-01T00:00:00Z',
    run_id: 'r_1',
    metrics: null,
    path: 'D:/folo/runs/fasteners-n/weights/best.folo',
  };

  it('builds folo val --save-thresholds and folo predict --thresholds commands with safe quoting and non-val splits', () => {
    expect(buildValCli(model).line).toBe(
      'folo val --model D:/folo/runs/fasteners-n/weights/best.folo --data <dataset> --save-thresholds thresholds.json',
    );
    expect(
      buildValCli(model, {
        datasetPath: 'D:/my data/fasteners',
        split: 'train',
        saveThresholds: true,
        thresholdsPath: 'fasteners-n-best-thresholds.json',
      }).line,
    ).toBe(
      "folo val --model D:/folo/runs/fasteners-n/weights/best.folo --data 'D:/my data/fasteners' --split train --save-thresholds fasteners-n-best-thresholds.json",
    );
    expect(buildValCli(model, { split: 'test' }).line).toContain('--split test');
    expect(buildValCli(model, { saveThresholds: false }).line).toBe(
      'folo val --model D:/folo/runs/fasteners-n/weights/best.folo --data <dataset>',
    );

    // --thresholds combines cleanly with standard, SAHI, TTA, and ByteTrack without emitting --conf.
    const predStd = buildPredictCli(model, 'standard', { thresholdsPath: 'thresholds.json' }).line;
    expect(predStd).toBe('folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <images> --thresholds thresholds.json');
    expect(predStd).not.toContain('--conf');
    expect(buildPredictCli(model, 'sahi', { thresholdsPath: 'thresholds.json' }).line).toBe(
      'folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <images> --thresholds thresholds.json --tile 640 --tile-overlap 0.2',
    );
    expect(buildPredictCli(model, 'track', { thresholdsPath: 'my thresholds.json' }).line).toBe(
      "folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <frames_dir> --thresholds 'my thresholds.json' --track",
    );
  });

  it('toggles --save-thresholds and --thresholds in ExportDialog CLI commands and Run CheckpointCli', async () => {
    view = render(
      () => <ExportDialog model={model} open={true} datasetPath="D:/data/fasteners" evalSplit="val" />,
      target,
    );
    flushSync();
    target.ownerDocument.querySelector<HTMLButtonElement>('[data-export-cli] button.cli-toggle')!.click();
    flushSync();

    const valPre = await until(() => target.ownerDocument.querySelector('pre[aria-label="Validate CLI command"]'), 'val CLI');
    expect(valPre.textContent).toBe(
      'folo val --model D:/folo/runs/fasteners-n/weights/best.folo --data D:/data/fasteners --save-thresholds fasteners-n-best-thresholds.json',
    );
    const predPre = target.ownerDocument.querySelector('pre[aria-label="Predict CLI command"]')!;
    expect(predPre.textContent).not.toContain('--thresholds');

    const checks = [...target.ownerDocument.querySelectorAll<HTMLInputElement>('[data-export-cli] input[type="checkbox"]')];
    expect(checks).toHaveLength(2);
    // Uncheck Save thresholds
    checks[0]!.click();
    flushSync();
    expect(valPre.textContent).not.toContain('--save-thresholds');
    // Check Use calibrated thresholds on Predict
    checks[1]!.click();
    flushSync();
    expect(predPre.textContent).toContain('--thresholds fasteners-n-best-thresholds.json');

    // CheckpointCli toggle for both Validate and Predict modes (plus `--split test`)
    view.unmount();
    const api = new MockApi('normal', 0);
    const run = (await api.runs()).find((r) => r.status === 'completed' && r.model_id)!;
    const bundle = buildCheckpointCli(run, model, undefined, { datasetId: 'ds_1', datasetName: 'fasteners', split: 'test', fallback: null })!;
    expect(bundle.val).toContain('--split test');
    view = render(() => <CheckpointCli bundle={bundle} />, target);
    flushSync();
    target.querySelector<HTMLButtonElement>('.ckpt-toggle')!.click();
    flushSync();
    const ckptCheck = target.querySelector<HTMLInputElement>('.ckpt-check input[type="checkbox"]')!;
    expect(target.querySelector('pre.line')?.textContent).toBe(bundle.val);
    ckptCheck.click();
    flushSync();
    expect(target.querySelector('pre.line')?.textContent).toBe(bundle.valThresholds);
    expect(bundle.valThresholds).toContain('--save-thresholds thresholds.json');

    const predictBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.trim() === 'Predict')!;
    predictBtn.click();
    flushSync();
    expect(target.querySelector('pre.line')?.textContent).toBe(bundle.predictThresholds);
    expect(bundle.predictThresholds).toContain('--thresholds thresholds.json');
  });
});

describe('Model Detail -> Playground handoff, TTL/scope, and Back/Forward/reload semantics', () => {
  it('enforces handoff TTL, clears on server switch, and refuses plain /playground without a matching modelId', () => {
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(1_000_000);
    stageThresholdHandoff({ modelId: 'm_a', globalConf: 0.42, perClassConf: { 0: 0.5 }, classes: ['bolt'] });
    // Plain /playground (modelId === null) must NOT consume or hijack, and clears the stale handoff.
    expect(consumeThresholdHandoff(null)).toBeNull();
    expect(consumeThresholdHandoff('m_a')).toBeNull();

    // Server switch clears the handoff so returning to the original server never resurrects it.
    stageThresholdHandoff({ modelId: 'm_a', globalConf: 0.42, perClassConf: { 0: 0.5 }, classes: ['bolt'] });
    hub.mode = 'live';
    expect(consumeThresholdHandoff('m_a')).toBeNull();
    hub.mode = 'mock';
    expect(consumeThresholdHandoff('m_a')).toBeNull();

    // Expired handoff past HANDOFF_TTL_MS returns null.
    stageThresholdHandoff({ modelId: 'm_a', globalConf: 0.42, perClassConf: { 0: 0.5 }, classes: ['bolt'] });
    now.mockReturnValue(1_000_000 + HANDOFF_TTL_MS + 1);
    expect(consumeThresholdHandoff('m_a')).toBeNull();
  });

  it('downloads measured ThresholdCalibration JSON, hands off cleanly into Playground, and handles Back/Forward and reload with default conf', async () => {
    const navCalls: string[] = [];
    vi.spyOn(router, 'go').mockImplementation((path) => {
      navCalls.push(path);
      router.route = parse(`#${path}`);
    });

    // Seed an unrelated prior session with different conf & per-class overrides to verify handoff & reload override them.
    session.set({
      source: null,
      pred: null,
      runOn: 'server',
      conf: 0.72,
      iou: 0.7,
      maxDet: 300,
      perClassConf: { 0: 0.85 },
      perClassModelId: 'm_other',
      perClassClasses: ['cat', 'dog'],
      labels: true,
      scores: true,
      truth: false,
      hidden: [],
    });

    router.route = parse('#/models/m_fast_n');
    view = render(() => <ModelPage />, target);
    const bar = await until(() => target.querySelector('[data-model-thresholds]'), 'thresholds bar');
    expect(bar.textContent).toContain('Operating point');
    expect(bar.textContent).toContain('macro-F1');

    // Download thresholds.json and verify payload structure
    bar.querySelector<HTMLButtonElement>('[data-download-thresholds]')!.click();
    await until(() => downloads.length === 1, 'thresholds download');
    expect(downloads[0]!.filename).toMatch(/-thresholds\.json$/);
    const model = await hub.api.model('m_fast_n');
    const ev = fromModelEval(await hub.api.modelEval('m_fast_n'));
    const expected = buildModelThresholds(model, ev)!;
    const parsedFile = JSON.parse(downloads[0]!.text);
    expect(parsedFile.global.best_conf).toBe(expected.payload.global.best_conf);
    expect(parsedFile.per_class).toHaveLength(expected.payload.per_class.length);
    const pccParsed = parseThresholdsJson(downloads[0]!.text, model.classes);
    expect(pccParsed.ok).toBe(true);

    // Click "Apply in Playground" -> URL explicitly carries conf & pcc=
    bar.querySelector<HTMLButtonElement>('[data-apply-playground]')!.click();
    flushSync();
    expect(navCalls).toHaveLength(1);
    expect(navCalls[0]).toContain('/playground?model=m_fast_n');
    expect(navCalls[0]).toContain(`conf=${expected.payload.global.best_conf}`);
    expect(navCalls[0]).toContain('pcc=');

    const qs = new URLSearchParams(navCalls[0]!.split('?')[1]);
    expect(perClassFromUrl(qs)).toEqual({});
    expect(serializePerClassParam({ 0: 0.45 })).toBe('0:0.45');
    // Threshold-bearing link with default conf 0.25 still explicitly serializes conf=0.25.
    const defLink = playgroundPath({ model: 'm_fast_n', source: null, conf: 0.25, iou: 0.7, maxDet: 300, perClassConf: { 0: 0.45 } });
    expect(defLink).toContain('conf=0.25');
    expect(defLink).toContain('pcc=0%3A0.45');

    // Mount Playground -> handoff applies measured conf, clears stale m_other overrides, and tags session origin as m_fast_n (not m_other).
    view.unmount();
    view = render(() => <Playground />, target);
    await until(() => toast.items.some((t) => t.title === 'Calibrated thresholds applied'), 'handoff toast');
    expect(target.querySelector('[data-per-class-badge]')?.textContent?.trim()).toBe('Default');
    expect(session.get()?.conf).toBe(expected.payload.global.best_conf);
    expect(session.get()?.perClassModelId).toBe('m_fast_n');

    // Navigate Forward to a URL with per-class overrides (`?pcc=0:0.45`), then Back to a URL without `pcc`:
    // Back MUST clear per-class overrides rather than retaining them!
    router.route = parse('#/playground?model=m_fast_n&conf=0.25&pcc=0:0.45');
    await until(() => target.querySelector('[data-per-class-badge]')?.textContent?.trim() === '1 override', 'forward pcc badge');

    router.route = parse('#/playground?model=m_fast_n');
    await until(() => target.querySelector('[data-per-class-badge]')?.textContent?.trim() === 'Default', 'back clears pcc badge');
    expect(session.get()?.perClassConf).toEqual({});

    // Fresh mount ("reload" / revisit after one-shot handoff was consumed) on an explicit default-conf link (`conf=0.25&pcc=`):
    // must NOT inherit unrelated saved conf (0.80) or saved perClassConf ({ 0: 0.90 }).
    view.unmount();
    session.set({
      source: null,
      pred: null,
      runOn: 'server',
      conf: 0.80,
      iou: 0.7,
      maxDet: 300,
      perClassConf: { 0: 0.90 },
      perClassModelId: 'm_other',
      perClassClasses: ['cat'],
      labels: true,
      scores: true,
      truth: false,
      hidden: [],
    });
    router.route = parse('#/playground?model=m_fast_n&conf=0.25&pcc=');
    view = render(() => <Playground />, target);
    await until(() => session.get()?.perClassModelId === 'm_fast_n', 'explicit default link reload');
    expect(session.get()?.conf).toBe(0.25);
    expect(session.get()?.perClassConf).toEqual({});
    expect(target.querySelector('[data-per-class-badge]')?.textContent?.trim()).toBe('Default');
  });
});
