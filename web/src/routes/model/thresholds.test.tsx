// @vitest-environment happy-dom
// Tests for calibrated per-class F1 thresholds on Model Detail, ThresholdCalibration JSON export &
// Playground handoff (`Apply in Playground`), plus `folo val --save-thresholds` & `folo predict --thresholds`
// CLI builders in Model ExportDialog and Run CheckpointCli.
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
import { perClassFromUrl, playgroundPath, serializePerClassParam, session } from '../playground/session';
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
  toast.items = [];
  vi.restoreAllMocks();
});

describe('buildModelThresholds', () => {
  it('produces ThresholdCalibration JSON compatible with folo predict --thresholds and Playground parseThresholdsJson', () => {
    const model: Pick<Model, 'classes' | 'metrics'> = {
      classes: ['bolt', 'nut', 'washer'],
      metrics: { map50: 0.85, map50_95: 0.62, precision: 0.84, recall: 0.80, best_f1_conf: 0.40 },
    };
    const ev: Evaluation = {
      perClass: [
        // Class 0 peaks at lower recall / higher precision -> threshold should rise above 0.40
        { class: 0, ap50: 0.90, ap50_95: 0.70, precision: 0.72, recall: 0.90, instances: 20 },
        // Class 1 peaks at higher recall -> threshold should drop below 0.40
        { class: 1, ap50: 0.82, ap50_95: 0.58, precision: 0.92, recall: 0.50, instances: 10 },
        // Class 2 has zero ground truth -> excluded from per_class (matching folo-vision)
        { class: 2, ap50: 0, ap50_95: 0, precision: 0, recall: 0, instances: 0 },
      ],
      curves: [
        [
          { x: 0.5, y: 0.96 },
          { x: 0.7, y: 0.94 },
          { x: 0.9, y: 0.72 },
        ],
        [
          { x: 0.5, y: 0.92 },
          { x: 0.85, y: 0.86 },
          { x: 0.95, y: 0.60 },
        ],
        [],
      ],
      mean: [],
      confusion: [
        [19, 0, 0, 1],
        [0, 9, 0, 1],
        [0, 0, 0, 0],
        [3, 1, 0, 0],
      ],
      images: 16,
      bestF1Conf: 0.40,
      source: { datasetId: 'ds_1', datasetName: 'fasteners', split: 'val', fallback: null },
      matrixAt: { conf: 0.25, iou: 0.45 },
    };

    const cal = buildModelThresholds(model, ev)!;
    expect(cal).not.toBeNull();
    expect(cal.payload.global.best_conf).toBe(0.4);
    expect(cal.payload.per_class.map((c) => c.class)).toEqual([0, 1]);
    expect(cal.payload.per_class.map((c) => c.name)).toEqual(['bolt', 'nut']);
    expect(cal.payload.per_class[0]!.best_conf).toBeGreaterThan(0.4);
    expect(cal.payload.per_class[1]!.best_conf).toBeLessThan(0.4);
    expect(cal.payload.per_class[0]!.truths).toBe(20);
    expect(cal.payload.per_class[0]!.preds).toBe(22);
    expect(cal.payload.macro_f1).toBeGreaterThanOrEqual(cal.payload.global.best_f1);

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
      expect(parsed.perClassConf[0]).toBe(cal.payload.per_class[0]!.best_conf);
      expect(parsed.perClassConf[1]).toBe(cal.payload.per_class[1]!.best_conf);
    }
  });

  it('returns null when no evaluated classes have ground-truth instances', () => {
    const ev: Evaluation = {
      perClass: [{ class: 0, ap50: 0, ap50_95: 0, precision: 0, recall: 0, instances: 0 }],
      curves: [[]],
      mean: [],
      confusion: [
        [0, 0],
        [0, 0],
      ],
      images: 0,
      bestF1Conf: null,
      source: { datasetId: null, datasetName: null, split: null, fallback: null },
      matrixAt: null,
    };
    expect(buildModelThresholds({ classes: ['a'], metrics: null }, ev)).toBeNull();
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

  it('builds folo val --save-thresholds and folo predict --thresholds commands with safe quoting', () => {
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

    // CheckpointCli toggle for both Validate and Predict modes
    view.unmount();
    const api = new MockApi('normal', 0);
    const run = (await api.runs()).find((r) => r.status === 'completed' && r.model_id)!;
    const bundle = buildCheckpointCli(run, model, undefined, null)!;
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

describe('Model Detail -> Playground calibrated thresholds handoff', () => {
  it('downloads ThresholdCalibration JSON and hands off calibrated per-class thresholds into Playground', async () => {
    const navCalls: string[] = [];
    vi.spyOn(router, 'go').mockImplementation((path) => {
      navCalls.push(path);
      router.route = parse(`#${path}`);
    });

    router.route = parse('#/models/m_fast_n');
    view = render(() => <ModelPage />, target);
    const bar = await until(() => target.querySelector('[data-model-thresholds]'), 'thresholds bar');
    expect(bar.textContent).toContain('Calibrated thresholds');
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

    // Click "Apply in Playground" and mount Playground at the handed-off URL
    bar.querySelector<HTMLButtonElement>('[data-apply-playground]')!.click();
    flushSync();
    expect(navCalls).toHaveLength(1);
    expect(navCalls[0]).toContain('/playground?model=m_fast_n');
    expect(navCalls[0]).toContain('pcc=');

    // URL param round-trip check
    const qs = new URLSearchParams(navCalls[0]!.split('?')[1]);
    expect(perClassFromUrl(qs)).toEqual(expected.perClassConf);
    expect(serializePerClassParam(expected.perClassConf)).toBe(qs.get('pcc'));
    expect(playgroundPath({ model: 'm_fast_n', source: null, conf: 0.4, iou: 0.7, maxDet: 300, perClassConf: { 0: 0.45 } })).toContain('pcc=0%3A0.45');

    view.unmount();
    view = render(() => <Playground />, target);
    const badge = await until(() => target.querySelector('[data-per-class-badge]'), 'per-class badge');
    expect(badge.textContent?.trim()).toBe(`${expected.payload.per_class.length} overrides`);
    expect(toast.items.some((t) => t.title === 'Calibrated thresholds applied')).toBe(true);
  });
});
