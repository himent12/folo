// @vitest-environment happy-dom
// Tests for Model Detail operating point + server `POST /api/val` per-class threshold calibration,
// ThresholdCalibration JSON export & Playground handoff (`Apply in Playground`, Back/Forward/reload,
// oversized >1200-char perClassConf preservation, handoff TTL/scope, session origin attribution),
// plus `folo val --save-thresholds` & `folo predict --thresholds` CLI builders.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import type { Evaluation } from '$lib/api/evaluation';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Model, ThresholdCalibration } from '$lib/api/types';
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
import { buildModelOperatingPoint, formatServerCalibration } from './thresholds';

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

describe('buildModelOperatingPoint & formatServerCalibration', () => {
  it('requires a valid /eval bestF1Conf and never substitutes checkpoint metrics', () => {
    const ev: Evaluation = {
      perClass: [
        { class: 0, ap50: 0.90, ap50_95: 0.70, precision: 0.75, recall: 0.90, instances: 20 },
        { class: 1, ap50: 0.82, ap50_95: 0.58, precision: 0.90, recall: 0.50, instances: 10 },
        { class: 2, ap50: 0, ap50_95: 0, precision: 0, recall: 0, instances: 0 },
      ],
      curves: [[], [], []],
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

    const op = buildModelOperatingPoint(ev)!;
    expect(op).toEqual({ globalConf: 0.4, macroF1: expect.any(Number), classCount: 2 });
    expect(buildModelOperatingPoint({ ...ev, bestF1Conf: null })).toBeNull();
    expect(buildModelOperatingPoint({ ...ev, perClass: [{ class: 0, ap50: 0, ap50_95: 0, precision: 0, recall: 0, instances: 0 }] })).toBeNull();
  });

  it('formats server POST /api/val ThresholdCalibration with evaluator-provided per-class thresholds and preds', () => {
    const raw: ThresholdCalibration = {
      global: { best_conf: 0.38, best_f1: 0.84, precision: 0.86, recall: 0.82 },
      macro_f1: 0.87,
      per_class: [
        // Out of order + zero-TP class with real false-positive preds count (142) preserved verbatim.
        { class: 1, best_conf: 0.52, best_f1: 0, precision: 0, recall: 0, truths: 10, preds: 142 },
        { class: 0, best_conf: 0.28, best_f1: 0.91, precision: 0.92, recall: 0.90, truths: 20, preds: 35 },
      ],
    };

    const cal = formatServerCalibration(raw, ['bolt', 'nut'])!;
    expect(cal).not.toBeNull();
    expect(cal.payload.global.best_conf).toBe(0.38);
    expect(cal.payload.macro_f1).toBe(0.87);
    expect(cal.payload.per_class.map((c) => c.class)).toEqual([0, 1]);
    expect(cal.payload.per_class.map((c) => c.name)).toEqual(['bolt', 'nut']);
    expect(cal.payload.per_class[0]!.best_conf).toBe(0.28);
    expect(cal.payload.per_class[1]!.preds).toBe(142);
    expect(cal.perClassConf).toEqual({ 0: 0.28, 1: 0.52 });

    const parsed = parseThresholdsJson(JSON.stringify(cal.payload), ['bolt', 'nut']);
    expect(parsed.ok).toBe(true);
    if (parsed.ok) {
      expect(parsed.globalConf).toBe(0.38);
      expect(parsed.perClassConf).toEqual({ 0: 0.28, 1: 0.52 });
    }
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
    checks[0]!.click();
    flushSync();
    expect(valPre.textContent).not.toContain('--save-thresholds');
    checks[1]!.click();
    flushSync();
    expect(predPre.textContent).toContain('--thresholds fasteners-n-best-thresholds.json');

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

describe('Model Detail -> Playground calibration, handoff, history, and oversized map preservation', () => {
  it('enforces handoff TTL, clears on server switch, and refuses plain /playground without a matching modelId', () => {
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(1_000_000);
    stageThresholdHandoff({ modelId: 'm_a', globalConf: 0.42, perClassConf: { 0: 0.5 }, classes: ['bolt'] });
    expect(consumeThresholdHandoff(null)).toBeNull();
    expect(consumeThresholdHandoff('m_a')).toBeNull();

    stageThresholdHandoff({ modelId: 'm_a', globalConf: 0.42, perClassConf: { 0: 0.5 }, classes: ['bolt'] });
    hub.mode = 'live';
    expect(consumeThresholdHandoff('m_a')).toBeNull();
    hub.mode = 'mock';
    expect(consumeThresholdHandoff('m_a')).toBeNull();

    stageThresholdHandoff({ modelId: 'm_a', globalConf: 0.42, perClassConf: { 0: 0.5 }, classes: ['bolt'] });
    now.mockReturnValue(1_000_000 + HANDOFF_TTL_MS + 1);
    expect(consumeThresholdHandoff('m_a')).toBeNull();
  });

  it('calibrates per-class thresholds via POST /api/val, downloads ThresholdCalibration JSON, hands off to Playground, and handles Back/Forward and reload', async () => {
    const navCalls: string[] = [];
    vi.spyOn(router, 'go').mockImplementation((path) => {
      navCalls.push(path);
      router.route = parse(`#${path}`);
    });

    router.route = parse('#/models/m_fast_n');
    view = render(() => <ModelPage />, target);
    const bar = await until(() => target.querySelector('[data-model-thresholds="operating-point"]'), 'operating point bar');
    expect(bar.textContent).toContain('Operating point');

    // Download triggers POST /api/val calibration on demand, caches exact server ThresholdCalibration, and saves JSON.
    const dlBtn = await until(() => target.querySelector<HTMLButtonElement>('[data-download-thresholds]'), 'download button');
    dlBtn.click();
    await until(() => downloads.length === 1, 'thresholds download');
    await until(() => target.querySelector('[data-model-thresholds="calibrated"]'), 'calibrated bar state');
    expect(downloads[0]!.filename).toMatch(/-thresholds\.json$/);

    const model = await hub.api.model('m_fast_n');
    const parsedFile = JSON.parse(downloads[0]!.text);
    expect(parsedFile.per_class.length).toBeGreaterThan(0);
    expect(parsedFile.per_class.every((c: { preds: number }) => Number.isInteger(c.preds) && c.preds > 0)).toBe(true);
    const pccParsed = parseThresholdsJson(downloads[0]!.text, model.classes);
    expect(pccParsed.ok).toBe(true);

    // Click "Apply in Playground" -> URL carries calibrated conf & server per-class pcc overrides.
    target.querySelector<HTMLButtonElement>('[data-apply-playground]')!.click();
    flushSync();
    expect(navCalls).toHaveLength(1);
    expect(navCalls[0]).toContain('/playground?model=m_fast_n');
    expect(navCalls[0]).toContain(`conf=${parsedFile.global.best_conf}`);
    expect(navCalls[0]).toContain('pcc=');

    const qs = new URLSearchParams(navCalls[0]!.split('?')[1]);
    expect(Object.keys(perClassFromUrl(qs) ?? {})).toHaveLength(parsedFile.per_class.length);
    expect(serializePerClassParam({ 0: 0.45 })).toBe('0:0.45');
    // Threshold-bearing link with default conf 0.25 still explicitly serializes conf=0.25.
    const defLink = playgroundPath({ model: 'm_fast_n', source: null, conf: 0.25, iou: 0.7, maxDet: 300, perClassConf: { 0: 0.45 } });
    expect(defLink).toContain('conf=0.25');
    expect(defLink).toContain('pcc=0%3A0.45');

    // Mount Playground -> handoff applies server calibrated per-class thresholds.
    view.unmount();
    view = render(() => <Playground />, target);
    const badge = await until(
      () => target.querySelector('[data-per-class-badge]')?.textContent?.trim() === `${parsedFile.per_class.length} overrides`,
      'per-class badge',
    );
    expect(badge).toBe(true);
    expect(toast.items.some((t) => t.title === 'Calibrated thresholds applied')).toBe(true);

    // Navigate Back to a URL omitting `pcc` -> clears per-class overrides.
    router.route = parse('#/playground?model=m_fast_n');
    await until(() => target.querySelector('[data-per-class-badge]')?.textContent?.trim() === 'Default', 'back clears pcc');
    expect(session.get()?.perClassConf).toEqual({});

    // Fresh mount on explicit default-conf link (`conf=0.25&pcc=`) ignores unrelated saved conf & perClassConf.
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
  });

  it('preserves oversized (>1200 char) per-class threshold imports across navigate away and back instead of writing pcc=', async () => {
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    // Build a 160-class override map that exceeds MAX_PCC_URL_LEN (1200 chars).
    const bigMap: Record<number, number> = {};
    for (let i = 0; i < 160; i++) bigMap[i] = Math.round((0.1 + (i % 80) * 0.01) * 100) / 100;
    expect(serializePerClassParam(bigMap)).toBeNull();
    // Even with explicitPerClass: true, playgroundPath MUST NOT emit `pcc=` for a non-empty oversized map.
    const bigPath = playgroundPath({
      model: 'm_fast_n',
      source: null,
      conf: 0.25,
      iou: 0.7,
      maxDet: 300,
      perClassConf: bigMap,
      explicitConf: true,
      explicitPerClass: true,
    });
    expect(bigPath).not.toContain('pcc=');
    expect(bigPath).toContain('conf=0.25');

    // Start from an explicit empty handoff URL (`pcc=`), import the 160-override map, let debounced URL sync run,
    // unmount (navigate away), and reopen Playground -> all 160 overrides MUST survive via tab session!
    router.route = parse('#/playground?model=m_fast_n&conf=0.25&pcc=');
    view = render(() => <Playground />, target);
    await until(() => target.querySelector('button.pc-toggle'), 'per-class toggle');
    target.querySelector<HTMLButtonElement>('button.pc-toggle')!.click();
    flushSync();
    target.querySelector<HTMLButtonElement>('[data-per-class-import-btn]')!.click();
    flushSync();
    const ta = target.querySelector<HTMLTextAreaElement>('textarea.pc-textarea')!;
    ta.value = JSON.stringify(bigMap);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    target.querySelector<HTMLButtonElement>('[data-per-class-apply]')!.click();
    await until(() => Object.keys(session.get()?.perClassConf ?? {}).length === 160, '160 overrides applied');

    // Wait past URL_DELAY_MS (300ms) so keepSession writes the URL.
    await new Promise((r) => setTimeout(r, 360));
    flushSync();
    expect(router.route.query.has('pcc')).toBe(false);

    // Navigate away (unmount) and reopen that exact route -> 160 overrides MUST still be restored!
    view.unmount();
    view = render(() => <Playground />, target);
    await until(() => target.querySelector('[data-per-class-badge]')?.textContent?.trim() === '160 overrides', '160 overrides restored');
    expect(Object.keys(session.get()?.perClassConf ?? {})).toHaveLength(160);
  });
});
