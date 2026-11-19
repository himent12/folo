import { describe, expect, it } from 'vitest';
import type { ModelEval } from '$lib/api/types';
import { evalSource, fallbackNotice, imagesOf, matrixAt, matrixVsPr, runEvalSource, scoresTitle, type EvalSource } from './evaluated';

const base: ModelEval = { pr: [], confusion: [], per_class: [], images: 360 };
const val: EvalSource = { datasetId: 'ds_fasteners', datasetName: 'fasteners', split: 'val', fallback: null };
const fellBack: EvalSource = {
  datasetId: 'ds_fasteners_mini',
  datasetName: 'fasteners-mini',
  split: 'train',
  fallback: { requested: 'val', reason: 'empty_split', message: "dataset 'fasteners-mini' has no val images, so this is its train split (24 images): training-set numbers, not held-out ones" },
};
const unknown: EvalSource = { datasetId: null, datasetName: null, split: null, fallback: null };

describe('evalSource', () => {
  it('reads the provenance, and missing fields (older servers) as unknown, never as val', () => {
    expect(evalSource(base)).toEqual(unknown);
    expect(evalSource({ ...base, dataset_id: 'ds_fasteners', dataset_name: 'fasteners', split: 'val', split_fallback: null })).toEqual(val);
  });
  it('takes a fallback without a split for the training split', () => {
    expect(evalSource({ ...base, split_fallback: fellBack.fallback }).split).toBe('train');
  });
});

describe('the fallback notice', () => {
  it('warns of training-split numbers whatever the reason, the dataset known or not, and nowhere else', () => {
    const sources: EvalSource[] = [
      fellBack,
      { ...fellBack, datasetName: null },
      { ...fellBack, fallback: { requested: 'val', reason: 'unreadable', message: 'val labels could not be read' } },
      // A training split the server names without a fallback.
      { ...val, split: 'train' },
    ];
    for (const s of sources)
      expect(fallbackNotice(s)?.text, JSON.stringify(s)).toMatch(/^Evaluated on the training split: /);
    expect(fallbackNotice(fellBack)!.why).toMatch(/^fasteners-mini has no validation images/);
    expect(fallbackNotice({ ...fellBack, datasetName: null })!.why).toMatch(/^This dataset has no validation images/);
    expect(fallbackNotice(val)).toBeNull();
    expect(fallbackNotice(unknown)).toBeNull();
  });
});

describe('what was evaluated', () => {
  it('names the dataset and the split actually used, never "validation" for training numbers', () => {
    expect(imagesOf(val, 360)).toBe('360 validation images of fasteners');
    expect(imagesOf(fellBack, 24)).toBe('24 training images of fasteners-mini');
    expect(imagesOf(unknown, 1500)).toBe('1,500 images');
    expect(imagesOf({ ...val, split: null }, 1)).toBe('1 image of fasteners');
  });
});

describe("the confusion matrix's thresholds", () => {
  it('are its own, never borrowed from precision and recall when the server leaves them out', () => {
    const m = matrixAt({ ...base, confusion_conf: 0.25, confusion_iou: 0.45 });
    expect(m).toEqual({ conf: 0.25, iou: 0.45 });
    expect(matrixAt(base)).toBeNull();
    expect(matrixAt({ ...base, confusion_conf: 0.25, confusion_iou: null })).toBeNull();
    expect(matrixVsPr(m, 'at conf 0.31 (best F1)')).toContain('confidence is above 0.25');
    expect(matrixVsPr(null, 'at conf 0.31 (best F1)')).not.toMatch(/0\.25|0\.45/);
  });
});

describe('runEvalSource / scoresTitle', () => {
  it("names the training split where a run's dataset has no validation images", () => {
    const mini = runEvalSource({ id: 'ds_m', name: 'fasteners-mini', splits: { train: 24, val: 0, test: 0 } });
    expect(mini).toMatchObject({ split: 'train', fallback: { requested: 'val', reason: 'empty_split' } });
    expect(scoresTitle(mini)).toBe('Training-split');
    const full = runEvalSource({ id: 'ds_f', name: 'fasteners', splits: { train: 1920, val: 360, test: 120 } });
    expect(scoresTitle(full)).toBe('Validation');
    expect(fallbackNotice(full!)).toBeNull();
    // Unknown provenance (the dataset is not registered, or not loaded yet) is never called validation.
    expect(scoresTitle(runEvalSource(undefined))).toBe('Evaluation');
    expect(scoresTitle(null)).toBe('Evaluation');
    expect(scoresTitle({ datasetId: null, datasetName: null, split: null, fallback: null })).toBe('Evaluation');
    expect(scoresTitle({ datasetId: 'ds_t', datasetName: 'test-only', split: 'test', fallback: null })).toBe('Test-split');
  });
});
