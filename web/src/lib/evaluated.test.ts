import { describe, expect, it } from 'vitest';
import type { ModelEval } from '$lib/api/types';
import {
  evalSource,
  evaluatedOn,
  fallbackNotice,
  imagesOf,
  matrixAt,
  matrixAtText,
  matrixCounted,
  matrixVsPr,
  metricsLabel,
  sourceShort,
  TRAINED_ON,
  type EvalSource,
} from './evaluated';

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
  it('says plainly that the numbers are on the training split, and why', () => {
    expect(fallbackNotice(fellBack)).toEqual({
      title: 'Evaluated on the training split',
      why: `fasteners-mini has no validation images, so these scores are ${TRAINED_ON}.`,
      text: 'Evaluated on the training split: fasteners-mini has no validation images, so these scores are on images the model trained on, not new ones.',
    });
  });
  it('is absent on the validation split and when the server does not say', () => {
    expect(fallbackNotice(val)).toBeNull();
    expect(fallbackNotice(unknown)).toBeNull();
  });
  it('names a dataset it does not know as "this dataset", capitalised as a sentence of its own', () => {
    const n = fallbackNotice({ ...fellBack, datasetName: null })!;
    expect(n.why).toBe(`This dataset has no validation images, so these scores are ${TRAINED_ON}.`);
    expect(n.text).toBe(`Evaluated on the training split: this dataset has no validation images, so these scores are ${TRAINED_ON}.`);
  });
  it('treats an unknown reason as a fallback too, in the server\'s words', () => {
    const n = fallbackNotice({ ...fellBack, fallback: { requested: 'val', reason: 'unreadable', message: 'val labels could not be read' } })!;
    expect(n.title).toBe('Evaluated on the training split');
    expect(n.why).toBe(`Not the validation split (val labels could not be read), so these scores are ${TRAINED_ON}.`);
  });
  it('says so for a training split the server names without a fallback', () => {
    expect(fallbackNotice({ ...val, split: 'train' })?.text).toBe(`Evaluated on the training split: not the validation split, so these scores are ${TRAINED_ON}.`);
  });
});

describe('what was evaluated', () => {
  it('names the dataset and the split actually used, never "validation" for training numbers', () => {
    expect(imagesOf(val, 360)).toBe('360 validation images of fasteners');
    expect(imagesOf(fellBack, 24)).toBe('24 training images of fasteners-mini');
    expect(imagesOf(unknown, 1500)).toBe('1,500 images');
    expect(imagesOf({ ...val, split: null }, 1)).toBe('1 image of fasteners');
    expect(sourceShort(val)).toBe('fasteners · validation split');
    expect(sourceShort(fellBack)).toBe('fasteners-mini · training split');
    expect(sourceShort(unknown)).toBeNull();
    expect(evaluatedOn(val, 360)).toBe('The validation split of fasteners (360 images)');
    expect(evaluatedOn(fellBack, null)).toBe('The training split of fasteners-mini');
    expect(evaluatedOn({ ...val, split: null }, 500)).toBe('fasteners, split not reported (500 images)');
    expect(metricsLabel(val)).toBe('Validation metrics');
    expect(metricsLabel(fellBack)).toBe('Metrics on the training split');
    expect(metricsLabel(unknown)).toBe('Evaluation metrics');
  });
});

describe("the confusion matrix's thresholds", () => {
  it('are its own, stated as such, apart from the precision / recall operating point', () => {
    const m = matrixAt({ ...base, confusion_conf: 0.25, confusion_iou: 0.45 });
    expect(m).toEqual({ conf: 0.25, iou: 0.45 });
    expect(matrixAtText(m)).toBe('at conf 0.25, IoU 0.45');
    expect(matrixCounted(val, 360, m)).toBe('Counted at conf 0.25, IoU 0.45 on 360 validation images of fasteners.');
    const how = matrixVsPr(m, 'at conf 0.31 (best F1)');
    expect(how).toContain('confidence is above 0.25');
    expect(how).toContain('IoU above 0.45');
    expect(how).toContain('Precision and recall are read at conf 0.31 (best F1), IoU 0.50');
  });
  it('are said to be unreported when the server leaves them out, never borrowed from precision and recall', () => {
    expect(matrixAt(base)).toBeNull();
    expect(matrixAt({ ...base, confusion_conf: 0.25, confusion_iou: null })).toBeNull();
    expect(matrixCounted(unknown, 500, null)).toBe('Counted on 500 images, at thresholds this server does not report.');
    const how = matrixVsPr(null, 'at conf 0.31 (best F1)');
    expect(how).toContain('which this server does not report');
    expect(how).not.toMatch(/0\.25|0\.45/);
  });
});
