import { describe, expect, it } from 'vitest';
import type { ScoreContext } from '$lib/scoreSummary';
import { bestNote, unrankedKinds } from './bestNote';

const ctx = (kind: ScoreContext['kind']): ScoreContext => ({ kind, datasetId: null, datasetName: null, split: null });

describe('bestNote', () => {
  it('says why each model is left out: past the sixth dataset, trained elsewhere, split unknown', () => {
    const why = 'evaluation split unknown';
    expect(bestNote({ more: 0, elsewhere: 0, unknown: 0, why })).toBe('');
    // The README's first model, copied into the models folder, on a fresh data dir (studio-cloud handover).
    expect(bestNote({ more: 0, ...unrankedKinds([ctx('imported'), ctx('published')]), why })).toBe(
      '2 models trained elsewhere (imported or downloaded) are not ranked.',
    );
    expect(bestNote({ more: 14, ...unrankedKinds([ctx('published'), ctx('run')]), why })).toBe(
      '14 more datasets in All models. 1 model trained elsewhere (imported or downloaded) is not ranked. 1 model not ranked: evaluation split unknown.',
    );
    expect(bestNote({ more: 1, elsewhere: 0, unknown: 2, why: 'could not load runs' })).toBe('1 more dataset in All models. 2 models not ranked: could not load runs.');
  });
});
