import { describe, expect, it } from 'vitest';
import type { LogLine } from '$lib/api/types';
import { scoresTitle } from '$lib/evaluated';
import { goneEvalOn } from './goneEvalOn';

const line = (level: LogLine['level'], message: string): LogLine => ({ t: '2026-11-11T10:00:00Z', level, message });

describe('goneEvalOn', () => {
  it('keeps "Validation" for a removed dataset unless the trainer said it validated on the training split', () => {
    expect(scoresTitle(goneEvalOn('bccd', [line('info', 'epoch 1/3: loss 13.998')]))).toBe('Validation');
    const onTrain = goneEvalOn('bccd', [line('warn', 'the dataset has no validation images; validating on the training split')]);
    expect(scoresTitle(onTrain)).toBe('Training-split');
    expect(onTrain.fallback?.message).toBe('bccd has no validation images');
  });
});
