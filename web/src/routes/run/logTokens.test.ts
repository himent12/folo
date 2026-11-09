import { describe, expect, it } from 'vitest';
import { logPieces } from './logTokens';

const words = (m: string) => logPieces(m).filter((p) => !p.space).map((p) => p.text);

describe('logPieces', () => {
  it('keeps each metric with its value', () => {
    expect(words('epoch 141/150  loss 2.221  mAP50 0.781  mAP50-95 0.588  21.4s')).toEqual([
      'epoch 141/150',
      'loss 2.221',
      'mAP50 0.781',
      'mAP50-95 0.588',
      '21.4s',
    ]);
  });

  it('keeps a flag with its argument', () => {
    expect(words('consider --workers 12')).toEqual(['consider', '--workers 12']);
    expect(words('run with --amp --batch 16')).toEqual(['run', 'with', '--amp', '--batch 16']);
  });

  it('keeps the text as written, spaces included', () => {
    const m = 'data loader stalled 140 ms waiting for decode workers (8 threads); consider --workers 12';
    expect(logPieces(m).map((p) => p.text).join('')).toBe(m);
    expect(logPieces('a  b').map((p) => p.text)).toEqual(['a', '  ', 'b']);
  });

  it('marks only pieces too long for a line as breakable', () => {
    const path = 'D:/folo/runs/fasteners-n/weights/best-epoch-150.folo';
    expect(logPieces(`saved ${path}`).find((p) => p.text === path)?.long).toBe(true);
    expect(logPieces('mAP50-95 0.588').every((p) => !p.long)).toBe(true);
  });
});
