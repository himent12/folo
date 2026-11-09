import { describe, expect, it } from 'vitest';
import { startFrom } from './RunCompare.svelte';
import { lrParts } from './run/RunMetrics.svelte';

describe('Compare: startFrom', () => {
  const models = [{ id: 'm_shapes_n', name: 'shapes-n' }];
  it('names the model a run was fine-tuned from, or says it started from scratch', () => {
    expect(startFrom('m_shapes_n', models)).toBe('shapes-n');
    expect(startFrom(null, models)).toBeNull();
    expect(startFrom('', models)).toBeNull();
  });
  it('shows a weights path by its file name, and an unknown id as given', () => {
    expect(startFrom('D:/runs/shapes-n/best.folo', models)).toBe('best.folo');
    expect(startFrom('C:\\runs\\x\\last.folo', models)).toBe('last.folo');
    expect(startFrom('m_gone', undefined)).toBe('m_gone');
  });
});

describe('Run metrics: lrParts', () => {
  it('splits a learning rate into a short value and its power of ten', () => {
    expect(lrParts(3.42e-4)).toEqual({ value: '3.42', unit: '× 10⁻⁴', exp: '−4' });
    expect(lrParts(0.01)).toEqual({ value: '1.00', unit: '× 10⁻²', exp: '−2' });
    expect(lrParts(25)).toEqual({ value: '2.50', unit: '× 10¹', exp: '1' });
    expect(lrParts(null)).toEqual({ value: '—' });
    expect(lrParts(0)).toEqual({ value: '0' });
  });
});
