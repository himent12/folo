import { describe, expect, it } from 'vitest';
import { lineageOf } from './lineage';

describe('lineageOf', () => {
  it('reads no weights as trained from scratch', () => {
    expect(lineageOf(null)).toEqual({ kind: 'scratch' });
    expect(lineageOf('')).toEqual({ kind: 'scratch' });
  });

  it('reads a model id as fine-tuned from that model', () => {
    expect(lineageOf('m_shapes_n')).toEqual({ kind: 'model', id: 'm_shapes_n' });
  });

  it('reads a path as fine-tuned from an imported file, named by its file name', () => {
    expect(lineageOf('D:/weights/coco-n.folo')).toEqual({ kind: 'file', path: 'D:/weights/coco-n.folo', name: 'coco-n.folo' });
    expect(lineageOf(String.raw`C:\w\best.folo`)).toMatchObject({ kind: 'file', name: 'best.folo' });
    expect(lineageOf('pretrained.folo')).toMatchObject({ kind: 'file', name: 'pretrained.folo' });
  });
});
