import { describe, expect, it } from 'vitest';
import { landed, type Generation } from './changes';

const reading = (path: string): Generation => ({ server: 'http://127.0.0.1:7878', name: 'coco', since: 0, ids: [], path, what: 'Reading coco' });
const listed = (path: string) => [{ id: 'ds_1', name: 'something else', path }];

describe('a folder being registered has landed', () => {
  it('when a dataset of its path is listed, however it was typed', () => {
    expect(landed(reading('D:\\data\\coco\\'), listed('D:/data/coco'))).toBe(true);
    expect(landed(reading('d:/Data/./sets/../coco'), listed('D:/data/coco'))).toBe(true);
    expect(landed(reading('\\\\nas\\share\\COCO'), listed('//nas/share/coco'))).toBe(true);
    expect(landed(reading('/data/./coco/'), listed('/data/coco'))).toBe(true);
  });

  it('not when only another folder is listed: on a Linux server case counts', () => {
    expect(landed(reading('/data/COCO'), listed('/data/coco'))).toBe(false);
    expect(landed(reading('D:/data/coco'), listed('D:/data/coco-2'))).toBe(false);
    expect(landed(reading('D:/data/coco'), listed('D:/coco'))).toBe(false);
  });
});
