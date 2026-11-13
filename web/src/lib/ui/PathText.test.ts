import { describe, expect, it } from 'vitest';
import { splitPath } from './PathText';

describe('splitPath', () => {
  it('keeps the folder and the one holding it whole, so folders of one name in two places read apart', () => {
    expect(splitPath('D:/folo-sweep3/studio-data/work/messy/place-a/cells')).toEqual({
      head: 'D:/folo-sweep3/studio-data/work/messy',
      tail: '/place-a/cells',
    });
    expect(splitPath('D:/folo-sweep3/studio-data/work/messy/place-b/cells').tail).toBe('/place-b/cells');
  });

  it('cuts nothing off a short path, and reads back slashes and a trailing separator', () => {
    expect(splitPath('D:/data/coco')).toEqual({ head: '', tail: 'D:/data/coco' });
    expect(splitPath('/data/coco/')).toEqual({ head: '', tail: '/data/coco' });
    expect(splitPath('C:\\Users\\me\\data\\bccd')).toEqual({ head: 'C:\\Users\\me', tail: '\\data\\bccd' });
    expect(splitPath('/')).toEqual({ head: '', tail: '/' });
  });
});
