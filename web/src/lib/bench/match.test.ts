import { describe, expect, it } from 'vitest';
import { MockApi } from '../api/mock/index';
import { iou, matchImage } from './match';

const box = (c: number, x1: number, y1: number, x2: number, y2: number, score?: number) => ({ class: c, x1, y1, x2, y2, ...(score === undefined ? {} : { score }) });

describe('iou', () => {
  it('is the overlap over the union, 0 apart', () => {
    expect(iou(box(0, 0, 0, 10, 10), box(0, 0, 0, 10, 10))).toBe(1);
    expect(iou(box(0, 0, 0, 10, 10), box(0, 5, 0, 15, 10))).toBeCloseTo(50 / 150);
    expect(iou(box(0, 0, 0, 10, 10), box(0, 20, 20, 30, 30))).toBe(0);
  });
});

describe('matchImage', () => {
  it('matches in descending confidence, one true box per detection, same class, IoU from 0.5', () => {
    const truth = [box(0, 0, 0, 10, 10), box(1, 50, 50, 60, 60), box(0, 100, 100, 110, 110)];
    const dets = [
      box(0, 1, 0, 11, 10, 0.4), // a lower-scored duplicate of the first object: wrong once the other takes it
      box(0, 0, 0, 10, 10, 0.9),
      box(0, 50, 50, 60, 60, 0.8), // right place, wrong class
      box(0, 100, 100, 104, 104, 0.7), // too loose: IoU 0.16
    ];
    const m = matchImage(truth, dets);
    expect(m.detections.map((d) => d.truth)).toEqual([null, 0, null, null]);
    expect(m.found).toEqual([1, null, null]);
    expect(m.missed).toBe(2);
    expect(m.wrong).toBe(3);
  });

  it('gives a tie in IoU to the later true box, as the server does', () => {
    // The wide detection overlaps both true boxes by 2/3. The server gives it the second, which leaves the first
    // for the exact one: nothing missed, nothing wrong. Giving it the first would strand the other detection.
    const truth = [box(0, 0, 0, 10, 10), box(0, 5, 0, 15, 10)];
    const m = matchImage(truth, [box(0, 0, 0, 15, 10, 0.9), box(0, 0, 0, 10, 10, 0.8)]);
    expect(m.detections.map((d) => d.truth)).toEqual([1, 0]);
    expect([m.missed, m.wrong]).toEqual([0, 0]);
  });

  it('agrees with the counts the server sends for every image of an entry', async () => {
    const api = new MockApi('normal', 0);
    for (const [b, e] of [['bn_3c90', 'be_f101'], ['bn_3c90', 'be_f104'], ['bn_7a41', 'be_p201']] as const) {
      const page = await api.benchEntryImages(b, e, { limit: 1000 });
      for (const im of page.images) {
        const m = matchImage(im.image.boxes, im.detections);
        expect([m.missed, m.wrong]).toEqual([im.missed, im.wrong]);
      }
    }
  });
});
