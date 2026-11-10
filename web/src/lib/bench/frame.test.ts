import { describe, expect, it } from 'vitest';
import { dominantAspect } from '$lib/ui/TileGrid.svelte';
import { containFit, placeBox } from './frame';

// Real photos are not square (COCO: 640×427, 427×640, panoramas): a benchmark's gallery tile, its large view and the
// condition preview show each one whole in a frame of another shape, and every box has to land on its object.

/** The image's own rectangle in the frame, and a box at each of its corners and its centre. */
function check(cw: number, ch: number, w: number, h: number) {
  const f = containFit(cw, ch, w, h)!;
  const img = placeBox(f, { x1: 0, y1: 0, x2: w, y2: h });
  // The whole image, as large as fits, centred: one side fills the frame, the other is centred in it.
  expect(img.left).toBeGreaterThanOrEqual(-1e-9);
  expect(img.top).toBeGreaterThanOrEqual(-1e-9);
  expect(img.left + img.width).toBeLessThanOrEqual(cw + 1e-9);
  expect(img.top + img.height).toBeLessThanOrEqual(ch + 1e-9);
  expect(Math.max(img.width / cw, img.height / ch)).toBeCloseTo(1, 9);
  expect(img.left).toBeCloseTo(cw - img.left - img.width, 9);
  expect(img.top).toBeCloseTo(ch - img.top - img.height, 9);
  // Its aspect kept.
  expect(img.width / img.height).toBeCloseTo(w / h, 9);
  // A box anywhere maps by the same scale and offset: corners to corners, the centre to the centre.
  const b = placeBox(f, { x1: w * 0.25, y1: h * 0.5, x2: w * 0.75, y2: h });
  expect(b.left).toBeCloseTo(img.left + img.width * 0.25, 9);
  expect(b.top).toBeCloseTo(img.top + img.height * 0.5, 9);
  expect(b.left + b.width).toBeCloseTo(img.left + img.width * 0.75, 9);
  expect(b.top + b.height).toBeCloseTo(img.top + img.height, 9);
  return img;
}

describe('an image in a frame of another shape', () => {
  it('shows a landscape, a portrait and a panorama whole, centred, its boxes on it', () => {
    // A 4:3 gallery tile, 216 px wide.
    const land = check(216, 162, 640, 427);
    expect(land.width).toBeCloseTo(216, 9);
    expect(land.top).toBeGreaterThan(0);
    const port = check(216, 162, 427, 640);
    expect(port.height).toBeCloseTo(162, 9);
    expect(port.left).toBeGreaterThan(50);
    const pano = check(216, 162, 1600, 520);
    expect(pano.width).toBeCloseTo(216, 9);
    // The large view, sized to the image itself: no band at all.
    const own = check(640, 427, 640, 427);
    expect([own.left, own.top]).toEqual([0, 0]);
    // A square frame, a phone's preview.
    check(358, 358, 960, 640);
    check(358, 358, 720, 1280);
  });

  it('waits for both sizes', () => {
    expect(containFit(0, 100, 640, 480)).toBeNull();
    expect(containFit(100, 100, 0, 480)).toBeNull();
  });

  it("gives a gallery's tiles the commonest shape among its images, never the first one's", () => {
    // The real case: a portrait photo first made every tile of a row of landscapes a tall box.
    const coco = [
      { width: 427, height: 640 },
      { width: 640, height: 427 },
      { width: 640, height: 480 },
      { width: 640, height: 427 },
      { width: 640, height: 427 },
      { width: 480, height: 640 },
    ];
    expect(dominantAspect(coco)).toBeCloseTo(1.5, 2);
    // Within 3:4 to 16:9: a set of panoramas does not make slivers of tiles.
    expect(dominantAspect([{ width: 1600, height: 520 }])).toBeCloseTo(16 / 9, 9);
    expect(dominantAspect([])).toBeCloseTo(4 / 3, 9);
  });
});
