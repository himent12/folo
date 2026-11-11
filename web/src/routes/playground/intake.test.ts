// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { droppedImages, serverReads, takeImage } from './intake';

const bytes = (...b: number[]) => new Uint8Array(b);
const GIF = bytes(0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0, 0, 0, 0, 0, 0);
const PNG = bytes(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0);

describe('serverReads', () => {
  it('knows the four formats the server reads by their first bytes, whatever the name says', () => {
    expect(serverReads(bytes(0xff, 0xd8, 0xff, 0xe0))).toBe(true);
    expect(serverReads(PNG)).toBe(true);
    expect(serverReads(bytes(0x42, 0x4d, 0, 0))).toBe(true);
    expect(serverReads(bytes(0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50))).toBe(true);
    expect(serverReads(GIF)).toBe(false);
    expect(serverReads(bytes(0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45))).toBe(false); // WAVE
  });
});

describe('takeImage', () => {
  afterEach(() => vi.unstubAllGlobals());
  const bitmap = { width: 4, height: 3, close: () => {} };

  it('sends a GIF as a PNG of what the browser decoded', async () => {
    vi.stubGlobal('createImageBitmap', async () => bitmap);
    const png = new Blob([PNG], { type: 'image/png' });
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage: () => {} } as never);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => cb(png));
    const t = await takeImage(new Blob([GIF], { type: 'image/gif' }), 'anim.gif');
    expect(t).toMatchObject({ file: png, name: 'anim.gif', width: 4, height: 3, converted: true });
  });

  it('keeps a file the server reads as it is', async () => {
    vi.stubGlobal('createImageBitmap', async () => bitmap);
    const file = new Blob([PNG], { type: 'image/png' });
    expect(await takeImage(file, 'a.png')).toMatchObject({ file, converted: false });
  });

  it('names the formats when a file cannot be opened', async () => {
    vi.stubGlobal('createImageBitmap', async () => {
      throw new Error('nope');
    });
    expect(await takeImage(new Blob(['text'], { type: 'image/jpeg' }), 'fake.jpg')).toEqual({
      error: 'fake.jpg could not be opened as an image. Use a JPEG, PNG, BMP or WebP file.',
    });
    expect(await takeImage(new Blob(['text'], { type: 'text/plain' }), 'notes.txt')).toEqual({ error: 'notes.txt is not an image. Use a JPEG, PNG, BMP or WebP file.' });
  });
});

describe('droppedImages', () => {
  it('keeps the images in order, or the first file when there is none', () => {
    const f = (name: string, type: string) => new File(['x'], name, { type });
    expect(droppedImages([f('a.txt', 'text/plain'), f('b.jpg', 'image/jpeg'), f('c.png', 'image/png')]).map((x) => x.name)).toEqual(['b.jpg', 'c.png']);
    expect(droppedImages([f('a.txt', 'text/plain'), f('b.txt', 'text/plain')]).map((x) => x.name)).toEqual(['a.txt']);
    expect(droppedImages(undefined)).toEqual([]);
  });
});
