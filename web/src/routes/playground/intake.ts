// An image file as the Playground takes it (picked, dropped or pasted). The server reads JPEG, PNG, BMP and WebP
// (docs/API.md, POST /api/predict); any other image this browser can open (GIF, AVIF, ...) is drawn once and sent as a
// PNG, so a file that previews is a file that runs. The format is told by the file's first bytes, not its name.

/** The formats the server reads, as every message here names them. */
export const FORMATS = 'JPEG, PNG, BMP or WebP';

/** The picker's filter: every image (the rest are converted). */
export const ACCEPT = 'image/*';

export type Taken = { file: Blob; name: string; width: number; height: number; converted: boolean } | { error: string };

/** True when `head` (a file's first bytes) starts a JPEG, PNG, BMP or WebP file. Exported for tests. */
export function serverReads(head: Uint8Array): boolean {
  const at = (i: number, ...b: number[]) => b.every((x, k) => head[i + k] === x);
  return (
    at(0, 0xff, 0xd8, 0xff) || // JPEG
    at(0, 0x89, 0x50, 0x4e, 0x47) || // PNG
    at(0, 0x42, 0x4d) || // BMP
    (at(0, 0x52, 0x49, 0x46, 0x46) && at(8, 0x57, 0x45, 0x42, 0x50)) // RIFF....WEBP
  );
}

/** A decoded image drawn into a PNG. */
async function toPng(bmp: ImageBitmap): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  canvas.getContext('2d')!.drawImage(bmp, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png'));
}

/** Reads `file` for the Playground: its size, and the bytes to send (a PNG when the server cannot read the original). */
export async function takeImage(file: Blob, name: string): Promise<Taken> {
  if (file.type && !file.type.startsWith('image/')) return { error: `${name} is not an image. Use a ${FORMATS} file.` };
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file);
  } catch {
    return { error: `${name} could not be opened as an image. Use a ${FORMATS} file.` };
  }
  try {
    const head = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    const { width, height } = bmp;
    if (serverReads(head)) return { file, name, width, height, converted: false };
    return { file: await toPng(bmp), name, width, height, converted: true };
  } catch {
    return { error: `${name} could not be converted for the server. Use a ${FORMATS} file.` };
  } finally {
    bmp.close();
  }
}

/**
 * The image files of a drop, in the order given (anything that is not an image is left out; a drop of no image at all
 * keeps its first file, so the page can say why it does not run).
 */
export function droppedImages(files: FileList | readonly File[] | undefined): File[] {
  const all = [...(files ?? [])];
  const images = all.filter((f) => !f.type || f.type.startsWith('image/'));
  return images.length ? images : all.slice(0, 1);
}
