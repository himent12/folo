// A quick look at a picked file before it is handed to the in-browser runtime as model weights: a .folo file is a
// safetensors file (docs/FORMAT.md), so it opens with an 8-byte little-endian header length N (at most 100,000,000,
// and the header must fit in the file) followed by a JSON object starting with `{`. Anything else (a JPEG picked by
// mistake) is refused here, by name, instead of failing deep in the loader.

const MAX_HEADER = 100_000_000;

/** Whether `file` starts like a .folo (safetensors) file. Reads its first 9 bytes only. */
export async function looksLikeFolo(file: Blob): Promise<boolean> {
  if (file.size < 10) return false;
  const head = new DataView(await file.slice(0, 9).arrayBuffer());
  const n = head.getBigUint64(0, true);
  return n >= 2n && n <= BigInt(MAX_HEADER) && 8n + n <= BigInt(file.size) && head.getUint8(8) === 0x7b;
}
