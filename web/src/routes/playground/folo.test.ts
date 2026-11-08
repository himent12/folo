import { describe, expect, it } from 'vitest';
import { looksLikeFolo } from './folo';

/** A safetensors file: an 8-byte little-endian header length, the JSON header, then `data` bytes. */
function safetensors(header: string, data = 8, claim?: number): Blob {
  const json = new TextEncoder().encode(header);
  const len = new DataView(new ArrayBuffer(8));
  len.setBigUint64(0, BigInt(claim ?? json.length), true);
  return new Blob([len.buffer, json, new Uint8Array(data)]);
}

describe('looksLikeFolo', () => {
  it('accepts a safetensors header', async () => {
    expect(await looksLikeFolo(safetensors('{"__metadata__":{"folo.arch":"folo-n"}}'))).toBe(true);
  });
  it('refuses an image picked by mistake', async () => {
    // A JPEG's first bytes read as an enormous header length (the "exceeds the 100000000-byte limit" failure).
    const jpeg = new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01])]);
    expect(await looksLikeFolo(jpeg)).toBe(false);
  });
  it('refuses a header longer than the file, or one that is not a JSON object', async () => {
    expect(await looksLikeFolo(safetensors('{"a":1}', 0, 4096))).toBe(false);
    expect(await looksLikeFolo(safetensors('["a"]'))).toBe(false);
    expect(await looksLikeFolo(new Blob(['tiny']))).toBe(false);
  });
});
