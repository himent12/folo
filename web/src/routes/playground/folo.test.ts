import { describe, expect, it } from 'vitest';
import { buildWasmDetectOptions, hasFoloQuantMetadata, looksLikeFolo } from './folo';

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

describe('hasFoloQuantMetadata and buildWasmDetectOptions', () => {
  it('detects folo.quant calibration metadata in safetensors headers', async () => {
    expect(await hasFoloQuantMetadata(safetensors('{"__metadata__":{"folo.arch":"folo2-n","folo.quant":"{\\"version\\":1}"}}'))).toBe(true);
    expect(await hasFoloQuantMetadata(safetensors('{"__metadata__":{"folo.arch":"folo2-n"}}'))).toBe(false);
    expect(await hasFoloQuantMetadata(new Blob(['tiny']))).toBe(false);
  });

  it('builds strict folo-wasm detect options with perClassConf and confInclusive', () => {
    expect(buildWasmDetectOptions({ conf: 0.01, iou: 1, maxDet: 1000 })).toEqual({
      conf: 0.01,
      iou: 1,
      maxDet: 1000,
    });
    expect(
      buildWasmDetectOptions({
        conf: 0.01,
        iou: 1,
        maxDet: 1000,
        perClassConf: { 0: 0.4, 2: 0.75 },
      }),
    ).toEqual({
      conf: 0.01,
      iou: 1,
      maxDet: 1000,
      perClassConf: { 0: 0.4, 2: 0.75 },
      confInclusive: true,
    });
  });
});
