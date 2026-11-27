import { describe, expect, it } from 'vitest';
import { parseMp4FrameStep, parseWebmFrameStep, probeVideoFrameStep, summarizeFrameDeltas } from './videoMeta';

function concat(...parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(new ArrayBuffer(total));
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

function ebmlVintSize(n: number | null): Uint8Array {
  if (n === null) return new Uint8Array([0xff]); // 1-byte unknown size
  if (n < 0x7f) return new Uint8Array([0x80 | n]);
  if (n < 0x3fff) return new Uint8Array([0x40 | (n >> 8), n & 0xff]);
  return new Uint8Array([0x20 | (n >> 16), (n >> 8) & 0xff, n & 0xff]);
}

function ebmlIdBytes(id: number): Uint8Array {
  const out: number[] = [];
  let v = id;
  while (v > 0) {
    out.unshift(v & 0xff);
    v = Math.floor(v / 256);
  }
  return new Uint8Array(out);
}

function ebmlUint(value: number, byteLen = 4): Uint8Array {
  const out = new Uint8Array(byteLen);
  let v = value;
  for (let i = byteLen - 1; i >= 0; i--) {
    out[i] = v & 0xff;
    v = Math.floor(v / 256);
  }
  return out;
}

function ebmlElem(id: number, body: Uint8Array, unknownSize = false): Uint8Array {
  return concat(ebmlIdBytes(id), ebmlVintSize(unknownSize ? null : body.length), body);
}

function simpleBlock(track: number, relTs: number): Uint8Array {
  const hi = (relTs >> 8) & 0xff;
  const lo = relTs & 0xff;
  return ebmlElem(0xa3, new Uint8Array([0x80 | track, hi, lo, 0x80, 0x00]));
}

function makeWebm(opts: {
  timestampScale?: number;
  defaultDurationNs?: number;
  blocks?: { track: number; relTs: number }[];
  unknownSegmentSize?: boolean;
}): Uint8Array {
  const ebml = ebmlElem(0x1a45dfa3, ebmlElem(0x4286, new Uint8Array([1])));
  const info = ebmlElem(0x1549a966, ebmlElem(0x2ad7b1, ebmlUint(opts.timestampScale ?? 1_000_000, 4)));
  const videoParts = [ebmlElem(0xd7, new Uint8Array([1])), ebmlElem(0x83, new Uint8Array([1]))];
  if (opts.defaultDurationNs !== undefined) {
    videoParts.push(ebmlElem(0x23e383, ebmlUint(opts.defaultDurationNs, 4)));
  }
  const audioEntry = ebmlElem(0xae, concat(ebmlElem(0xd7, new Uint8Array([2])), ebmlElem(0x83, new Uint8Array([2]))));
  const videoEntry = ebmlElem(0xae, concat(...videoParts));
  const tracks = ebmlElem(0x1654ae6b, concat(audioEntry, videoEntry));
  const clusterParts = [ebmlElem(0xe7, ebmlUint(0, 2))];
  for (const b of opts.blocks ?? []) clusterParts.push(simpleBlock(b.track, b.relTs));
  const cluster = ebmlElem(0x1f43b675, concat(...clusterParts));
  const segment = ebmlElem(0x18538067, concat(info, tracks, cluster), opts.unknownSegmentSize);
  return concat(ebml, segment);
}

function u32(n: number): Uint8Array {
  return new Uint8Array([(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff]);
}

function fourcc(s: string): Uint8Array {
  return new Uint8Array([...s].map((c) => c.charCodeAt(0)));
}

function mp4Box(type: string, body: Uint8Array): Uint8Array {
  return concat(u32(body.length + 8), fourcc(type), body);
}

function makeMp4(opts: { timescale: number; stts: [number, number][]; trailingMoov?: boolean; mdhdV1?: boolean }): Uint8Array<ArrayBuffer> {
  const ftyp = mp4Box('ftyp', fourcc('isom0000'));
  const hdlrAudio = mp4Box('hdlr', concat(new Uint8Array(8), fourcc('soun'), new Uint8Array(12)));
  const hdlrVideo = mp4Box('hdlr', concat(new Uint8Array(8), fourcc('vide'), new Uint8Array(12)));
  const mdhdBody = opts.mdhdV1
    ? concat(new Uint8Array([1, 0, 0, 0]), new Uint8Array(16), u32(opts.timescale), new Uint8Array(8))
    : concat(new Uint8Array([0, 0, 0, 0]), new Uint8Array(8), u32(opts.timescale), new Uint8Array(4));
  const mdhd = mp4Box('mdhd', mdhdBody);
  const sttsBody = concat(
    new Uint8Array(4),
    u32(opts.stts.length),
    ...opts.stts.map(([count, delta]) => concat(u32(count), u32(delta))),
  );
  const stts = mp4Box('stts', sttsBody);
  const stbl = mp4Box('stbl', stts);
  const minf = mp4Box('minf', stbl);
  const audioTrak = mp4Box('trak', mp4Box('mdia', concat(hdlrAudio, mdhd, minf)));
  const videoTrak = mp4Box('trak', mp4Box('mdia', concat(hdlrVideo, mdhd, minf)));
  const moov = mp4Box('moov', concat(audioTrak, videoTrak));
  if (opts.trailingMoov) {
    const mdat = mp4Box('mdat', new Uint8Array(600 * 1024));
    return concat(ftyp, mdat, moov);
  }
  return concat(ftyp, moov);
}

describe('Playground video container frame-step probing', () => {
  it('summarizes inter-frame deltas accurately under 1 ms rounding jitter and dropped frames', () => {
    // 30 fps WebM with 1 ms timecode scale produces alternating 33/34 ms deltas.
    const avg30 = summarizeFrameDeltas([33, 34, 33, 33, 34, 33]);
    expect(avg30).toBeCloseTo(33.333, 2);
    // Single dropped frame (200 ms) on a 10 fps clip is excluded as an outlier.
    const avg10 = summarizeFrameDeltas([100, 100, 200, 100, 100]);
    expect(avg10).toBeCloseTo(100, 5);
  });

  it('extracts frame duration from WebM DefaultDuration', () => {
    const buf = makeWebm({ defaultDurationNs: 100_000_000 }); // 10 fps
    expect(parseWebmFrameStep(buf)).toBeCloseTo(0.1, 6);
  });

  it('extracts frame duration from WebM SimpleBlock timestamps when DefaultDuration is omitted and ignores audio blocks', () => {
    const buf = makeWebm({
      unknownSegmentSize: true,
      blocks: [
        { track: 2, relTs: 0 },
        { track: 1, relTs: 0 },
        { track: 2, relTs: 20 },
        { track: 2, relTs: 40 },
        { track: 1, relTs: 100 },
        { track: 2, relTs: 60 },
        { track: 1, relTs: 200 },
        { track: 1, relTs: 300 },
      ],
    });
    expect(parseWebmFrameStep(buf)).toBeCloseTo(0.1, 6);
  });

  it('handles 30 fps WebM SimpleBlock rounding without drifting away from 1/30 s', () => {
    const buf = makeWebm({
      blocks: [0, 33, 67, 100, 133, 167, 200].map((relTs) => ({ track: 1, relTs })),
    });
    expect(parseWebmFrameStep(buf)).toBeCloseTo(1 / 30, 4);
  });

  it('extracts frame duration from MP4 mdhd + stts (front moov, v1 mdhd, dominant delta, and trailing moov)', async () => {
    // 10 fps MP4 with an odd final frame delta.
    const mp4Front = makeMp4({ timescale: 10_000, stts: [[10, 1000], [1, 1500]] });
    expect(parseMp4FrameStep(mp4Front)).toBeCloseTo(0.1, 6);

    // 29.97 fps MP4 with version-1 mdhd box.
    const mp4V1 = makeMp4({ timescale: 30_000, stts: [[120, 1001]], mdhdV1: true });
    expect(parseMp4FrameStep(mp4V1)).toBeCloseTo(1001 / 30_000, 6);

    // Trailing moov placed after >512 KB mdat is located via Blob.slice walking.
    const mp4Trailing = makeMp4({ timescale: 24_000, stts: [[48, 1000]], trailingMoov: true });
    const step = await probeVideoFrameStep(new Blob([mp4Trailing], { type: 'video/mp4' }));
    expect(step).toBeCloseTo(1 / 24, 6);
  });

  it('returns null safely on corrupt, truncated, or non-video files', async () => {
    expect(parseWebmFrameStep(new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 0x00, 0x00]))).toBeNull();
    expect(parseMp4FrameStep(new Uint8Array([0, 0, 0, 0, 0x66, 0x74, 0x79, 0x70]))).toBeNull();
    expect(await probeVideoFrameStep(new Blob(['not a video'], { type: 'text/plain' }))).toBeNull();
    expect(await probeVideoFrameStep(new Blob([]))).toBeNull();
  });
});
