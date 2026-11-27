// Zero-dependency container header probe for local video frame duration (WebM/Matroska and MP4/MOV).
// Lets Playground step cleanly across adjacent decoded frames and display accurate frame counts on non-30fps clips
// (e.g. 10 fps WebM, 24/25/60 fps MP4) before playback starts.

/** Bounds accepted for detected frame durations (240 fps down to 0.5 fps). */
export const MIN_FRAME_STEP_SEC = 1 / 240;
export const MAX_FRAME_STEP_SEC = 2.0;

const HEAD_PROBE_BYTES = 512 * 1024;
const MAX_MOOV_BYTES = 2 * 1024 * 1024;
const MAX_TOP_BOXES = 64;

export function isValidFrameStep(sec: number): boolean {
  return Number.isFinite(sec) && sec >= MIN_FRAME_STEP_SEC && sec <= MAX_FRAME_STEP_SEC;
}

/** Robustly summarizes positive inter-frame deltas (inlier mean around the median to handle 1 ms WebM jitter and dropped frames). */
export function summarizeFrameDeltas(deltas: number[]): number | null {
  const valid = deltas.filter((d) => Number.isFinite(d) && d > 0).sort((a, b) => a - b);
  if (valid.length === 0) return null;
  const median = valid[Math.floor(valid.length / 2)]!;
  const lo = median * 0.5;
  const hi = median * 1.5;
  let sum = 0;
  let count = 0;
  for (const d of valid) {
    if (d >= lo && d <= hi) {
      sum += d;
      count++;
    }
  }
  return count > 0 ? sum / count : median;
}

// ---------------------------------------------------------------------------
// WebM / Matroska (EBML) parser
// ---------------------------------------------------------------------------

const ID_EBML = 0x1a45dfa3;
const ID_SEGMENT = 0x18538067;
const ID_INFO = 0x1549a966;
const ID_TIMESTAMP_SCALE = 0x2ad7b1;
const ID_TRACKS = 0x1654ae6b;
const ID_TRACK_ENTRY = 0xae;
const ID_TRACK_NUMBER = 0xd7;
const ID_TRACK_TYPE = 0x83;
const ID_DEFAULT_DURATION = 0x23e383;
const ID_CLUSTER = 0x1f43b675;
const ID_CLUSTER_TIMESTAMP = 0xe7;
const ID_SIMPLE_BLOCK = 0xa3;
const ID_BLOCK_GROUP = 0xa0;
const ID_BLOCK = 0xa1;

/** Level-1 Segment children and top-level IDs that terminate an unknown-size Cluster per RFC 8794 §6.2. */
const EBML_CLUSTER_TERMINATORS = new Set([
  ID_EBML,
  ID_SEGMENT,
  ID_INFO,
  ID_TRACKS,
  ID_CLUSTER,
  0x114d9b74, // SeekHead
  0x1c53bb6b, // Cues
  0x1043a770, // Chapters
  0x1254c367, // Tags
  0x1941a469, // Attachments
]);

function readEbmlId(buf: Uint8Array, pos: number, end: number): { id: number; next: number } | null {
  if (pos >= end) return null;
  const b0 = buf[pos]!;
  let len = 0;
  for (let mask = 0x80; mask >= 0x10; mask >>= 1) {
    len++;
    if (b0 & mask) break;
    if (mask === 0x10) return null;
  }
  if (pos + len > end) return null;
  let id = 0;
  for (let i = 0; i < len; i++) id = id * 256 + buf[pos + i]!;
  return { id, next: pos + len };
}

function readEbmlSize(buf: Uint8Array, pos: number, end: number): { size: number | null; next: number } | null {
  if (pos >= end) return null;
  const b0 = buf[pos]!;
  if (b0 === 0) return null;
  let len = 1;
  let mask = 0x80;
  while (len <= 8 && (b0 & mask) === 0) {
    len++;
    mask >>= 1;
  }
  if (len > 8 || pos + len > end) return null;
  const firstMask = mask - 1;
  let val = b0 & firstMask;
  let allOnes = val === firstMask;
  for (let i = 1; i < len; i++) {
    const b = buf[pos + i]!;
    if (b !== 0xff) allOnes = false;
    val = val * 256 + b;
  }
  return { size: allOnes ? null : val, next: pos + len };
}

function readUintBe(buf: Uint8Array, pos: number, len: number): number {
  if (len <= 0 || len > 8 || pos + len > buf.length) return 0;
  let val = 0;
  for (let i = 0; i < len; i++) val = val * 256 + buf[pos + i]!;
  return val;
}

function readBlockHeader(buf: Uint8Array, pos: number, end: number): { track: number; relTs: number } | null {
  const trk = readEbmlSize(buf, pos, end);
  if (!trk || trk.size === null || trk.next + 2 > end) return null;
  const raw = (buf[trk.next]! << 8) | buf[trk.next + 1]!;
  const relTs = raw >= 0x8000 ? raw - 0x10000 : raw;
  return { track: trk.size, relTs };
}

/** Extracts the frame duration (in seconds) of a WebM / Matroska buffer from `DefaultDuration` or video block timestamps. */
export function parseWebmFrameStep(buf: Uint8Array): number | null {
  if (buf.length < 4 || buf[0] !== 0x1a || buf[1] !== 0x45 || buf[2] !== 0xdf || buf[3] !== 0xa3) return null;

  let timestampScale = 1_000_000;
  let videoTrack: number | null = null;
  let defaultDurationNs: number | null = null;
  const timestamps: number[] = [];

  let pos = 0;
  while (pos < buf.length) {
    const idR = readEbmlId(buf, pos, buf.length);
    if (!idR) break;
    const szR = readEbmlSize(buf, idR.next, buf.length);
    if (!szR) break;
    const dataStart = szR.next;
    const dataEnd = szR.size === null ? buf.length : Math.min(buf.length, dataStart + szR.size);
    if (dataEnd < dataStart) break;

    if (idR.id === ID_SEGMENT) {
      let sp = dataStart;
      while (sp < dataEnd) {
        const sId = readEbmlId(buf, sp, dataEnd);
        if (!sId) break;
        const sSz = readEbmlSize(buf, sId.next, dataEnd);
        if (!sSz) break;
        const cStart = sSz.next;
        const cEnd = sSz.size === null ? dataEnd : Math.min(dataEnd, cStart + sSz.size);

        let nextSp = cEnd;
        if (sId.id === ID_INFO) {
          let ip = cStart;
          while (ip < cEnd) {
            const iId = readEbmlId(buf, ip, cEnd);
            if (!iId) break;
            if (sSz.size === null && EBML_CLUSTER_TERMINATORS.has(iId.id)) {
              nextSp = ip;
              break;
            }
            const iSz = readEbmlSize(buf, iId.next, cEnd);
            if (!iSz || iSz.size === null) break;
            const iEnd = Math.min(cEnd, iSz.next + iSz.size);
            if (iId.id === ID_TIMESTAMP_SCALE && iSz.size > 0 && iSz.size <= 8) {
              const sc = readUintBe(buf, iSz.next, iSz.size);
              if (sc > 0) timestampScale = sc;
            }
            if (iEnd <= ip) break;
            ip = iEnd;
          }
        } else if (sId.id === ID_TRACKS) {
          let tp = cStart;
          while (tp < cEnd) {
            const tId = readEbmlId(buf, tp, cEnd);
            if (!tId) break;
            if (sSz.size === null && EBML_CLUSTER_TERMINATORS.has(tId.id)) {
              nextSp = tp;
              break;
            }
            const tSz = readEbmlSize(buf, tId.next, cEnd);
            if (!tSz) break;
            const tEnd = tSz.size === null ? cEnd : Math.min(cEnd, tSz.next + tSz.size);
            if (tId.id === ID_TRACK_ENTRY) {
              let num: number | null = null;
              let type: number | null = null;
              let defDur: number | null = null;
              let ep = tSz.next;
              while (ep < tEnd) {
                const eId = readEbmlId(buf, ep, tEnd);
                if (!eId) break;
                const eSz = readEbmlSize(buf, eId.next, tEnd);
                if (!eSz || eSz.size === null) break;
                const eEnd = Math.min(tEnd, eSz.next + eSz.size);
                if (eId.id === ID_TRACK_NUMBER) num = readUintBe(buf, eSz.next, eSz.size);
                else if (eId.id === ID_TRACK_TYPE) type = readUintBe(buf, eSz.next, eSz.size);
                else if (eId.id === ID_DEFAULT_DURATION) defDur = readUintBe(buf, eSz.next, eSz.size);
                if (eEnd <= ep) break;
                ep = eEnd;
              }
              if (type === 1 && videoTrack === null) {
                if (num !== null) videoTrack = num;
                if (defDur !== null && defDur > 0) defaultDurationNs = defDur;
              }
            }
            if (tEnd <= tp) break;
            tp = tEnd;
          }
          if (defaultDurationNs !== null && isValidFrameStep(defaultDurationNs / 1e9)) {
            return defaultDurationNs / 1e9;
          }
        } else if (sId.id === ID_CLUSTER && timestamps.length < 48) {
          let clusterTs = 0;
          let cp = cStart;
          while (cp < cEnd && timestamps.length < 48) {
            const clId = readEbmlId(buf, cp, cEnd);
            if (!clId) break;
            if (sSz.size === null && EBML_CLUSTER_TERMINATORS.has(clId.id)) {
              nextSp = cp;
              break;
            }
            const clSz = readEbmlSize(buf, clId.next, cEnd);
            if (!clSz) break;
            const clStart = clSz.next;
            const clEnd = clSz.size === null ? cEnd : Math.min(cEnd, clStart + clSz.size);
            if (clId.id === ID_CLUSTER_TIMESTAMP && clSz.size !== null) {
              clusterTs = readUintBe(buf, clStart, clSz.size);
            } else if (clId.id === ID_SIMPLE_BLOCK) {
              const blk = readBlockHeader(buf, clStart, clEnd);
              if (blk && (videoTrack === null || blk.track === videoTrack)) {
                timestamps.push(clusterTs + blk.relTs);
              }
            } else if (clId.id === ID_BLOCK_GROUP) {
              let bp = clStart;
              while (bp < clEnd) {
                const bId = readEbmlId(buf, bp, clEnd);
                if (!bId) break;
                const bSz = readEbmlSize(buf, bId.next, clEnd);
                if (!bSz || bSz.size === null) break;
                const bEnd = Math.min(clEnd, bSz.next + bSz.size);
                if (bId.id === ID_BLOCK) {
                  const blk = readBlockHeader(buf, bSz.next, bEnd);
                  if (blk && (videoTrack === null || blk.track === videoTrack)) {
                    timestamps.push(clusterTs + blk.relTs);
                  }
                }
                if (bEnd <= bp) break;
                bp = bEnd;
              }
            }
            if (clEnd <= cp) break;
            cp = clEnd;
          }
        }

        if (nextSp <= sp) break;
        sp = nextSp;
      }
      break;
    }

    if (dataEnd <= pos) break;
    pos = dataEnd;
  }

  if (defaultDurationNs !== null && isValidFrameStep(defaultDurationNs / 1e9)) {
    return defaultDurationNs / 1e9;
  }
  if (timestamps.length >= 2) {
    timestamps.sort((a, b) => a - b);
    const diffs: number[] = [];
    for (let i = 1; i < timestamps.length; i++) {
      const d = timestamps[i]! - timestamps[i - 1]!;
      if (d > 0) diffs.push(d);
    }
    const avgTicks = summarizeFrameDeltas(diffs);
    if (avgTicks !== null) {
      const sec = (avgTicks * timestampScale) / 1e9;
      if (isValidFrameStep(sec)) return sec;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// MP4 / MOV (ISO BMFF) parser
// ---------------------------------------------------------------------------

function readU32(buf: Uint8Array, pos: number): number {
  return ((buf[pos]! << 24) | (buf[pos + 1]! << 16) | (buf[pos + 2]! << 8) | buf[pos + 3]!) >>> 0;
}

function readFourCC(buf: Uint8Array, pos: number): string {
  return String.fromCharCode(buf[pos]!, buf[pos + 1]!, buf[pos + 2]!, buf[pos + 3]!);
}

interface BoxHeader {
  type: string;
  bodyStart: number;
  end: number;
}

function readBox(buf: Uint8Array, pos: number, limit: number): BoxHeader | null {
  if (pos + 8 > limit) return null;
  let size = readU32(buf, pos);
  const type = readFourCC(buf, pos + 4);
  let headerLen = 8;
  if (size === 1) {
    if (pos + 16 > limit) return null;
    const hi = readU32(buf, pos + 8);
    const lo = readU32(buf, pos + 12);
    size = hi * 4294967296 + lo;
    headerLen = 16;
  } else if (size === 0) {
    size = limit - pos;
  }
  if (size < headerLen) return null;
  const end = Math.min(limit, pos + size);
  return { type, bodyStart: pos + headerLen, end };
}

function parseSttsDelta(buf: Uint8Array, start: number, end: number): number | null {
  if (start + 8 > end) return null;
  const entryCount = readU32(buf, start + 4);
  const maxEntries = Math.min(entryCount, 512);
  const counts = new Map<number, number>();
  let totalSamples = 0;
  let p = start + 8;
  for (let i = 0; i < maxEntries && p + 8 <= end; i++, p += 8) {
    const sampleCount = readU32(buf, p);
    const sampleDelta = readU32(buf, p + 4);
    if (sampleCount > 0 && sampleDelta > 0) {
      counts.set(sampleDelta, (counts.get(sampleDelta) ?? 0) + sampleCount);
      totalSamples += sampleCount;
    }
  }
  if (totalSamples === 0) return null;
  const sorted = [...counts.entries()].sort((a, b) => a[0] - b[0]);
  // Find weighted median sample_delta, then average inliers within ±max(1 tick, 10%) so repeating rounded
  // durations (e.g. 33/34/33 ms at 1 kHz for 30 fps) resolve to 33.333 ms without drifting, while odd final
  // or dropped-frame entries are excluded.
  const half = totalSamples / 2;
  let seen = 0;
  let median = sorted[0]![0];
  for (const [delta, count] of sorted) {
    seen += count;
    if (seen >= half) {
      median = delta;
      break;
    }
  }
  const tol = Math.max(1, median * 0.1);
  let weightedSum = 0;
  let weightedCount = 0;
  for (const [delta, count] of sorted) {
    if (Math.abs(delta - median) <= tol) {
      weightedSum += delta * count;
      weightedCount += count;
    }
  }
  return weightedCount > 0 ? weightedSum / weightedCount : median;
}

function parseMoovFrameStep(buf: Uint8Array, start: number, end: number): number | null {
  let pos = start;
  while (pos < end) {
    const box = readBox(buf, pos, end);
    if (!box || box.end <= pos) break;
    if (box.type === 'trak') {
      let tp = box.bodyStart;
      while (tp < box.end) {
        const tb = readBox(buf, tp, box.end);
        if (!tb || tb.end <= tp) break;
        if (tb.type === 'mdia') {
          let isVideo = false;
          let timescale = 0;
          let sampleDelta: number | null = null;
          let mp = tb.bodyStart;
          while (mp < tb.end) {
            const mb = readBox(buf, mp, tb.end);
            if (!mb || mb.end <= mp) break;
            if (mb.type === 'hdlr' && mb.bodyStart + 12 <= mb.end) {
              isVideo = readFourCC(buf, mb.bodyStart + 8) === 'vide';
            } else if (mb.type === 'mdhd' && mb.bodyStart + 16 <= mb.end) {
              const ver = buf[mb.bodyStart]!;
              if (ver === 0 && mb.bodyStart + 16 <= mb.end) timescale = readU32(buf, mb.bodyStart + 12);
              else if (ver === 1 && mb.bodyStart + 24 <= mb.end) timescale = readU32(buf, mb.bodyStart + 20);
            } else if (mb.type === 'minf') {
              let np = mb.bodyStart;
              while (np < mb.end) {
                const nb = readBox(buf, np, mb.end);
                if (!nb || nb.end <= np) break;
                if (nb.type === 'stbl') {
                  let sp = nb.bodyStart;
                  while (sp < nb.end) {
                    const sb = readBox(buf, sp, nb.end);
                    if (!sb || sb.end <= sp) break;
                    if (sb.type === 'stts') sampleDelta = parseSttsDelta(buf, sb.bodyStart, sb.end);
                    sp = sb.end;
                  }
                }
                np = nb.end;
              }
            }
            mp = mb.end;
          }
          if (isVideo && timescale > 0 && sampleDelta !== null) {
            const step = sampleDelta / timescale;
            if (isValidFrameStep(step)) return step;
          }
        }
        tp = tb.end;
      }
    }
    pos = box.end;
  }
  return null;
}

/** Extracts the video frame duration (in seconds) from an MP4 / MOV buffer containing a `moov` box. */
export function parseMp4FrameStep(buf: Uint8Array): number | null {
  let pos = 0;
  while (pos < buf.length) {
    const box = readBox(buf, pos, buf.length);
    if (!box || box.end <= pos) break;
    if (box.type === 'moov') {
      const step = parseMoovFrameStep(buf, box.bodyStart, box.end);
      if (step !== null) return step;
    }
    pos = box.end;
  }
  return null;
}

function looksLikeIsoBmff(buf: Uint8Array): boolean {
  if (buf.length < 8) return false;
  const t = readFourCC(buf, 4);
  return t === 'ftyp' || t === 'moov' || t === 'mdat' || t === 'free' || t === 'wide' || t === 'skip';
}

async function probeMp4Blob(blob: Blob, head: Uint8Array): Promise<number | null> {
  const fromHead = parseMp4FrameStep(head);
  if (fromHead !== null) return fromHead;
  if (blob.size <= head.length) return null;

  // Walk top-level ISO BMFF boxes using tiny 16-byte slices to find a trailing `moov` atom.
  let offset = 0;
  for (let i = 0; i < MAX_TOP_BOXES && offset + 8 <= blob.size; i++) {
    const hdr = new Uint8Array(await blob.slice(offset, Math.min(blob.size, offset + 16)).arrayBuffer());
    if (hdr.length < 8) break;
    let size = readU32(hdr, 0);
    const type = readFourCC(hdr, 4);
    let headerLen = 8;
    if (size === 1) {
      if (hdr.length < 16) break;
      size = readU32(hdr, 8) * 4294967296 + readU32(hdr, 12);
      headerLen = 16;
    } else if (size === 0) {
      size = blob.size - offset;
    }
    if (size < headerLen) break;
    if (type === 'moov') {
      const bodyLen = Math.min(size - headerLen, MAX_MOOV_BYTES);
      const moovBody = new Uint8Array(await blob.slice(offset + headerLen, offset + headerLen + bodyLen).arrayBuffer());
      return parseMoovFrameStep(moovBody, 0, moovBody.length);
    }
    offset += size;
  }
  return null;
}

/**
 * Probes a local video File/Blob to determine its frame duration in seconds (e.g. `0.1` for 10 fps).
 * Returns `null` when the container format is unsupported or exposes no frame timing metadata.
 */
export async function probeVideoFrameStep(file: Blob): Promise<number | null> {
  try {
    if (!file || file.size < 8) return null;
    const head = new Uint8Array(await file.slice(0, Math.min(file.size, HEAD_PROBE_BYTES)).arrayBuffer());
    if (head.length >= 4 && head[0] === 0x1a && head[1] === 0x45 && head[2] === 0xdf && head[3] === 0xa3) {
      return parseWebmFrameStep(head);
    }
    if (looksLikeIsoBmff(head)) {
      return await probeMp4Blob(file, head);
    }
    return null;
  } catch {
    return null;
  }
}
