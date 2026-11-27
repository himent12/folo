// `.folo` (safetensors) quick-inspection and `folo-wasm` option builders for Playground's browser inference path:
// validates picked weight files before wasm load, detects embedded `folo.quant` INT8 calibration metadata, and
// builds strict `FoloDetectOptions` (`perClassConf`, `confInclusive`) for `folo-wasm`.
import { sanitizeWasmPerClassConf, type WasmDetectOptions } from '$lib/inference/browser';

const MAX_HEADER = 100_000_000;
const MAX_META_READ = 2_000_000;

/** Whether `file` starts like a .folo (safetensors) file. Reads its first 9 bytes only. */
export async function looksLikeFolo(file: Blob): Promise<boolean> {
  if (file.size < 10) return false;
  const head = new DataView(await file.slice(0, 9).arrayBuffer());
  const n = head.getBigUint64(0, true);
  return n >= 2n && n <= BigInt(MAX_HEADER) && 8n + n <= BigInt(file.size) && head.getUint8(8) === 0x7b;
}

/**
 * Inspects a local `.folo` file's safetensors JSON header for embedded `folo.quant` INT8 calibration metadata.
 * Returns `false` for non-quantized, oversized-header, or unreadable files without throwing.
 */
export async function hasFoloQuantMetadata(file: Blob): Promise<boolean> {
  if (file.size < 10) return false;
  try {
    const head = new DataView(await file.slice(0, 8).arrayBuffer());
    const n = head.getBigUint64(0, true);
    if (n < 2n || n > BigInt(MAX_META_READ) || 8n + n > BigInt(file.size)) return false;
    const text = new TextDecoder().decode(await file.slice(8, 8 + Number(n)).arrayBuffer());
    const parsed = JSON.parse(text) as { __metadata__?: Record<string, unknown> };
    return typeof parsed.__metadata__?.['folo.quant'] === 'string' && parsed.__metadata__['folo.quant'].length > 0;
  } catch {
    return false;
  }
}

/**
 * Builds `folo-wasm` per-call detect options (`conf`, `iou`, `maxDet`, `perClassConf`, `confInclusive`)
 * for Playground's browser inference path.
 */
export function buildWasmDetectOptions(opts: {
  conf: number;
  iou: number;
  maxDet: number;
  perClassConf?: Readonly<Record<number, number>> | null;
  confInclusive?: boolean;
}): WasmDetectOptions {
  const perClassConf = sanitizeWasmPerClassConf(opts.perClassConf);
  return {
    conf: opts.conf,
    iou: opts.iou,
    maxDet: opts.maxDet,
    ...(perClassConf ? { perClassConf } : {}),
    ...(perClassConf || opts.confInclusive !== undefined ? { confInclusive: opts.confInclusive ?? true } : {}),
  };
}
