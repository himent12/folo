/**
 * Pure formatting, validation, and sanitization helpers for Settings › Diagnostics.
 * Produces a secret-free environment & connectivity summary safe to paste into bug reports or support threads.
 */
import { ApiError } from '$lib/api/client';
import type { Mode } from '$lib/api/index';
import type { CpuSimd, SystemInfo } from '$lib/api/types';
import type { ThemePref, Units } from '$lib/prefs';
import { deviceSummary, simdExtensions } from './capabilities';

export type GridDensity = 's' | 'm' | 'l';

export interface DiagnosticsSnapshot {
  studioVersion: string;
  serverVersion: string | null;
  mode: Mode;
  offline: boolean;
  serverMock: boolean | null;
  serverUrl: string;
  origin: string;
  deviceId: string;
  deviceSummary: string;
  /** Optional CPU int8 SIMD kernel tier & detected ISA flags from `/api/system`'s `cpu_simd`. */
  cpuSimd?: { kernel: string; extensions: string[] } | null;
  features: readonly string[];
  pingMs: number | null;
  /** Safe, validated failure classification (never raw server prose) for the copied report. */
  pingErrorSafe: string | null;
  themeMode: ThemePref;
  resolvedTheme: 'dark' | 'light';
  themeName: string;
  themeId: string;
  units: Units;
  gridDensity: GridDensity;
  gridBoxes: boolean;
  viewport: string | null;
}

const VALID_FEATURE = /^[a-z0-9][a-z0-9._-]{0,63}$/i;
const VALID_DEVICE_ID = /^(?:cpu|gpu:\d+|—)$/i;

function isRecord(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === 'object' && !Array.isArray(v);
}

/**
 * Runtime shape check for `/api/system` JSON so unexpected `200` payloads (for example `{ "status": "ok" }` or a
 * partial `capabilities` object missing `tensor_cores`/`amp`) are rejected cleanly instead of crashing rendering.
 */
export function isSystemInfo(raw: unknown): raw is SystemInfo {
  if (!isRecord(raw)) return false;
  if (!isRecord(raw.cpu) || typeof raw.cpu.name !== 'string') return false;
  if (!Array.isArray(raw.gpus)) return false;
  for (const g of raw.gpus) {
    if (!isRecord(g)) return false;
    if (typeof g.ordinal !== 'number' || typeof g.name !== 'string' || typeof g.backend !== 'string') return false;
  }
  if (raw.capabilities !== undefined) {
    if (!isRecord(raw.capabilities)) return false;
    const c = raw.capabilities;
    if (typeof c.device !== 'string' || typeof c.shader_f16 !== 'boolean') return false;
    if (!isRecord(c.tensor_cores)) return false;
    if (
      typeof c.tensor_cores.available !== 'boolean' ||
      typeof c.tensor_cores.enabled !== 'boolean' ||
      (c.tensor_cores.reason !== null && typeof c.tensor_cores.reason !== 'string')
    ) {
      return false;
    }
    if (!isRecord(c.amp)) return false;
    if (typeof c.amp.available !== 'boolean' || (c.amp.reason !== null && typeof c.amp.reason !== 'string')) return false;
  }
  if (raw.features !== undefined && (!Array.isArray(raw.features) || !raw.features.every((f) => typeof f === 'string'))) return false;
  if (raw.cpu_simd !== undefined) {
    if (!isRecord(raw.cpu_simd)) return false;
    const s = raw.cpu_simd;
    if (
      typeof s.avx2 !== 'boolean' ||
      typeof s.avx_vnni !== 'boolean' ||
      typeof s.avx512_vnni !== 'boolean' ||
      typeof s.neon !== 'boolean' ||
      typeof s.int8_fast !== 'boolean' ||
      typeof s.int8_kernel !== 'string'
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Strips credentials, path segments, query strings, and hash fragments — returning ONLY a validated `http:`/`https:`
 * origin (or `'same-origin'` when unset / opaque `'null'`, or `'(invalid URL)'` for malformed/relative/local-path input).
 */
export function sanitizeServerUrl(raw: string, fallbackOrigin = ''): string {
  // Reject non-empty inputs with leading/trailing or internal whitespace (never silently trimming malformed raw input).
  if (raw.length > 0 && raw.trim().length > 0 && raw !== raw.trim()) return '(invalid URL)';
  const trimmed = raw.trim();
  const fallback = fallbackOrigin.trim();
  const input = trimmed || (fallback.toLowerCase() === 'null' ? '' : fallback);
  if (!input) return 'same-origin';
  // Reject backslashes, whitespace/control chars, leading slashes, relative paths (`./`, `../`), and opaque origins.
  if (/[\\\s\u0000-\u001f\u007f]/.test(input) || input.startsWith('/') || input.startsWith('.') || input.toLowerCase() === 'null') {
    return '(invalid URL)';
  }
  const explicitScheme = /^https?:\/\//i.test(input);
  let candidate: string;
  if (explicitScheme) {
    candidate = input;
  } else if (input.includes('://')) {
    return '(invalid URL)';
  } else if (/^[a-z][a-z0-9+.-]*:(?!\d+(?:[/?#]|$))/i.test(input)) {
    // Reject local drive paths (`C:/...`) and non-HTTP schemes without `://` (`file:/...`, `javascript:...`).
    return '(invalid URL)';
  } else {
    candidate = `http://${input}`;
  }
  try {
    const parsed = new URL(candidate);
    if ((parsed.protocol !== 'http:' && parsed.protocol !== 'https:') || !parsed.hostname || parsed.origin === 'null') {
      return '(invalid URL)';
    }
    if (!explicitScheme) {
      // Bare inputs without a scheme must look like a real host (localhost, IPv6, dotted host, or explicit port),
      // never a relative path such as `private-folo/data`.
      const h = parsed.hostname.toLowerCase();
      const validBareHost = h === 'localhost' || h.startsWith('[') || (h.includes('.') && !h.startsWith('.') && !h.endsWith('.')) || parsed.port !== '';
      if (!validBareHost) return '(invalid URL)';
    }
    return parsed.origin;
  } catch {
    return '(invalid URL)';
  }
}

/**
 * Safe, fixed-vocabulary failure classification for copied reports so raw server error prose (which may contain local
 * paths, URLs, or tokens) is never copied verbatim.
 */
export function safePingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 0) return 'Unreachable';
    return Number.isInteger(err.status) && err.status >= 100 && err.status <= 599 ? `HTTP ${err.status}` : 'Server error';
  }
  if (err instanceof DOMException && (err.name === 'TimeoutError' || err.name === 'AbortError')) return 'Timed out';
  if (err instanceof Error && err.message === 'Invalid /api/system response') return 'Invalid response';
  return 'Unreachable';
}

/** Strips URLs, local filesystem paths, inline credentials/tokens, and control characters from copied report lines. */
function scrubInline(text: string, maxLen = 96): string {
  const cleaned = text
    .replace(/https?:\/\/\S+/gi, '[redacted]')
    .replace(/\b(?:token|secret|password|passwd|pwd|api[_-]?key|auth|bearer)\s*[:=]\s*\S+/gi, '[redacted]')
    .replace(/(^|\s)(?:[a-zA-Z]:[\\/]|\/)[^\s,)]+/g, '$1[redacted]')
    .replace(/[\r\n\t\u0000-\u001f\u007f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return 'unknown';
  return cleaned.length > maxLen ? `${cleaned.slice(0, maxLen - 1)}…` : cleaned;
}

/** Deduplicated, sorted list of valid server feature identifiers from `/api/system` and `/api/health`. */
export function mergeFeatures(systemFeatures?: readonly string[], hubFeatures?: readonly string[] | null): string[] {
  const seen = new Set<string>();
  for (const list of [systemFeatures, hubFeatures]) {
    if (!Array.isArray(list)) continue;
    for (const item of list) {
      if (typeof item !== 'string') continue;
      const clean = item.trim();
      if (clean && VALID_FEATURE.test(clean)) seen.add(clean);
    }
  }
  return [...seen].sort((a, b) => a.localeCompare(b));
}

/** Defensive, human-readable compute device id + summary from `/api/system`. */
export function resolveDeviceLabel(system?: SystemInfo): { id: string; summary: string } {
  if (!system || !isSystemInfo(system)) return { id: '—', summary: 'Not loaded' };
  if (system.capabilities) {
    if (!VALID_DEVICE_ID.test(system.capabilities.device)) {
      return { id: '—', summary: 'Unavailable' };
    }
    return {
      id: system.capabilities.device,
      summary: deviceSummary(system.capabilities, system.gpus, system.cpu.name),
    };
  }
  const firstGpu = system.gpus[0];
  if (firstGpu) {
    return {
      id: `gpu:${firstGpu.ordinal}`,
      summary: deviceSummary({ device: `gpu:${firstGpu.ordinal}` }, system.gpus, system.cpu.name),
    };
  }
  return { id: 'cpu', summary: system.cpu.name || 'CPU' };
}

/** Formats `/api/system`'s optional `cpu_simd` into a clean display + report summary. */
export function resolveCpuSimd(simd?: CpuSimd): { kernel: string; extensions: string[]; detail: string } | null {
  if (!simd) return null;
  const kernel = simd.int8_kernel.trim() || 'portable';
  const isPortable = kernel.toLowerCase() === 'portable';
  const extensions = simdExtensions(simd);
  const detail =
    isPortable && extensions.length > 0
      ? `Portable scalar kernel (detected ISA: ${extensions.join(', ')})`
      : extensions.length > 0
        ? `ISA: ${extensions.join(', ')}`
        : 'Portable scalar fallback (no SIMD ISA detected)';
  return { kernel, extensions, detail };
}

/** Human-readable ping round-trip duration or failure note. */
export function formatPing(ms: number | null, error?: string | null): string {
  if (error) return `Failed (${error})`;
  if (ms === null || !Number.isFinite(ms) || ms < 0) return 'Not measured';
  if (ms < 1) return '<1 ms';
  return `${Math.round(ms)} ms`;
}

/** Human-readable thumbnail grid density label. */
export function densityLabel(density: GridDensity): string {
  if (density === 's') return 'Compact (S)';
  if (density === 'l') return 'Large (L)';
  return 'Medium (M)';
}

/** Concise one-line server version and connection mode label. */
export function serverSummary(mode: Mode, version: string | null, serverMock: boolean | null, offline: boolean): string {
  if (mode === 'connecting') return offline ? 'Connecting (offline)' : 'Connecting';
  const ver = version ? scrubInline(version, 40) : 'unknown';
  if (mode === 'mock') return `Built-in demo ${ver} (mock)`;
  const kind = serverMock ? 'mock server' : 'live';
  const flags = offline ? `${kind}, offline` : kind;
  return `folo serve ${ver} (${flags})`;
}

/** Formats a sanitized plain-text diagnostics block suitable for issue reports. */
export function formatDiagnosticsReport(snap: DiagnosticsSnapshot): string {
  const endpoint = sanitizeServerUrl(snap.serverUrl, snap.origin);
  const safeDeviceId = VALID_DEVICE_ID.test(snap.deviceId) ? snap.deviceId : '—';
  const safeDeviceSummary = scrubInline(snap.deviceSummary, 80);
  const device = safeDeviceId === '—' ? safeDeviceSummary : `${safeDeviceSummary} (${safeDeviceId})`;
  const safeFeatures = mergeFeatures(snap.features);
  const features = safeFeatures.length > 0 ? safeFeatures.join(', ') : 'none reported';
  const safeThemeName = scrubInline(snap.themeName, 64);
  const safeThemeId = scrubInline(snap.themeId, 64);
  const theme =
    snap.themeMode === 'system'
      ? `${safeThemeName} (${safeThemeId}, system → ${snap.resolvedTheme})`
      : `${safeThemeName} (${safeThemeId}, ${snap.themeMode})`;
  const lines = [
    'FOLO Studio Diagnostics',
    `- Studio: ${scrubInline(snap.studioVersion, 40)}`,
    `- Server: ${serverSummary(snap.mode, snap.serverVersion, snap.serverMock, snap.offline)}`,
    `- Endpoint: ${endpoint}`,
    `- Ping (/api/system): ${formatPing(snap.pingMs, snap.pingErrorSafe)}`,
    `- Device: ${device}`,
  ];
  if (snap.cpuSimd) {
    const kernel = scrubInline(snap.cpuSimd.kernel, 48);
    const exts = snap.cpuSimd.extensions.map((e) => scrubInline(e, 24)).filter(Boolean);
    lines.push(`- CPU int8 SIMD: ${exts.length ? `${kernel} (${exts.join(', ')})` : kernel}`);
  }
  lines.push(
    `- Features: ${features}`,
    `- Theme: ${theme}`,
    `- Units: ${snap.units}`,
    `- Grid: ${densityLabel(snap.gridDensity)}, boxes ${snap.gridBoxes ? 'on' : 'off'}`,
  );
  if (snap.viewport) lines.push(`- Viewport: ${scrubInline(snap.viewport, 32)}`);
  return lines.join('\n');
}
