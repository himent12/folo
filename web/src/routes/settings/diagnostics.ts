/**
 * Pure formatting, validation, and sanitization helpers for Settings › Diagnostics.
 * Produces a secret-free environment & connectivity summary safe to paste into bug reports or support threads.
 */
import { ApiError } from '$lib/api/client';
import type { Mode } from '$lib/api/index';
import type { SystemInfo } from '$lib/api/types';
import type { ThemePref, Units } from '$lib/prefs';
import { deviceSummary } from './capabilities';

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

/**
 * Runtime shape check for `/api/system` JSON so unexpected `200` payloads (for example `{ "status": "ok" }` from a
 * reverse proxy) are flagged as invalid responses instead of crashing rendering or counting as a healthy ping.
 */
export function isSystemInfo(raw: unknown): raw is SystemInfo {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return false;
  const o = raw as Record<string, unknown>;
  if (!o.cpu || typeof o.cpu !== 'object' || Array.isArray(o.cpu)) return false;
  if (typeof (o.cpu as Record<string, unknown>).name !== 'string') return false;
  if (!Array.isArray(o.gpus)) return false;
  for (const g of o.gpus) {
    if (!g || typeof g !== 'object' || Array.isArray(g)) return false;
    const gpu = g as Record<string, unknown>;
    if (typeof gpu.ordinal !== 'number' || typeof gpu.name !== 'string' || typeof gpu.backend !== 'string') return false;
  }
  if (o.capabilities !== undefined) {
    if (!o.capabilities || typeof o.capabilities !== 'object' || Array.isArray(o.capabilities)) return false;
    if (typeof (o.capabilities as Record<string, unknown>).device !== 'string') return false;
  }
  if (o.features !== undefined && (!Array.isArray(o.features) || !o.features.every((f) => typeof f === 'string'))) return false;
  return true;
}

/**
 * Strips credentials, path segments, query strings, and hash fragments — returning ONLY a validated `http:`/`https:`
 * origin (or `'same-origin'` when unset / opaque `'null'`, or `'(invalid URL)'` for malformed/local-path input).
 */
export function sanitizeServerUrl(raw: string, fallbackOrigin = ''): string {
  const trimmed = raw.trim();
  const fallback = fallbackOrigin.trim();
  const input = trimmed || (fallback.toLowerCase() === 'null' ? '' : fallback);
  if (!input) return 'same-origin';
  // Reject backslashes, internal whitespace/control characters, leading slashes, and opaque origins.
  if (/[\\\s\u0000-\u001f\u007f]/.test(input) || input.startsWith('/') || input.toLowerCase() === 'null') {
    return '(invalid URL)';
  }
  let candidate: string;
  if (/^https?:\/\//i.test(input)) {
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
    return Number.isInteger(err.status) && err.status >= 100 && err.status <= 599 ? `HTTP ${err.status}` : 'Server error';
  }
  if (err instanceof DOMException && (err.name === 'TimeoutError' || err.name === 'AbortError')) return 'Timed out';
  if (err instanceof Error && err.message === 'Invalid /api/system response') return 'Invalid response';
  return 'Unreachable';
}

/** Strips URLs, control characters, and excessive length from display strings placed into copied bug reports. */
function scrubInline(text: string, maxLen = 96): string {
  const cleaned = text
    .replace(/https?:\/\/\S+/gi, '[redacted-url]')
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
    `- Features: ${features}`,
    `- Theme: ${theme}`,
    `- Units: ${snap.units}`,
    `- Grid: ${densityLabel(snap.gridDensity)}, boxes ${snap.gridBoxes ? 'on' : 'off'}`,
  ];
  if (snap.viewport) lines.push(`- Viewport: ${scrubInline(snap.viewport, 32)}`);
  return lines.join('\n');
}
