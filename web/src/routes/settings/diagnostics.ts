/**
 * Pure formatting and sanitization helpers for Settings › Diagnostics.
 * Produces a secret-free environment & connectivity summary safe to paste into bug reports or support threads.
 */
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
  pingError: string | null;
  themeMode: ThemePref;
  resolvedTheme: 'dark' | 'light';
  themeName: string;
  themeId: string;
  units: Units;
  gridDensity: GridDensity;
  gridBoxes: boolean;
  viewport: string | null;
}

/**
 * Strips credentials (`user:pass@`), query strings (`?token=...`), and hash fragments from a URL so secrets can never
 * leak into a copied diagnostics block. Returns `(invalid URL)` if the value is not a valid http(s) URL.
 */
export function sanitizeServerUrl(raw: string, fallbackOrigin = ''): string {
  const input = raw.trim() || fallbackOrigin.trim();
  if (!input) return 'same-origin';
  const candidate = /^[a-z][a-z0-9+.-]*:\/\//i.test(input) ? input : `http://${input}`;
  try {
    const parsed = new URL(candidate);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return '(invalid URL)';
    const path = parsed.pathname.replace(/\/+$/, '');
    return `${parsed.origin}${path}`;
  } catch {
    return '(invalid URL)';
  }
}

/** Deduplicated, sorted list of optional server capabilities/features from `/api/system` and `/api/health`. */
export function mergeFeatures(systemFeatures?: readonly string[], hubFeatures?: readonly string[] | null): string[] {
  const seen = new Set<string>();
  for (const list of [systemFeatures, hubFeatures]) {
    if (!list) continue;
    for (const item of list) {
      const clean = item.trim();
      if (clean) seen.add(clean);
    }
  }
  return [...seen].sort((a, b) => a.localeCompare(b));
}

/** Human-readable compute device id + summary from `/api/system`. */
export function resolveDeviceLabel(system?: SystemInfo): { id: string; summary: string } {
  if (!system) return { id: '—', summary: 'Not loaded' };
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
  if (mode === 'mock') return `Built-in demo ${version ?? 'unknown'} (mock)`;
  const kind = serverMock ? 'mock server' : 'live';
  const flags = offline ? `${kind}, offline` : kind;
  return `folo serve ${version ?? 'unknown'} (${flags})`;
}

/** Formats a sanitized plain-text diagnostics block suitable for issue reports. */
export function formatDiagnosticsReport(snap: DiagnosticsSnapshot): string {
  const endpoint = sanitizeServerUrl(snap.serverUrl, snap.origin);
  const device = snap.deviceId === '—' ? snap.deviceSummary : `${snap.deviceSummary} (${snap.deviceId})`;
  const features = snap.features.length > 0 ? snap.features.join(', ') : 'none reported';
  const theme =
    snap.themeMode === 'system'
      ? `${snap.themeName} (${snap.themeId}, system → ${snap.resolvedTheme})`
      : `${snap.themeName} (${snap.themeId}, ${snap.themeMode})`;
  const lines = [
    'FOLO Studio Diagnostics',
    `- Studio: ${snap.studioVersion}`,
    `- Server: ${serverSummary(snap.mode, snap.serverVersion, snap.serverMock, snap.offline)}`,
    `- Endpoint: ${endpoint}`,
    `- Ping (/api/system): ${formatPing(snap.pingMs, snap.pingError)}`,
    `- Device: ${device}`,
    `- Features: ${features}`,
    `- Theme: ${theme}`,
    `- Units: ${snap.units}`,
    `- Grid: ${densityLabel(snap.gridDensity)}, boxes ${snap.gridBoxes ? 'on' : 'off'}`,
  ];
  if (snap.viewport) lines.push(`- Viewport: ${snap.viewport}`);
  return lines.join('\n');
}
