// @vitest-environment happy-dom
// Tests for Settings › Diagnostics: URL sanitization, report formatting, live /api/system ping latency,
// error handling, and clipboard copy with polite aria-live feedback.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import { gridView } from '$lib/ui/gridView';
import { toast } from '$lib/ui/toast';
import { render } from '../../../framework/testing';
import Settings from '../Settings';
import { densityLabel, formatDiagnosticsReport, formatPing, mergeFeatures, sanitizeServerUrl } from './diagnostics';
import DiagnosticsPanel from './DiagnosticsPanel';

let target: HTMLElement;
let mounted: { unmount(): void } | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse('#/settings');
  hub.mode = 'mock';
  hub.version = '0.4.2';
  hub.features = null;
  hub.serverMock = null;
  prefs.serverUrl = '';
  prefs.theme = 'dark';
  prefs.units = 'binary';
  gridView.density = 'm';
  gridView.boxes = true;
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

describe('diagnostics helpers', () => {
  it('sanitizes URLs by stripping credentials, query tokens, and hash fragments', () => {
    expect(sanitizeServerUrl('https://alice:supersecret@folo.example.com:8443/app/?token=xyz#frag')).toBe(
      'https://folo.example.com:8443/app',
    );
    expect(sanitizeServerUrl('127.0.0.1:7878/')).toBe('http://127.0.0.1:7878');
    expect(sanitizeServerUrl('', 'http://localhost:5173/?secret=1')).toBe('http://localhost:5173');
    expect(sanitizeServerUrl('', '')).toBe('same-origin');
    expect(sanitizeServerUrl('ftp://secret:pass@host/path')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('javascript:alert(1)')).toBe('(invalid URL)');
  });

  it('formats ping durations, densities, and deduplicated sorted features', () => {
    expect(formatPing(null)).toBe('Not measured');
    expect(formatPing(0.3)).toBe('<1 ms');
    expect(formatPing(14.6)).toBe('15 ms');
    expect(formatPing(null, 'Network down')).toBe('Failed (Network down)');
    expect(densityLabel('s')).toBe('Compact (S)');
    expect(densityLabel('m')).toBe('Medium (M)');
    expect(densityLabel('l')).toBe('Large (L)');
    expect(mergeFeatures(['throughput', 'cloud'], ['cloud', 'benchmark', '  '])).toEqual(['benchmark', 'cloud', 'throughput']);
  });

  it('formats a sanitized bug-report block omitting credentials and local paths', () => {
    const report = formatDiagnosticsReport({
      studioVersion: '0.1.0',
      serverVersion: '0.4.2',
      mode: 'live',
      offline: false,
      serverMock: false,
      serverUrl: 'https://user:topsecret@gpu.internal:7878/?api_key=leak',
      origin: 'http://localhost:5173',
      deviceId: 'gpu:0',
      deviceSummary: 'RTX 4080 SUPER · Vulkan',
      features: ['cloud', 'throughput'],
      pingMs: 12.4,
      pingError: null,
      themeMode: 'system',
      resolvedTheme: 'dark',
      themeName: 'Midnight',
      themeId: 'midnight',
      units: 'binary',
      gridDensity: 's',
      gridBoxes: false,
      viewport: '1440×900',
    });
    expect(report).toContain('FOLO Studio Diagnostics');
    expect(report).toContain('- Studio: 0.1.0');
    expect(report).toContain('- Server: folo serve 0.4.2 (live)');
    expect(report).toContain('- Endpoint: https://gpu.internal:7878');
    expect(report).toContain('- Ping (/api/system): 12 ms');
    expect(report).toContain('- Device: RTX 4080 SUPER · Vulkan (gpu:0)');
    expect(report).toContain('- Features: cloud, throughput');
    expect(report).toContain('- Theme: Midnight (midnight, system → dark)');
    expect(report).toContain('- Units: binary');
    expect(report).toContain('- Grid: Compact (S), boxes off');
    expect(report).toContain('- Viewport: 1440×900');
    expect(report).not.toContain('topsecret');
    expect(report).not.toContain('api_key');
    expect(report).not.toContain('leak');
  });
});

describe('DiagnosticsPanel', () => {
  it('renders live system diagnostics in Settings and re-measures latency on Ping', async () => {
    const api = new MockApi('normal', 0);
    const baseSystem = await api.system();
    let call = 0;
    let clock = 100;
    vi.spyOn(performance, 'now').mockImplementation(() => clock);
    vi.spyOn(api, 'system').mockImplementation(async () => {
      call++;
      if (call === 1) {
        clock += 18;
        return baseSystem;
      }
      clock += 7;
      return { ...baseSystem, features: [...(baseSystem.features ?? []), 'custom_diag_flag'] };
    });
    hub.api = api;
    mounted = render(() => <Settings />, target);

    await vi.waitFor(() => {
      flushSync();
      const panel = target.querySelector('#diagnostics');
      if (!panel?.textContent?.includes('18 ms')) throw new Error('waiting for initial ping');
    });

    const panel = target.querySelector<HTMLElement>('#diagnostics')!;
    expect(panel.textContent).toContain('Built-in demo 0.4.2 (mock)');
    expect(panel.textContent).toContain('Medium (M)');

    const pingBtn = [...panel.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Ping')!;
    pingBtn.click();

    await vi.waitFor(() => {
      flushSync();
      if (!panel.textContent?.includes('7 ms')) throw new Error('waiting for manual ping');
    });
    expect(panel.textContent).toContain('custom_diag_flag');
    expect(panel.querySelector('[role="status"]')?.textContent).toContain('7 ms');
  });

  it('shows ping failure cleanly when /api/system rejects', async () => {
    const api = new MockApi('normal', 0);
    vi.spyOn(api, 'system').mockRejectedValue(new ApiError(503, 'unavailable', 'Server busy', null, 'GET /api/system'));
    hub.api = api;
    mounted = render(
      () => <DiagnosticsPanel initialPingMs={null} initialPingSeq={0} studioVersion="0.1.0" />,
      target,
    );
    flushSync();

    const panel = target.querySelector<HTMLElement>('#diagnostics')!;
    expect(panel.textContent).toContain('Not measured');
    const pingBtn = [...panel.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Ping')!;
    pingBtn.click();

    await vi.waitFor(() => {
      flushSync();
      if (!panel.querySelector('.diag-err')) throw new Error('waiting for error');
    });
    expect(panel.querySelector('.diag-err')?.textContent).toContain('Server busy');
    expect(panel.querySelector('[role="status"]')?.textContent).toContain('Server ping failed');
  });

  it('copies sanitized diagnostics to clipboard, announces status, and handles clipboard errors', async () => {
    vi.useFakeTimers();
    try {
      const api = new MockApi('normal', 0);
      const sys = await api.system();
      hub.api = api;
      prefs.serverUrl = 'http://admin:pw@127.0.0.1:7878/?token=secret';
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });

      mounted = render(
        () => <DiagnosticsPanel system={sys} initialPingMs={11.2} initialPingSeq={1} studioVersion="0.1.0" />,
        target,
      );
      flushSync();

      const panel = target.querySelector<HTMLElement>('#diagnostics')!;
      const copyBtn = () => [...panel.querySelectorAll('button')].find((b) => b.textContent?.includes('Copy diagnostics') || b.textContent?.includes('Copied'))!;

      copyBtn().click();
      await vi.waitFor(() => {
        flushSync();
        if (copyBtn().textContent?.trim() !== 'Copied') throw new Error('waiting for Copied');
      });

      expect(writeText).toHaveBeenCalledTimes(1);
      const copiedText = writeText.mock.calls[0]![0] as string;
      expect(copiedText).toContain('FOLO Studio Diagnostics');
      expect(copiedText).toContain('http://127.0.0.1:7878');
      expect(copiedText).not.toContain('pw');
      expect(copiedText).not.toContain('secret');
      expect(panel.querySelector('[role="status"]')?.textContent).toBe('Diagnostics copied to clipboard');

      vi.advanceTimersByTime(2100);
      flushSync();
      expect(copyBtn().textContent?.trim()).toBe('Copy diagnostics');

      // Clipboard rejection shows toast error and announces failure without crashing
      const toastErr = vi.spyOn(toast, 'error').mockImplementation(() => 0);
      writeText.mockRejectedValueOnce(new Error('Permission denied'));
      copyBtn().click();
      await vi.waitFor(() => {
        flushSync();
        if (!toastErr.mock.calls.length) throw new Error('waiting for toast');
      });
      expect(copyBtn().textContent?.trim()).toBe('Copy diagnostics');
      expect(panel.querySelector('[role="status"]')?.textContent).toBe('Could not copy diagnostics');
    } finally {
      vi.useRealTimers();
    }
  });
});
