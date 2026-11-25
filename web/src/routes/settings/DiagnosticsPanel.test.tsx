// @vitest-environment happy-dom
// Tests for Settings › Diagnostics: strict URL sanitization, secret-free report formatting, payload validation,
// live /api/system ping latency, quiet refresh failures, connection switches, and clipboard copy feedback.
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
import {
  densityLabel,
  formatDiagnosticsReport,
  formatPing,
  isSystemInfo,
  mergeFeatures,
  resolveDeviceLabel,
  safePingError,
  sanitizeServerUrl,
} from './diagnostics';
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
  it('sanitizes URLs to validated HTTP(S) origins and rejects local paths, malformed inputs, and opaque origins', () => {
    expect(sanitizeServerUrl('https://alice:supersecret@folo.example.com:8443/users/alice:pw@other/?token=xyz#frag')).toBe(
      'https://folo.example.com:8443',
    );
    expect(sanitizeServerUrl('127.0.0.1:7878/')).toBe('http://127.0.0.1:7878');
    expect(sanitizeServerUrl('[::1]:7878/api')).toBe('http://[::1]:7878');
    expect(sanitizeServerUrl('', 'http://localhost:5173/?secret=1')).toBe('http://localhost:5173');
    expect(sanitizeServerUrl('', 'null')).toBe('same-origin');
    expect(sanitizeServerUrl('', '')).toBe('same-origin');
    expect(sanitizeServerUrl('C:\\Users\\alice\\private-folo')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('C:/Users/alice/private-folo')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('./private-folo')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('../private-folo')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('private-folo/data')).toBe('(invalid URL)');
    expect(sanitizeServerUrl(' https://example.com ')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('file:/home/alice/private-folo')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('/etc/passwd')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('\\\\server\\share\\secret')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('https://user\\@secret.com')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('http://bad host:7878')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('ftp://secret:pass@host/path')).toBe('(invalid URL)');
    expect(sanitizeServerUrl('javascript:alert(1)')).toBe('(invalid URL)');
  });

  it('validates SystemInfo shapes (including nested capabilities) and classifies errors safely without leaking prose', () => {
    expect(isSystemInfo({ status: 'ok' })).toBe(false);
    expect(isSystemInfo({ cpu: { name: 'CPU' }, gpus: [], capabilities: { device: 'cpu' } })).toBe(false);
    expect(isSystemInfo(null)).toBe(false);
    expect(resolveDeviceLabel({ status: 'ok' } as never)).toEqual({ id: '—', summary: 'Not loaded' });
    expect(safePingError(new ApiError(0, 'network', 'Failed to fetch', null, 'GET /api/system'))).toBe('Unreachable');
    expect(safePingError(new ApiError(503, 'unavailable', 'secret path C:/Users/alice/data https://u:p@h/?t=1', null, 'GET /api/system'))).toBe(
      'HTTP 503',
    );
    expect(safePingError(new DOMException('timeout', 'TimeoutError'))).toBe('Timed out');
    expect(safePingError(new Error('Invalid /api/system response'))).toBe('Invalid response');
    expect(safePingError(new Error('raw internal stack'))).toBe('Unreachable');
  });

  it('formats ping durations, densities, and validated deduplicated features', () => {
    expect(formatPing(null)).toBe('Not measured');
    expect(formatPing(0.3)).toBe('<1 ms');
    expect(formatPing(14.6)).toBe('15 ms');
    expect(formatPing(null, 'HTTP 503')).toBe('Failed (HTTP 503)');
    expect(densityLabel('s')).toBe('Compact (S)');
    expect(densityLabel('m')).toBe('Medium (M)');
    expect(densityLabel('l')).toBe('Large (L)');
    expect(mergeFeatures(['throughput', 'cloud', 'https://u:p@host/?t=1'], ['cloud', 'benchmark', '  '])).toEqual([
      'benchmark',
      'cloud',
      'throughput',
    ]);
  });

  it('formats a sanitized bug-report block omitting credentials, paths, and raw error prose', () => {
    const report = formatDiagnosticsReport({
      studioVersion: '0.1.0',
      serverVersion: '0.4.2',
      mode: 'live',
      offline: false,
      serverMock: false,
      serverUrl: 'https://user:topsecret@gpu.internal:7878/users/alice:pw@h/?api_key=leak',
      origin: 'http://localhost:5173',
      deviceId: 'gpu:0',
      deviceSummary: 'RTX 4080 SUPER · Vulkan /home/alice/private-folo token=example-secret',
      features: ['cloud', 'throughput', 'bad feature/secret'],
      pingMs: null,
      pingErrorSafe: 'HTTP 503',
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
    expect(report).toContain('- Ping (/api/system): Failed (HTTP 503)');
    expect(report).toContain('- Device: RTX 4080 SUPER · Vulkan [redacted] [redacted] (gpu:0)');
    expect(report).toContain('- Features: cloud, throughput');
    expect(report).toContain('- Theme: Midnight (midnight, system → dark)');
    expect(report).toContain('- Units: binary');
    expect(report).toContain('- Grid: Compact (S), boxes off');
    expect(report).toContain('- Viewport: 1440×900');
    expect(report).not.toContain('topsecret');
    expect(report).not.toContain('alice:pw');
    expect(report).not.toContain('private-folo');
    expect(report).not.toContain('example-secret');
    expect(report).not.toContain('api_key');
    expect(report).not.toContain('bad feature');
  });
});

describe('DiagnosticsPanel', () => {
  it('renders live system diagnostics in Settings, supports ?section=diagnostics, and re-measures latency on Ping', async () => {
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
    router.route = parse('#/settings?section=diagnostics');
    mounted = render(() => <Settings />, target);

    await vi.waitFor(() => {
      flushSync();
      const panel = target.querySelector('#diagnostics');
      if (!panel?.textContent?.includes('18 ms')) throw new Error('waiting for initial ping');
    });

    const panel = target.querySelector<HTMLElement>('#diagnostics')!;
    expect(panel.textContent).toContain('Built-in demo 0.4.2 (mock)');
    expect(panel.textContent).toContain('Medium (M)');
    expect(target.querySelector('nav.jump a.on')?.getAttribute('href')).toBe('#/settings?section=diagnostics');

    const pingBtn = [...panel.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Ping')!;
    pingBtn.click();

    await vi.waitFor(() => {
      flushSync();
      if (!panel.textContent?.includes('7 ms')) throw new Error('waiting for manual ping');
    });
    expect(panel.textContent).toContain('custom_diag_flag');
    expect(panel.querySelector('[role="status"]')?.textContent).toContain('7 ms');
  });

  it('shows failed measurement on quiet refresh errors while preserving last known device details, and rejects invalid payloads', async () => {
    const api = new MockApi('normal', 0);
    const baseSystem = await api.system();
    let fail = false;
    vi.spyOn(api, 'system').mockImplementation(async () => {
      if (fail) throw new ApiError(503, 'unavailable', 'Secret /home/alice/path down', null, 'GET /api/system');
      return baseSystem;
    });
    hub.api = api;
    mounted = render(() => <Settings />, target);

    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('#diagnostics')?.textContent?.includes('gpu:0')) throw new Error('waiting for device');
    });

    // Simulate a quiet refresh failure (data stays in sys.data, error goes into sys.stale)
    fail = true;
    const panel = target.querySelector<HTMLElement>('#diagnostics')!;
    const pingBtn = [...panel.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Ping')!;
    pingBtn.click();

    await vi.waitFor(() => {
      flushSync();
      if (!panel.querySelector('.diag-err')) throw new Error('waiting for error');
    });
    // Hardware details stay visible; status announces safe HTTP status instead of raw path
    expect(panel.textContent).toContain('gpu:0');
    expect(panel.querySelector('[role="status"]')?.textContent).toBe('Server ping failed: HTTP 503');

    // Invalid JSON payload (including partial capabilities) does not crash and reports Invalid response
    vi.spyOn(api, 'system').mockResolvedValueOnce({ cpu: { name: 'CPU' }, gpus: [], capabilities: { device: 'cpu' } } as never);
    pingBtn.click();
    await vi.waitFor(() => {
      flushSync();
      if (!panel.querySelector('.diag-err')?.textContent?.includes('Invalid /api/system response')) throw new Error('waiting for invalid');
    });

    // Consecutive identical failure still mutates the live region text node
    const live = panel.querySelector('[role="status"]')!;
    const seenTexts: string[] = [];
    const obs = new MutationObserver(() => seenTexts.push(live.textContent ?? ''));
    obs.observe(live, { characterData: true, childList: true, subtree: true });
    vi.spyOn(api, 'system').mockResolvedValueOnce({ status: 'ok' } as never);
    pingBtn.click();
    await vi.waitFor(() => {
      flushSync();
      if (!seenTexts.includes('Server ping failed: Invalid response')) throw new Error('waiting for repeated live mutation');
    });
    obs.disconnect();
  });

  it('preserves manual ping hardware details across a subsequent Resource refresh failure', async () => {
    const api = new MockApi('normal', 0);
    const baseSystem = await api.system();
    let mode: 'fail' | 'ok' = 'fail';
    vi.spyOn(api, 'system').mockImplementation(async () => {
      if (mode === 'fail') throw new ApiError(503, 'unavailable', 'Temporary outage', null, 'GET /api/system');
      return { ...baseSystem, features: ['manual_ping_feature'] };
    });
    hub.api = api;
    mounted = render(() => <Settings />, target);

    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('#system .err')) throw new Error('waiting for initial System error');
    });

    // Manual Ping succeeds while initial Resource failed -> loads hardware details
    mode = 'ok';
    const panel = target.querySelector<HTMLElement>('#diagnostics')!;
    const pingBtn = [...panel.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Ping')!;
    pingBtn.click();
    await vi.waitFor(() => {
      flushSync();
      if (!panel.textContent?.includes('manual_ping_feature')) throw new Error('waiting for manual ping feature');
    });

    // Subsequent Resource retry fails -> ping shows failure while hardware details from manual ping are preserved
    mode = 'fail';
    const retryBtn = [...target.querySelectorAll('#system button')].find((b) => b.textContent?.includes('Retry'))!;
    (retryBtn as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      if (!panel.querySelector('.diag-err')) throw new Error('waiting for retry error');
    });
    expect(panel.textContent).toContain('manual_ping_feature');
    expect(panel.textContent).toContain('gpu:0');
  });

  it('aborts and discards in-flight manual pings when the connection switches', async () => {
    const apiA = new MockApi('normal', 0);
    const sysA = await apiA.system();
    let resolvePing!: (v: typeof sysA) => void;
    let capturedSignal: AbortSignal | undefined;
    apiA.system = ((signal?: AbortSignal) => {
      capturedSignal = signal;
      return new Promise<typeof sysA>((r) => {
        resolvePing = r;
      });
    }) as typeof apiA.system;
    hub.api = apiA;
    mounted = render(() => <DiagnosticsPanel initialPingMs={null} initialPingSeq={0} studioVersion="0.1.0" />, target);
    flushSync();

    const panel = target.querySelector<HTMLElement>('#diagnostics')!;
    const pingBtn = [...panel.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Ping')!;
    pingBtn.click();
    flushSync();
    expect(capturedSignal?.aborted).toBe(false);

    // Switch connection to connecting -> aborts in-flight ping and disables button
    hub.mode = 'connecting';
    flushSync();
    expect(capturedSignal?.aborted).toBe(true);
    expect((pingBtn as HTMLButtonElement).disabled).toBe(true);

    // Late resolution from server A is ignored and never announces
    resolvePing({ ...sysA, features: ['server_a_only'] });
    await Promise.resolve();
    flushSync();
    expect(panel.textContent).not.toContain('server_a_only');
    expect(panel.querySelector('[role="status"]')?.textContent).toBe('');
  });

  it('copies sanitized diagnostics to clipboard, announces status, and passes (err, title) to toast.error on failure', async () => {
    vi.useFakeTimers();
    const prevClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    try {
      const api = new MockApi('normal', 0);
      const sys = await api.system();
      hub.api = api;
      prefs.serverUrl = 'http://admin:pw@127.0.0.1:7878/users/alice:pw@h/?token=secret';
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });

      mounted = render(
        () => (
          <DiagnosticsPanel
            system={sys}
            systemError={new ApiError(503, 'unavailable', 'leak /home/alice/secret https://u:p@h/?t=1', null, 'GET /api/system')}
            initialPingMs={null}
            initialPingSeq={1}
            studioVersion="0.1.0"
          />
        ),
        target,
      );
      flushSync();

      const panel = target.querySelector<HTMLElement>('#diagnostics')!;
      const copyBtn = () =>
        [...panel.querySelectorAll('button')].find((b) => b.textContent?.includes('Copy diagnostics') || b.textContent?.includes('Copied'))!;

      copyBtn().click();
      await vi.waitFor(() => {
        flushSync();
        if (copyBtn().textContent?.trim() !== 'Copied') throw new Error('waiting for Copied');
      });

      expect(writeText).toHaveBeenCalledTimes(1);
      const copiedText = writeText.mock.calls[0]![0] as string;
      expect(copiedText).toContain('FOLO Studio Diagnostics');
      expect(copiedText).toContain('- Endpoint: http://127.0.0.1:7878');
      expect(copiedText).toContain('- Ping (/api/system): Failed (HTTP 503)');
      expect(copiedText).not.toContain('pw');
      expect(copiedText).not.toContain('secret');
      expect(copiedText).not.toContain('/home/alice');
      expect(panel.querySelector('[role="status"]')?.textContent).toBe('Diagnostics copied to clipboard');

      vi.advanceTimersByTime(2100);
      flushSync();
      expect(copyBtn().textContent?.trim()).toBe('Copy diagnostics');

      // Clipboard rejection passes (err, title) to toast.error and announces failure
      const toastErr = vi.spyOn(toast, 'error').mockImplementation(() => 0);
      const err = new Error('Permission denied');
      writeText.mockRejectedValueOnce(err);
      copyBtn().click();
      await vi.waitFor(() => {
        flushSync();
        if (!toastErr.mock.calls.length) throw new Error('waiting for toast');
      });
      expect(toastErr).toHaveBeenCalledWith(err, 'Could not copy diagnostics');
      expect(copyBtn().textContent?.trim()).toBe('Copy diagnostics');
      expect(panel.querySelector('[role="status"]')?.textContent).toBe('Could not copy diagnostics');

      // Missing navigator.clipboard also reports cleanly
      toastErr.mockClear();
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
      copyBtn().click();
      await vi.waitFor(() => {
        flushSync();
        if (!toastErr.mock.calls.length) throw new Error('waiting for missing clipboard toast');
      });
      expect(toastErr.mock.calls[0]![1]).toBe('Could not copy diagnostics');
    } finally {
      if (prevClipboard) Object.defineProperty(navigator, 'clipboard', prevClipboard);
      else delete (navigator as unknown as Record<string, unknown>).clipboard;
      vi.useRealTimers();
    }
  });
});
