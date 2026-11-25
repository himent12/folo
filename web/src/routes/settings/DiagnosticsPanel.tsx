/**
 * Settings › Diagnostics: live server status, `/api/system` round-trip ping latency, active capabilities,
 * Studio theme/density environment, and one-click sanitized diagnostics copy for bug reports.
 */
import Check from '$icons/check';
import Copy from '$icons/copy';
import RefreshCw from '$icons/refresh-cw';
import { computed, css, effect, flush, For, onCleanup, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { SystemInfo } from '$lib/api/types';
import { prefs } from '$lib/prefs';
import Button from '$lib/ui/Button';
import { gridView } from '$lib/ui/gridView';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import { toast } from '$lib/ui/toast';
import {
  densityLabel,
  formatDiagnosticsReport,
  formatPing,
  isSystemInfo,
  mergeFeatures,
  resolveDeviceLabel,
  safePingError,
  sanitizeServerUrl,
  serverSummary,
  type DiagnosticsSnapshot,
} from './diagnostics';
import Row from './Row';

export interface DiagnosticsPanelProps {
  /** Latest `/api/system` payload loaded by Settings, when available. */
  system?: SystemInfo;
  /** Latest `/api/system` error (initial failure or quiet refresh failure), if any. */
  systemError?: unknown;
  /** Round-trip latency (ms) of Settings' latest `/api/system` load. */
  initialPingMs: number | null;
  /** Incremented each time Settings finishes a `/api/system` load or resets on reconnect. */
  initialPingSeq: number;
  /** FOLO Studio package version. */
  studioVersion: string;
}

interface ManualPing {
  seq: number;
  epoch: number;
  ms: number | null;
  error: string | null;
  errorSafe: string | null;
  system?: SystemInfo;
}

function errorMessage(e: unknown): string {
  if (e instanceof Error && e.message.trim()) return e.message.trim();
  return 'Request failed';
}

export default function DiagnosticsPanel(props: DiagnosticsPanelProps) {
  const pinging = signal(false);
  const manual = signal<ManualPing | null>(null);
  /** Latest valid hardware payload observed for the current hub epoch (preserved across quiet refresh failures). */
  const lastKnown = signal<{ epoch: number; system: SystemInfo } | null>(null);
  const copied = signal(false);
  const announcement = signal('');

  let unmounted = false;
  let pingToken = 0;
  let pingCtrl: AbortController | null = null;
  let copyTimer: ReturnType<typeof setTimeout> | undefined;

  function abortManualPing() {
    pingToken++;
    pingCtrl?.abort();
    pingCtrl = null;
    pinging.set(false);
  }

  /** Clears the live region first when repeating identical text so `fw` actually mutates the DOM text node. */
  function announce(message: string) {
    if (announcement.peek() === message) {
      announcement.set('');
      flush();
    }
    announcement.set(message);
  }

  onCleanup(() => {
    unmounted = true;
    abortManualPing();
    clearTimeout(copyTimer);
  });

  // Invalidate any in-flight or previous manual ping whenever the active server connection changes or reconnects.
  let prevEpoch = hub.epoch;
  let prevConnecting = hub.mode === 'connecting';
  effect(() => {
    const ep = hub.epoch;
    const connecting = hub.mode === 'connecting';
    if (ep === prevEpoch && connecting === prevConnecting) return;
    prevEpoch = ep;
    prevConnecting = connecting;
    abortManualPing();
    manual.set(null);
    lastKnown.set(null);
  });

  const activeManual = computed(() => {
    if (hub.mode === 'connecting') return null;
    const m = manual.get();
    return m && m.epoch === hub.epoch && m.seq === props.initialPingSeq ? m : null;
  });

  const validPropSystem = computed(() => (hub.mode !== 'connecting' && isSystemInfo(props.system) ? props.system : undefined));
  const activeSystem = computed(() => {
    if (hub.mode === 'connecting') return undefined;
    const fromManual = activeManual.get()?.system;
    if (fromManual) return fromManual;
    const cached = lastKnown.get();
    if (props.systemError !== undefined && cached && cached.epoch === hub.epoch) return cached.system;
    return validPropSystem.get() ?? (cached && cached.epoch === hub.epoch ? cached.system : undefined);
  });
  const pingMs = computed(() => {
    if (hub.mode === 'connecting') return null;
    const m = activeManual.get();
    return m ? m.ms : props.initialPingMs;
  });
  const pingError = computed(() => {
    if (hub.mode === 'connecting') return null;
    const m = activeManual.get();
    if (m) return m.error;
    if (props.systemError !== undefined) return errorMessage(props.systemError);
    if (props.system !== undefined && !isSystemInfo(props.system)) return 'Invalid /api/system response';
    return null;
  });
  const pingErrorSafe = computed(() => {
    if (hub.mode === 'connecting') return null;
    const m = activeManual.get();
    if (m) return m.errorSafe;
    if (props.systemError !== undefined) return safePingError(props.systemError);
    if (props.system !== undefined && !isSystemInfo(props.system)) return 'Invalid response';
    return null;
  });

  const device = computed(() => resolveDeviceLabel(activeSystem.get()));
  const features = computed(() => mergeFeatures(activeSystem.get()?.features, hub.features));
  const origin = typeof location === 'undefined' ? '' : location.origin;
  const endpoint = computed(() => sanitizeServerUrl(prefs.serverUrl, origin));
  const serverLine = computed(() => serverSummary(hub.mode, hub.version, hub.serverMock, hub.offline));
  const pingText = computed(() => formatPing(pingMs.get(), pingError.get()));

  const themeLine = computed(() => {
    const t = prefs.activeTheme;
    return prefs.theme === 'system' ? `${t.name} (System → ${prefs.resolvedTheme})` : `${t.name} (${prefs.theme})`;
  });
  const envDesc = computed(
    () => `Theme: ${themeLine.get()} · Units: ${prefs.units === 'binary' ? 'Binary' : 'Metric'} · Grid: ${densityLabel(gridView.density)}`,
  );

  function snapshot(): DiagnosticsSnapshot {
    const d = device.peek();
    const viewport =
      typeof window !== 'undefined' && window.innerWidth > 0 && window.innerHeight > 0 ? `${window.innerWidth}×${window.innerHeight}` : null;
    return {
      studioVersion: props.studioVersion,
      serverVersion: hub.version,
      mode: hub.mode,
      offline: hub.offline,
      serverMock: hub.serverMock,
      serverUrl: prefs.serverUrl,
      origin,
      deviceId: d.id,
      deviceSummary: d.summary,
      features: features.peek(),
      pingMs: pingMs.peek(),
      pingErrorSafe: pingErrorSafe.peek(),
      themeMode: prefs.theme,
      resolvedTheme: prefs.resolvedTheme,
      themeName: prefs.activeTheme.name,
      themeId: prefs.activeTheme.id,
      units: prefs.units,
      gridDensity: gridView.density,
      gridBoxes: gridView.boxes,
      viewport,
    };
  }

  async function measurePing() {
    const isConnecting = () => hub.mode === 'connecting';
    if (isConnecting()) return;
    pingCtrl?.abort();
    const ctrl = new AbortController();
    pingCtrl = ctrl;
    const token = ++pingToken;
    const ep = hub.epoch;
    const api = hub.api;
    pinging.set(true);
    const t0 = performance.now();
    try {
      const info = await api.system(ctrl.signal);
      if (unmounted || token !== pingToken || ctrl.signal.aborted || hub.epoch !== ep || isConnecting()) return;
      if (!isSystemInfo(info)) throw new Error('Invalid /api/system response');
      const ms = performance.now() - t0;
      const seq = props.initialPingSeq;
      lastKnown.set({ epoch: ep, system: info });
      manual.set({ seq, epoch: ep, ms, error: null, errorSafe: null, system: info });
      announce(`Server ping ${formatPing(ms)}`);
    } catch (e) {
      if (unmounted || token !== pingToken || ctrl.signal.aborted || hub.epoch !== ep || isConnecting()) return;
      const msg = errorMessage(e);
      const safe = safePingError(e);
      const seq = props.initialPingSeq;
      const prevSystem = manual.peek()?.epoch === ep ? manual.peek()?.system : undefined;
      manual.set({ seq, epoch: ep, ms: null, error: msg, errorSafe: safe, system: prevSystem });
      announce(`Server ping failed: ${safe}`);
    } finally {
      if (!unmounted && token === pingToken) pinging.set(false);
    }
  }

  async function copyReport() {
    const text = formatDiagnosticsReport(snapshot());
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(text);
      if (unmounted) return;
      copied.set(true);
      announce('Diagnostics copied to clipboard');
      clearTimeout(copyTimer);
      copyTimer = setTimeout(() => {
        if (!unmounted) {
          copied.set(false);
          announcement.set('');
        }
      }, 2000);
    } catch (e) {
      if (unmounted) return;
      copied.set(false);
      announce('Could not copy diagnostics');
      toast.error(e, 'Could not copy diagnostics');
    }
  }

  return (
    <Panel title="Diagnostics" id="diagnostics" divided pad="none">
      <Row
        name="Server & latency"
        desc={
          <span class="diag-sub">
            <Meta>
              <span class="diag-meta-item">
                <span class="diag-wrap">{serverLine.get()}</span>
              </span>
              <span class="diag-meta-item">
                <span class="mono endpoint diag-wrap">{endpoint.get()}</span>
              </span>
            </Meta>
          </span>
        }
        control={
          <div class="ping-ctl">
            <span class="diag-val num" class:diag-err={!!pingError.get()}>
              {pingText.get()}
            </span>
            <Button size="sm" variant="ghost" icon={RefreshCw} loading={pinging.get()} disabled={hub.mode === 'connecting'} onclick={measurePing}>
              Ping
            </Button>
          </div>
        }
      />

      <Row
        fact
        name="Compute device"
        desc={<span class="diag-wrap">{device.get().summary}</span>}
        control={<span class="mono diag-val">{device.get().id}</span>}
      />

      <Row
        fact
        name="Server features"
        desc={
          features.get().length > 0 ? (
            <span class="feat-list">
              <For each={features.get()} key={(f) => f}>
                {(f) => <code class="feat">{f()}</code>}
              </For>
            </span>
          ) : (
            'None reported by this server.'
          )
        }
        control={<span class="num diag-val">{features.get().length}</span>}
      />

      <Row
        fact
        name="Studio environment"
        desc={<span class="diag-wrap">{envDesc.get()}</span>}
        control={<span class="mono diag-val">v{props.studioVersion}</span>}
      />

      <Row
        name="Bug report summary"
        desc="Plain-text snapshot of server version, latency, capabilities, theme, and density. Credentials, paths, and raw error text are omitted."
        control={
          <Button size="sm" icon={copied.get() ? Check : Copy} onclick={copyReport}>
            {copied.get() ? 'Copied' : 'Copy diagnostics'}
          </Button>
        }
      />

      <span class="sr-only" role="status" aria-live="polite">
        {announcement.get()}
      </span>
    </Panel>
  );
}

css`
  .diag-sub {
    display: block;
    min-width: 0;
  }
  /* Follow ModelId's <Meta> wrapping pattern: flex item + inner overflow-wrap keeps wrapped lines clear of Meta's 16 px left clip-path. */
  .diag-sub :global(.ui-meta) > .diag-meta-item {
    display: flex;
    max-width: 100%;
    white-space: normal;
  }
  .diag-wrap {
    display: block;
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
  }
  .endpoint {
    color: var(--text-secondary);
  }
  .ping-ctl {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    min-width: 0;
    max-width: 100%;
  }
  .diag-val {
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .diag-err {
    color: var(--danger-text);
  }
  .feat-list {
    display: inline-flex;
    flex-wrap: wrap;
    gap: var(--space-1);
    margin-top: 2px;
    min-width: 0;
    max-width: 100%;
  }
  .feat {
    display: inline-flex;
    align-items: center;
    max-width: 100%;
    padding: 0 var(--space-1-5);
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-subtle);
    background: var(--bg-inset);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    overflow-wrap: anywhere;
  }
`;
