/**
 * Settings › Diagnostics: live server status, `/api/system` round-trip ping latency, active capabilities,
 * Studio theme/density environment, and one-click sanitized diagnostics copy for bug reports.
 */
import Check from '$icons/check';
import Copy from '$icons/copy';
import RefreshCw from '$icons/refresh-cw';
import { computed, css, For, onCleanup, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { SystemInfo } from '$lib/api/types';
import { prefs } from '$lib/prefs';
import Button from '$lib/ui/Button';
import { gridView } from '$lib/ui/gridView';
import Panel from '$lib/ui/Panel';
import { toast } from '$lib/ui/toast';
import {
  densityLabel,
  formatDiagnosticsReport,
  formatPing,
  mergeFeatures,
  resolveDeviceLabel,
  sanitizeServerUrl,
  serverSummary,
  type DiagnosticsSnapshot,
} from './diagnostics';
import Row from './Row';

export interface DiagnosticsPanelProps {
  /** Latest `/api/system` payload loaded by Settings, when available. */
  system?: SystemInfo;
  /** Error from Settings' `/api/system` load, if any. */
  systemError?: unknown;
  /** Round-trip latency (ms) of Settings' latest `/api/system` load. */
  initialPingMs: number | null;
  /** Incremented each time Settings finishes a `/api/system` load, superseding older manual pings. */
  initialPingSeq: number;
  /** FOLO Studio package version. */
  studioVersion: string;
}

interface ManualPing {
  seq: number;
  ms: number | null;
  error: string | null;
  system?: SystemInfo;
}

function errorMessage(e: unknown): string {
  if (e instanceof Error && e.message.trim()) return e.message.trim();
  return 'Request failed';
}

export default function DiagnosticsPanel(props: DiagnosticsPanelProps) {
  const pinging = signal(false);
  const manual = signal<ManualPing | null>(null);
  const copied = signal(false);
  const announcement = signal('');

  let unmounted = false;
  let pingToken = 0;
  let pingCtrl: AbortController | null = null;
  let copyTimer: ReturnType<typeof setTimeout> | undefined;

  onCleanup(() => {
    unmounted = true;
    pingCtrl?.abort();
    pingCtrl = null;
    clearTimeout(copyTimer);
  });

  const activeManual = computed(() => {
    const m = manual.get();
    return m && m.seq === props.initialPingSeq ? m : null;
  });

  const activeSystem = computed(() => activeManual.get()?.system ?? props.system);
  const pingMs = computed(() => {
    const m = activeManual.get();
    return m ? m.ms : props.initialPingMs;
  });
  const pingError = computed(() => {
    const m = activeManual.get();
    if (m) return m.error;
    return props.systemError !== undefined && !props.system ? errorMessage(props.systemError) : null;
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
      pingError: pingError.peek(),
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
    pingCtrl?.abort();
    const ctrl = new AbortController();
    pingCtrl = ctrl;
    const token = ++pingToken;
    const seq = props.initialPingSeq;
    pinging.set(true);
    const t0 = performance.now();
    try {
      const info = await hub.api.system(ctrl.signal);
      if (unmounted || token !== pingToken || ctrl.signal.aborted) return;
      const ms = performance.now() - t0;
      manual.set({ seq, ms, error: null, system: info });
      announcement.set(`Server ping ${formatPing(ms)}`);
    } catch (e) {
      if (unmounted || token !== pingToken || ctrl.signal.aborted) return;
      const msg = errorMessage(e);
      manual.set({ seq, ms: null, error: msg });
      announcement.set(`Server ping failed: ${msg}`);
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
      announcement.set('Diagnostics copied to clipboard');
      clearTimeout(copyTimer);
      copyTimer = setTimeout(() => {
        if (!unmounted) copied.set(false);
      }, 2000);
    } catch (e) {
      if (unmounted) return;
      copied.set(false);
      announcement.set('Could not copy diagnostics');
      toast.error('Could not copy diagnostics', e instanceof Error ? e.message : undefined);
    }
  }

  return (
    <Panel title="Diagnostics" id="diagnostics" divided pad="none">
      <Row
        name="Server & latency"
        desc={
          <span class="diag-sub">
            <span>{serverLine.get()}</span>
            <span class="sep" aria-hidden="true">
              ·
            </span>
            <span class="mono endpoint">{endpoint.get()}</span>
          </span>
        }
        control={
          <>
            <span class="diag-val num" class:diag-err={!!pingError.get()}>
              {pingText.get()}
            </span>
            <Button size="sm" variant="ghost" icon={RefreshCw} loading={pinging.get()} onclick={measurePing}>
              Ping
            </Button>
          </>
        }
      />

      <Row
        fact
        name="Compute device"
        desc={device.get().summary}
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
        desc={envDesc.get()}
        control={<span class="mono diag-val">v{props.studioVersion}</span>}
      />

      <Row
        name="Bug report summary"
        desc="Plain-text snapshot of server version, latency, capabilities, theme, and density. Credentials and tokens are omitted."
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
    display: inline-flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-1-5);
  }
  .sep {
    color: var(--text-tertiary);
  }
  .endpoint {
    overflow-wrap: anywhere;
    color: var(--text-secondary);
  }
  .diag-val {
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
  }
  .feat {
    display: inline-flex;
    align-items: center;
    padding: 0 var(--space-1-5);
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-subtle);
    background: var(--bg-inset);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    white-space: nowrap;
  }
`;
