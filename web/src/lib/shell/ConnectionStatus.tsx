/**
 * The sidebar's foot (design/UNIFIED.md §1): one status line, a status dot, where the data comes from (the live server
 * or the built-in mock) and the server's version ("Demo data · v0.1.0"). Links to Settings.
 */
import FlaskConical from '$icons/flask-conical';
import TriangleAlert from '$icons/triangle-alert';
import WifiOff from '$icons/wifi-off';
import { computed, css } from 'fw';
import { hub } from '$lib/api/index';
import { href } from '$lib/router';
import { tooltip } from '$lib/ui/tooltip';

export default function ConnectionStatus(props: { collapsed: boolean }) {
  const demoErr = computed(() => hub.demoError !== null);
  // Prioritize slow demo load over `hub.offline` so a stalled retry (or `?mock=1&scenario=error`) still surfaces it.
  const demoSlow = computed(() => !demoErr.get() && hub.demoSlow);
  const label = computed(() =>
    demoErr.get()
      ? 'Demo failed'
      : demoSlow.get()
        ? 'Loading demo…'
        : hub.offline
          ? hub.refused
            ? 'Not connected'
            : 'Server offline'
          : hub.mode === 'live'
            ? 'Connected'
            : hub.mode === 'mock'
              ? 'Demo data'
              : 'Connecting…',
  );
  const detail = computed(() =>
    demoErr.get()
      ? 'Built-in demo data failed to load. Reload the page to try again.'
      : demoSlow.get()
        ? 'Built-in demo data is taking a while to load. Reload if it does not finish.'
        : hub.offline
          ? hub.refused
            ? `${hub.serverUrl}: ${hub.refused} Retrying…`
            : `No answer from ${hub.serverUrl}. Retrying…`
          : hub.mode === 'live'
            ? `folo serve ${hub.version ?? ''} at ${hub.serverUrl}`
            : hub.mode === 'mock'
              ? 'Built-in demo data (?mock=1)'
              : 'Looking for folo serve…',
  );

  // The version once a server (or the mock) has answered; never while offline, when it would be a guess.
  const version = computed(() => (!demoErr.get() && !demoSlow.get() && !hub.offline && (hub.mode === 'live' || hub.mode === 'mock') && hub.version ? `v${hub.version}` : null));

  // Collapsed, the state is a glyph as well as a colour (offline: no signal), named in the tooltip and the accessible
  // name, so colour is never the only cue. Demo data is a mode, not a degraded state: a neutral dot on the line and a
  // neutral flask in the icon rail, never the warning hue (DESIGN_DIRECTION §2.6).
  return (
    <a
      href={href('/settings')}
      class={`conn ${demoErr.get() || (!demoSlow.get() && hub.offline) ? 'offline' : hub.mode}`}
      class:collapsed={props.collapsed}
      aria-label={props.collapsed ? label.get() : undefined}
      use:tooltip={{ text: props.collapsed ? `${label.get()}. ${detail.get()}` : detail.get(), side: props.collapsed ? 'right' : 'top' }}
    >
      {props.collapsed && demoErr.get() ? (
        <TriangleAlert class="glyph" size={14} strokeWidth={2} aria-hidden="true" />
      ) : props.collapsed && !demoSlow.get() && hub.offline ? (
        <WifiOff class="glyph" size={14} strokeWidth={2} aria-hidden="true" />
      ) : props.collapsed && !demoSlow.get() && hub.mode === 'mock' && !hub.offline ? (
        <FlaskConical class="glyph" size={14} strokeWidth={2} aria-hidden="true" />
      ) : (
        <span class="dot" aria-hidden="true"></span>
      )}
      {!props.collapsed ? (
        <span class="line">
          <span class="label">{label.get()}</span>
          {version.get() ? <span class="ver num">{` · ${version.get()}`}</span> : null}
        </span>
      ) : null}
      <span class="sr-only">{detail.get()}</span>
    </a>
  );
}

css`
  .conn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: var(--control-md);
    padding: 0 8px;
    border-radius: var(--radius-md);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    min-width: 0;
    flex: 1;
  }
  .conn.collapsed {
    flex: none;
    justify-content: center;
    width: var(--control-md);
    padding: 0;
  }
  .conn:hover {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-tertiary);
    flex-shrink: 0;
  }
  .live .dot {
    background: var(--success-solid);
    box-shadow: 0 0 0 3px var(--success-muted);
  }
  .offline .dot {
    background: var(--danger-solid);
    box-shadow: 0 0 0 3px var(--danger-muted);
  }
  .offline :global(.glyph) {
    color: var(--danger-text);
  }
  .mock :global(.glyph) {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .line {
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
`;
