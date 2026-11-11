// fw port of ConnectionStatus.svelte.
/** Where the data comes from: the live server or the built-in mock. Links to Settings. */
import FlaskConical from '$icons/flask-conical';
import WifiOff from '$icons/wifi-off';
import { computed, css } from 'fw';
import { hub } from '$lib/api/index';
import { href } from '$lib/router';
import { tooltip } from '$lib/ui/tooltip';

export default function ConnectionStatus(props: { collapsed: boolean }) {
  const label = computed(() =>
    hub.offline ? 'Server offline' : hub.mode === 'live' ? 'Connected' : hub.mode === 'mock' ? 'Demo data' : 'Connecting…',
  );
  const detail = computed(() =>
    hub.offline
      ? `No answer from ${hub.serverUrl}`
      : hub.mode === 'live'
        ? `folo serve ${hub.version ?? ''} at ${hub.serverUrl}`
        : hub.mode === 'mock'
          ? hub.reason === 'unreachable'
            ? `No server at ${hub.serverUrl}; showing the built-in demo data`
            : 'Built-in demo data (?mock=1)'
          : 'Looking for folo serve…',
  );

  // Collapsed, the state is a glyph as well as a colour (offline: no signal), named in the tooltip and the accessible
  // name, so colour is never the only cue. Demo data is a mode, not a degraded state: a neutral flask at every width,
  // never the warning hue (DESIGN_DIRECTION §2.6).
  return (
    <a
      href={href('/settings')}
      class={`conn ${hub.offline ? 'offline' : hub.mode}`}
      class:collapsed={props.collapsed}
      aria-label={props.collapsed ? label.get() : undefined}
      use:tooltip={{ text: props.collapsed ? `${label.get()}. ${detail.get()}` : detail.get(), side: props.collapsed ? 'right' : 'top' }}
    >
      {props.collapsed && hub.offline ? (
        <WifiOff class="glyph" size={14} strokeWidth={2} aria-hidden="true" />
      ) : hub.mode === 'mock' && !hub.offline ? (
        <FlaskConical class="glyph" size={14} strokeWidth={2} aria-hidden="true" />
      ) : (
        <span class="dot" aria-hidden="true"></span>
      )}
      {!props.collapsed ? <span class="label">{label.get()}</span> : null}
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
  .label {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
`;
