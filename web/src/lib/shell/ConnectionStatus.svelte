<script lang="ts">
  /** Where the data comes from: the live server or the built-in mock. Links to Settings. */
  import { hub } from '$lib/api/index.svelte';
  import { href } from '$lib/router.svelte';
  import { tooltip } from '$lib/ui/tooltip';

  let { collapsed }: { collapsed: boolean } = $props();

  const label = $derived(hub.offline ? 'Offline' : hub.mode === 'live' ? 'Connected' : hub.mode === 'mock' ? 'Demo data' : 'Connecting…');
  const detail = $derived(
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
</script>

<a href={href('/settings')} class="conn {hub.offline ? 'offline' : hub.mode}" class:collapsed use:tooltip={{ text: detail, side: collapsed ? 'right' : 'top' }}>
  <span class="dot" aria-hidden="true"></span>
  {#if !collapsed}<span class="label">{label}</span>{/if}
  <span class="sr-only">{detail}</span>
</a>

<style>
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
  .mock .dot {
    background: var(--warning-solid);
    box-shadow: 0 0 0 3px var(--warning-muted);
  }
  .label {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
