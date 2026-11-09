<script lang="ts">
  import PanelLeftClose from '@lucide/svelte/icons/panel-left-close';
  import PanelLeftOpen from '@lucide/svelte/icons/panel-left-open';
  import type { Snippet } from 'svelte';
  import { router, section, href } from '$lib/router.svelte';
  import { sidebar } from './sidebarState.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
  import Logo from './Logo.svelte';
  import { mainNav, settingsNav, type NavItem } from './nav';
  import { reducedMotion } from '$lib/ui/motion';
  import { tooltip } from '$lib/ui/tooltip';

  let { footer, live }: { footer?: Snippet<[boolean]>; live?: Snippet<[boolean]> } = $props();

  const current = $derived(section(router.route.name));
  // Collapse and expand move the page once: the rail takes its new width at once (so charts re-measure one time, not
  // every frame of a width transition), and the navigation clips open or closed over the page (a clip-path animation,
  // composited). Closing keeps the wide look until its clip has run, then collapses.
  const slim = $derived(sidebar.collapsed);
  let anim = $state<'open' | 'close' | null>(null);
  let seen: boolean | undefined;
  $effect(() => {
    const c = slim;
    if (seen !== undefined && c !== seen) anim = reducedMotion() ? null : c ? 'close' : 'open';
    seen = c;
  });
  const collapsed = $derived(slim && anim !== 'close');
</script>

{#snippet item(n: NavItem)}
  {@const active = current === n.id}
  <li>
    <a
      href={href(n.path)}
      class="item"
      class:active
      aria-current={active ? 'page' : undefined}
      aria-label={collapsed ? n.label : undefined}
      use:tooltip={collapsed ? { text: n.label, keys: keyPrefs.singleKeys ? n.keys : undefined, side: 'right' } : null}
    >
      <n.icon size={16} strokeWidth={1.75} aria-hidden="true" />
      <span class="label">{n.label}</span>
      {#if keyPrefs.singleKeys}<span class="keys"><Kbd keys={n.keys} subtle /></span>{/if}
    </a>
  </li>
{/snippet}

<!-- The rail is the shell's column: it runs the page's full height (its fill and edge reach the bottom of a long page,
     full-page captures included), and the navigation sticks to the viewport inside it. -->
<div class="shell-rail" class:slim class:collapsed class:anim-open={anim === 'open'} class:anim-close={anim === 'close'}>
<nav class="sidebar" aria-label="Main" onanimationend={(e) => e.target === e.currentTarget && (anim = null)}>
  <div class="head">
    <a href={href('/')} class="brand" aria-label="FOLO Studio home"><Logo wordmark={!collapsed} /></a>
  </div>

  <ul class="nav">
    {#each mainNav as n (n.id)}{@render item(n)}{/each}
  </ul>

  {#if live}<div class="live">{@render live(collapsed)}</div>{/if}

  <div class="spacer"></div>

  <ul class="nav">
    {@render item(settingsNav)}
  </ul>
  <div class="foot">
    {#if footer}{@render footer(collapsed)}{/if}
    <button
      class="collapse"
      onclick={() => sidebar.toggle()}
      aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      use:tooltip={{ text: collapsed ? 'Expand sidebar' : 'Collapse sidebar', keys: keyPrefs.singleKeys ? '[' : undefined, side: collapsed ? 'right' : 'top' }}
    >
      {#if collapsed}<PanelLeftOpen size={16} strokeWidth={1.75} />{:else}<PanelLeftClose
          size={16}
          strokeWidth={1.75}
        />{/if}
    </button>
  </div>
</nav>
</div>

<style>
  .shell-rail {
    width: var(--size-sidebar);
    flex-shrink: 0;
    background: var(--bg-app);
    border-right: 1px solid var(--line-subtle);
    z-index: var(--z-sidebar);
  }
  .shell-rail.slim {
    width: var(--size-sidebar-collapsed);
  }
  /* Opening: the wide navigation is revealed from the icon column's edge. Closing: it stays wide over the page while its
     clip closes to the icon column, then takes the collapsed look. */
  .anim-open .sidebar {
    animation: rail-open var(--dur-slow) var(--ease-standard);
  }
  .anim-close .sidebar {
    width: var(--size-sidebar);
    background: var(--bg-app);
    border-right: 1px solid var(--line-subtle);
    animation: rail-close var(--dur-slow) var(--ease-standard) forwards;
  }
  @keyframes rail-open {
    from {
      clip-path: inset(0 calc(100% - var(--size-sidebar-collapsed)) 0 0);
    }
    to {
      clip-path: inset(0);
    }
  }
  @keyframes rail-close {
    from {
      clip-path: inset(0);
    }
    to {
      clip-path: inset(0 calc(100% - var(--size-sidebar-collapsed)) 0 0);
    }
  }
  .sidebar {
    position: sticky;
    top: 0;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    padding: 0 var(--space-2) var(--space-2);
  }
  .head {
    height: var(--size-topbar);
    display: flex;
    align-items: center;
    padding: 0 var(--space-2);
    margin-bottom: var(--space-2);
  }
  .collapsed .head {
    padding: 0 0 0 8px;
  }
  .brand {
    display: inline-flex;
    padding: var(--space-1);
    margin: calc(-1 * var(--space-1));
  }
  .nav {
    display: grid;
    gap: 1px;
  }
  .item {
    position: relative;
    display: flex;
    align-items: center;
    gap: 8px;
    height: var(--control-md);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    overflow: hidden;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .item.active {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .item.active :global(svg) {
    color: var(--accent-text);
  }
  .item :global(svg) {
    flex-shrink: 0;
  }
  .label {
    flex: 1;
    min-width: 0;
  }
  .keys {
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  .item:hover .keys {
    opacity: 1;
  }
  .collapsed .label,
  .collapsed .keys {
    display: none;
  }
  .live {
    margin-top: var(--space-4);
  }
  .spacer {
    flex: 1;
  }
  .foot {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    margin-top: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .collapsed .foot {
    flex-direction: column;
  }
  .collapse {
    display: grid;
    place-items: center;
    width: var(--control-md);
    height: var(--control-md);
    border-radius: var(--radius-md);
    color: var(--text-tertiary);
    margin-left: auto;
    flex-shrink: 0;
  }
  .collapsed .collapse {
    margin: 0 auto;
  }
  .collapse:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
</style>
