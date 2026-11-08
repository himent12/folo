<script lang="ts">
  import Search from '@lucide/svelte/icons/search';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Monitor from '@lucide/svelte/icons/monitor';
  import Plus from '@lucide/svelte/icons/plus';
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Settings from '@lucide/svelte/icons/settings';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import Logo from './Logo.svelte';
  import { page } from './crumbs.svelte';

  let { onsearch }: { onsearch: () => void } = $props();

  /** On phones the breadcrumb collapses to its last item; a back button leads to the parent. */
  const parent = $derived(page.crumbs.length >= 2 ? page.crumbs[page.crumbs.length - 2] : undefined);
  const themeIcon = $derived(prefs.theme === 'dark' ? Moon : prefs.theme === 'light' ? Sun : Monitor);
  const themeLabel = $derived(`Theme: ${prefs.theme} (click to change)`);
</script>

<header class="topbar">
  {#if parent}
    <a href={href(parent.path ?? '/')} class="mobile-back" aria-label="Back to {parent.label}"><ChevronLeft size={20} strokeWidth={1.75} /></a>
  {:else}
    <a href={href('/')} class="mobile-logo" aria-label="FOLO Studio home"><Logo wordmark={false} /></a>
  {/if}
  <ol class="crumbs" aria-label="Breadcrumb">
    {#each page.crumbs as c, i (i)}
      {@const last = i === page.crumbs.length - 1}
      <li class:last>
        {#if c.path && !last}
          <a href={href(c.path)}>{c.label}</a>
        {:else}
          <span aria-current={last ? 'page' : undefined}>{c.label}</span>
        {/if}
      </li>
      {#if !last}<li class="sep" aria-hidden="true"><ChevronRight size={14} strokeWidth={1.75} /></li>{/if}
    {/each}
  </ol>

  <div class="actions">
    <button class="search" onclick={onsearch} aria-label="Search and commands" aria-keyshortcuts="Control+K Meta+K">
      <Search size={14} strokeWidth={1.75} aria-hidden="true" />
      <span class="search-label">Search or jump to…</span>
      <span class="search-keys"><Kbd keys="Mod+K" subtle /></span>
    </button>
    <Button variant="ghost" icon={themeIcon} aria-label={themeLabel} title={themeLabel} onclick={() => prefs.cycleTheme()} />
    <span class="mobile-only"><Button variant="ghost" icon={Settings} href={href('/settings')} aria-label="Settings" /></span>
    <!-- A global shortcut, so not lime: each page keeps a single lime primary of its own. -->
    {#if router.route.name !== 'run-new'}<span class="new-run"><Button icon={Plus} href={href('/runs/new')}>New run</Button></span>{/if}
  </div>
</header>

<style>
  .topbar {
    position: sticky;
    top: 0;
    z-index: var(--z-sticky);
    height: var(--size-topbar);
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: 0 var(--space-4) 0 var(--space-6);
    background: color-mix(in srgb, var(--bg-app) 86%, transparent);
    backdrop-filter: blur(12px) saturate(1.4);
    -webkit-backdrop-filter: blur(12px) saturate(1.4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .mobile-logo,
  .mobile-back,
  .mobile-only {
    display: none;
  }
  .crumbs {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
    flex: 1;
    font-weight: var(--weight-medium);
  }
  .crumbs li {
    display: flex;
    align-items: center;
    min-width: 0;
    color: var(--text-tertiary);
  }
  .crumbs li a {
    color: var(--text-secondary);
    border-radius: var(--radius-sm);
    padding: 2px 4px;
    margin: -2px -4px;
    transition: color var(--dur-fast) var(--ease-standard);
  }
  .crumbs li a:hover {
    color: var(--text-primary);
  }
  .crumbs li.last {
    color: var(--text-primary);
  }
  .crumbs li.last span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sep {
    flex-shrink: 0;
  }
  .crumbs li:not(.last):not(.sep) {
    flex-shrink: 0;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .search {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    width: 240px;
    padding: 0 var(--space-1) 0 10px;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-panel);
    color: var(--text-tertiary);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .search:hover {
    border-color: var(--line-strong);
    color: var(--text-secondary);
  }
  .search-label {
    flex: 1;
    text-align: left;
  }

  @media (max-width: 1023px) {
    .search {
      width: var(--control-md);
      padding: 0;
      justify-content: center;
      border-color: transparent;
      background: none;
    }
    .search:hover {
      background: var(--bg-hover);
    }
    .search-label,
    .search-keys {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .topbar {
      padding: 0 var(--space-2) 0 var(--space-4);
      gap: var(--space-3);
    }
    .mobile-logo,
    .mobile-only {
      display: inline-flex;
    }
    .mobile-back {
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      margin-left: calc(-1 * var(--space-3));
      margin-right: calc(-1 * var(--space-2));
      border-radius: var(--radius-md);
      color: var(--text-secondary);
    }
    .mobile-back:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
    /* The page's own title sits right below; the bar keeps only navigation and actions. */
    .crumbs li {
      display: none;
    }
    .new-run {
      display: none;
    }
  }
</style>
