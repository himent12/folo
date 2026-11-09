<script lang="ts" module>
  import type { IconType } from './types';

  export interface MenuItem {
    label: string;
    icon?: IconType;
    href?: string;
    onclick?: () => void;
    danger?: boolean;
  }
</script>

<script lang="ts">
  /** An overflow menu: a "⋯" button that opens a short list of actions (links or buttons). */
  import Ellipsis from '@lucide/svelte/icons/ellipsis';
  import { tick } from 'svelte';
  import Button from './Button.svelte';

  let {
    label,
    items,
    variant = 'secondary',
    fixed = false,
  }: {
    label: string;
    items: MenuItem[];
    variant?: 'secondary' | 'ghost';
    /** A row's menu inside a box that clips (a table that scrolls sideways): the list is placed on the viewport. */
    fixed?: boolean;
  } = $props();

  const uid = $props.id();
  let open = $state(false);
  let root = $state<HTMLElement>();
  let list = $state<HTMLElement>();
  /** `fixed`: where the list sits, under the trigger's right edge. */
  let at = $state<{ top: number; right: number } | null>(null);

  function place() {
    const r = root?.getBoundingClientRect();
    if (r) at = { top: r.bottom + 6, right: Math.max(8, window.innerWidth - r.right) };
  }

  const entries = () => [...(list?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])];

  async function show() {
    if (fixed) place();
    open = true;
    await tick();
    entries()[0]?.focus();
  }
  function hide(refocus = true) {
    open = false;
    if (refocus) root?.querySelector<HTMLElement>('button')?.focus();
  }

  function onkeydown(e: KeyboardEvent) {
    const all = entries();
    const i = all.indexOf(document.activeElement as HTMLElement);
    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const d = e.key === 'ArrowDown' ? 1 : -1;
      all[(i + d + all.length) % all.length]?.focus();
    } else if (e.key === 'Tab') {
      hide(false);
    }
  }

  $effect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      if (root && !root.contains(e.target as Node)) hide(false);
    };
    document.addEventListener('pointerdown', away);
    // A list placed on the viewport would drift from its row: scrolling or resizing closes it.
    const drift = (e: Event) => {
      if (!list?.contains(e.target as Node)) hide(false);
    };
    if (fixed) {
      window.addEventListener('scroll', drift, true);
      window.addEventListener('resize', drift);
    }
    return () => {
      document.removeEventListener('pointerdown', away);
      window.removeEventListener('scroll', drift, true);
      window.removeEventListener('resize', drift);
    };
  });
</script>

<div class="menu" bind:this={root}>
  <Button
    {variant}
    size={variant === 'ghost' ? 'sm' : undefined}
    icon={Ellipsis}
    aria-label={label}
    title={label}
    aria-haspopup="menu"
    aria-expanded={open}
    aria-controls="{uid}-list"
    onclick={() => (open ? hide() : show())}
  />
  {#if open}
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div
      class="list"
      class:fixed
      id="{uid}-list"
      role="menu"
      aria-label={label}
      bind:this={list}
      style:top={fixed && at ? `${at.top}px` : undefined}
      style:right={fixed && at ? `${at.right}px` : undefined}
      {onkeydown}
    >
      {#each items as it (it.label)}
        {#if it.href}
          <a role="menuitem" tabindex="-1" href={it.href} class:danger={it.danger} onclick={() => hide(false)}>
            {#if it.icon}<it.icon size={16} strokeWidth={1.75} aria-hidden="true" />{/if}{it.label}
          </a>
        {:else}
          <button
            role="menuitem"
            tabindex="-1"
            class:danger={it.danger}
            onclick={() => {
              // Focus goes back to the trigger first, so a dialog opened by the action returns focus there on close.
              hide(true);
              it.onclick?.();
            }}
          >
            {#if it.icon}<it.icon size={16} strokeWidth={1.75} aria-hidden="true" />{/if}{it.label}
          </button>
        {/if}
      {/each}
    </div>
  {/if}
</div>

<style>
  .menu {
    position: relative;
    display: inline-flex;
  }
  .list {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: var(--z-overlay);
    min-width: 200px;
    padding: var(--space-1);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    display: grid;
    animation: in var(--dur-fast) var(--ease-out);
  }
  .list.fixed {
    position: fixed;
  }
  [role='menuitem'] {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 36px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    text-align: left;
    white-space: nowrap;
  }
  [role='menuitem'] :global(svg) {
    color: var(--text-tertiary);
  }
  [role='menuitem']:hover {
    background: var(--bg-hover);
  }
  /* Keyboard focus is its own mark (an inset ring in the focus colour), not the faint hover fill. */
  [role='menuitem']:focus-visible {
    background: var(--bg-hover);
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: -2px;
  }
  [role='menuitem'].danger,
  [role='menuitem'].danger :global(svg) {
    color: var(--danger-text);
  }
  @media (pointer: coarse) {
    [role='menuitem'] {
      height: 44px;
    }
  }
  @keyframes in {
    from {
      opacity: 0;
      transform: translateY(-2px);
    }
  }
</style>
