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
  /**
   * An overflow menu: a "⋯" button that opens a short list of actions (links or buttons). Keyboard (WAI-ARIA menu
   * button): Enter, Space or ArrowDown on the button opens it on the first item (ArrowUp on the last); in the list,
   * the arrows wrap, Home and End go to the ends, typing a letter moves to the next item starting with it (typeahead),
   * Escape closes it back to the button, Tab closes it and moves on.
   * On phones it is the shared bottom sheet (ui/Dialog, as the loss view's options): the items as full-width 44 px
   * rows under the sheet's title, never a popover that lands on the text beside its button (round 24's queued run).
   */
  import Ellipsis from '@lucide/svelte/icons/ellipsis';
  import { tick } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { media } from '$lib/design/tokens';
  import Button from './Button.svelte';
  import Dialog from './Dialog.svelte';

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
  const phone = new MediaQuery(media.phone, false);
  /** `fixed`: where the list sits, under the trigger's right edge. */
  let at = $state<{ top: number; right: number } | null>(null);

  function place() {
    const r = root?.getBoundingClientRect();
    if (r) at = { top: r.bottom + 6, right: Math.max(8, window.innerWidth - r.right) };
  }

  const entries = () => [...(list?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])];

  async function show(at: 'first' | 'last' = 'first') {
    if (fixed) place();
    open = true;
    await tick();
    const all = entries();
    (at === 'first' ? all[0] : all.at(-1))?.focus();
  }
  function ontrigger(e: KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    void show(e.key === 'ArrowDown' ? 'first' : 'last');
  }
  // Typeahead: the letters typed within half a second, matched against the start of each label. A new search starts
  // after the focused item; a word being typed may stay on it; one letter pressed again cycles through its items.
  let typed = '';
  let typedAt = 0;
  function typeahead(key: string, all: HTMLElement[], i: number) {
    const now = Date.now();
    const k = key.toLowerCase();
    typed = now - typedAt < 500 && typed !== k ? typed + k : k;
    typedAt = now;
    const start = typed.length > 1 ? Math.max(0, i) : i + 1;
    const order = [...all.slice(start), ...all.slice(0, start)];
    order.find((el) => el.textContent?.trim().toLowerCase().startsWith(typed))?.focus();
  }
  function hide(refocus = true) {
    open = false;
    if (refocus) root?.querySelector<HTMLElement>('button')?.focus();
  }
  async function choose(it: MenuItem) {
    // Focus goes back to the trigger first, so a dialog opened by the action returns focus there on close. The phone
    // sheet is modal: it closes (and the page leaves inert) before the trigger can take focus.
    hide(true);
    if (phone.current) {
      await tick();
      root?.querySelector<HTMLElement>('button')?.focus();
    }
    it.onclick?.();
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
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      (e.key === 'Home' ? all[0] : all.at(-1))?.focus();
    } else if (e.key === 'Tab') {
      // The phone sheet is modal: Tab moves on to its Close.
      if (!phone.current) hide(false);
    } else if (e.key.length === 1 && e.key !== ' ' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      typeahead(e.key, all, i);
    }
  }

  $effect(() => {
    if (!open || phone.current) return;
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
    onkeydown={ontrigger}
  />
  {#if open && !phone.current}
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
      {@render entriesList()}
    </div>
  {/if}
</div>

{#snippet entriesList()}
  {#each items as it (it.label)}
    {#if it.href}
      <a role="menuitem" tabindex="-1" href={it.href} class:danger={it.danger} onclick={() => hide(false)}>
        {#if it.icon}<it.icon size={16} strokeWidth={1.75} aria-hidden="true" />{/if}{it.label}
      </a>
    {:else}
      <button role="menuitem" tabindex="-1" class:danger={it.danger} onclick={() => choose(it)}>
        {#if it.icon}<it.icon size={16} strokeWidth={1.75} aria-hidden="true" />{/if}{it.label}
      </button>
    {/if}
  {/each}
{/snippet}

<!-- Phones: the shared bottom sheet. Dialog returns focus to the button when it closes. -->
{#if phone.current}
  <Dialog bind:open title={label} size="sm">
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div class="sheet" id="{uid}-list" role="menu" aria-label={label} bind:this={list} {onkeydown}>
      {@render entriesList()}
    </div>
  </Dialog>
{/if}

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
  /* A destructive item is the page's quiet destructive style (ui/Button `danger`): the icon in the danger hue, the
     label in ink; red text is the confirm dialog's alone. */
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
  /* The sheet's items run edge to edge of its body, as the loss view's options do. */
  .sheet {
    display: grid;
    margin: 0 calc(-1 * var(--space-3));
  }
  .sheet [role='menuitem'] {
    height: 44px;
  }
</style>
