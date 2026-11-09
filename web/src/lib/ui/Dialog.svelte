<script lang="ts">
  /**
   * Modal dialog on the native <dialog> element: top layer, focus trap, Escape and an inert page come for free.
   * Clicking the backdrop closes it.
   */
  import X from '@lucide/svelte/icons/x';
  import type { Snippet } from 'svelte';
  import Button from './Button.svelte';
  import { rememberFocus } from './focus';

  let {
    open = $bindable(false),
    title,
    description,
    children,
    footer,
    size = 'md',
    onclose,
  }: {
    open?: boolean;
    title: string;
    description?: string;
    children?: Snippet;
    footer?: Snippet;
    size?: 'sm' | 'md' | 'lg';
    onclose?: () => void;
  } = $props();

  const uid = $props.id();
  let el: HTMLDialogElement;

  // Focus returns to the opener on close (browsers that restore it themselves are left alone).
  let restore: (() => void) | null = null;
  $effect(() => {
    if (open && !el.open) {
      restore = rememberFocus();
      el.showModal();
    } else if (!open && el.open) el.close();
  });

  function onDialogClose() {
    open = false;
    onclose?.();
    restore?.();
    restore = null;
  }

  // A click whose target is the <dialog> itself landed on the backdrop. It must also have started there: a text
  // selection or slider drag that begins inside and is released outside is not a request to close.
  let downOnBackdrop = false;
  function onpointerdown(e: PointerEvent) {
    downOnBackdrop = e.target === el;
  }
  function onclick(e: MouseEvent) {
    if (e.target === el && downOnBackdrop) el.close();
    downOnBackdrop = false;
  }
</script>

<dialog
  bind:this={el}
  class={size}
  aria-labelledby="{uid}-title"
  aria-describedby={description ? `${uid}-desc` : undefined}
  onclose={onDialogClose}
  {onpointerdown}
  {onclick}
>
  <div class="frame">
    <header>
      <div class="titles">
        <h2 id="{uid}-title">{title}</h2>
        {#if description}<p id="{uid}-desc">{description}</p>{/if}
      </div>
      <Button variant="ghost" size="sm" icon={X} aria-label="Close" onclick={() => el.close()} />
    </header>
    {#if children}<div class="body">{@render children()}</div>{/if}
    {#if footer}<footer>{@render footer()}</footer>{/if}
  </div>
</dialog>

<style>
  dialog {
    width: min(480px, calc(100vw - 32px));
    max-height: min(720px, calc(100dvh - 48px));
    margin: auto;
    padding: 0;
    border: 0;
    border-radius: var(--radius-xl);
    background: var(--bg-overlay);
    color: var(--text-primary);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
  }
  dialog.sm {
    width: min(400px, calc(100vw - 32px));
  }
  dialog.lg {
    width: min(720px, calc(100vw - 32px));
  }
  dialog[open] {
    animation: enter var(--dur-slow) var(--ease-out);
  }
  dialog::backdrop {
    background: var(--bg-scrim);
    backdrop-filter: blur(2px);
  }
  dialog[open]::backdrop {
    animation: fade var(--dur-slow) var(--ease-out);
  }
  .frame {
    display: flex;
    flex-direction: column;
    max-height: inherit;
  }
  header {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-4) var(--space-3) var(--space-3) var(--space-5);
  }
  .titles {
    flex: 1;
    display: grid;
    gap: 2px;
    padding-top: 2px;
  }
  h2 {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-semibold);
  }
  header p {
    color: var(--text-secondary);
  }
  .body {
    padding: var(--space-2) var(--space-5) var(--space-5);
    overflow-y: auto;
  }
  footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    background: color-mix(in srgb, var(--bg-overlay), var(--bg-app) 30%);
  }
  @keyframes enter {
    from {
      opacity: 0;
      transform: translateY(6px) scale(0.98);
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  /* Phones: a bottom sheet with full-width actions, clear of the home indicator. */
  @media (max-width: 640px) {
    dialog,
    dialog.sm,
    dialog.lg {
      width: 100%;
      max-width: 100%;
      margin: auto 0 0;
      max-height: calc(100dvh - 24px);
      border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }
    dialog[open] {
      animation: sheet var(--dur-slow) var(--ease-out);
    }
    footer {
      padding-bottom: calc(var(--space-3) + env(safe-area-inset-bottom));
    }
    footer > :global(*) {
      flex: 1;
    }
    .body:last-child {
      padding-bottom: calc(var(--space-5) + env(safe-area-inset-bottom));
    }
  }
  @keyframes sheet {
    from {
      transform: translateY(24px);
      opacity: 0;
    }
  }
</style>
