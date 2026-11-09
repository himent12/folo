<script lang="ts">
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import Info from '@lucide/svelte/icons/info';
  import X from '@lucide/svelte/icons/x';
  import { fly } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import { toast } from './toast.svelte';
  import { motion } from './motion';

  const icons = { success: CircleCheck, error: CircleAlert, info: Info };

  // Toasts never take focus: they announce through the live region, and a click on their buttons keeps focus where it
  // was (mousedown is not allowed to move it). A keyboard user can still Tab in; while focus or the pointer is in the
  // stack nothing auto-dismisses, and closing the toast that holds focus hands it back to where the user came from, so
  // it never falls to <body> (round 17: the Playground lost focus after a toast closed).
  let el: HTMLElement;
  let hovered = false;
  let focused = false;
  let from: HTMLElement | null = null;
  const hold = () => toast.hold(hovered || focused);

  function onfocusin(e: FocusEvent) {
    const prev = e.relatedTarget;
    if (prev instanceof HTMLElement && !el.contains(prev)) from = prev;
    focused = true;
    hold();
  }
  function onfocusout(e: FocusEvent) {
    if (e.relatedTarget instanceof Node && el.contains(e.relatedTarget)) return;
    focused = false;
    hold();
  }

  function close(id: number, run?: () => void) {
    run?.();
    // Only the toast that holds focus hands it back: closing another (by mouse) leaves focus, and the hold, where they
    // are. The action may also have moved focus itself (e.g. opened a page): then it is not ours to place.
    const inside = !!el.querySelector(`[data-toast="${id}"]`)?.contains(document.activeElement);
    toast.dismiss(id);
    if (!inside) return;
    focused = false;
    hold();
    if (from?.isConnected) from.focus({ preventScroll: true });
    from = null;
  }

  const keepFocus = (e: MouseEvent) => e.preventDefault();
</script>

<section
  class="toaster"
  aria-label="Notifications"
  bind:this={el}
  onpointerenter={() => ((hovered = true), hold())}
  onpointerleave={() => ((hovered = false), hold())}
  {onfocusin}
  {onfocusout}
>
  <ol aria-live="polite">
    {#each toast.items as t (t.id)}
      {@const Icon = icons[t.tone]}
      <li class="toast {t.tone}" data-toast={t.id} role={t.tone === 'error' ? 'alert' : 'status'} transition:fly={{ y: 8, duration: motion(160), easing: cubicOut }}>
        <span class="icon"><Icon size={16} strokeWidth={1.75} /></span>
        <div class="text">
          <p class="title">{t.title}</p>
          {#if t.detail}<p class="detail">{t.detail}</p>{/if}
        </div>
        {#if t.action}
          {@const a = t.action}
          <button class="action" onmousedown={keepFocus} onclick={() => close(t.id, a.run)}>{a.label}</button>
        {/if}
        <button class="close" aria-label="Dismiss" onmousedown={keepFocus} onclick={() => close(t.id)}><X size={14} /></button>
      </li>
    {/each}
  </ol>
</section>

<style>
  .toaster {
    position: fixed;
    right: var(--space-4);
    bottom: var(--space-4);
    z-index: var(--z-toast);
    width: min(380px, calc(100vw - 32px));
    pointer-events: none;
  }
  ol {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .toast {
    pointer-events: auto;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: var(--space-3) var(--space-2) var(--space-3) var(--space-3);
    border-radius: var(--radius-lg);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-lg);
  }
  .icon {
    display: flex;
    padding-top: 2px;
    color: var(--text-secondary);
  }
  .success .icon {
    color: var(--success-text);
  }
  .error .icon {
    color: var(--danger-text);
  }
  .info .icon {
    color: var(--info-text);
  }
  .text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .title {
    font-weight: var(--weight-medium);
  }
  .detail {
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
  .action {
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-weight: var(--weight-medium);
    color: var(--accent-text);
  }
  .action:hover {
    background: var(--bg-hover);
  }
  .close {
    display: grid;
    place-items: center;
    width: var(--control-sm);
    height: var(--control-sm);
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .close:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  /* Touch: both buttons are real 44 px boxes side by side (no invisible targets overlapping), pulled into the toast's
     padding so the card keeps its height. */
  @media (pointer: coarse) {
    .toast {
      align-items: center;
    }
    .icon {
      align-self: flex-start;
    }
    .action,
    .close {
      height: 44px;
      margin: calc(-1 * var(--space-2)) 0;
    }
    .action {
      padding: 0 var(--space-3);
    }
    .close {
      width: 44px;
    }
  }
  @media (max-width: 767px) {
    .toaster {
      right: var(--space-4);
      left: var(--space-4);
      width: auto;
      bottom: calc(var(--tabbar-h) + var(--dock-h) + var(--space-4) + env(safe-area-inset-bottom));
    }
  }
</style>
