<script lang="ts">
  /**
   * Modal dialog on the native <dialog> element: top layer, focus trap, Escape and an inert page come for free.
   * Clicking the backdrop closes it. Results of actions taken inside it (a copy, a failed confirm) show in its own status
   * region above the footer (`status`, ui/DialogStatus), never as a toast: the toaster sits outside the modal. What it
   * renders reports there through `feedback()` (ui/feedback.svelte.ts).
   * On phones it is a bottom sheet: a grab handle on its header (`ui/SheetHandle`), which a drag down closes
   * (`ui/sheetDrag`), and the labelled Close as ever.
   */
  import { MediaQuery } from 'svelte/reactivity';
  import { media } from '$lib/design/tokens';
  import X from '@lucide/svelte/icons/x';
  import type { Snippet } from 'svelte';
  import Button from './Button.svelte';
  import DialogStatusRegion from './DialogStatus.svelte';
  import { DialogStatus, provideStatus } from './feedback.svelte';
  import { rememberFocus } from './focus';
  import SheetHandle from './SheetHandle.svelte';
  import { sheetDrag } from './sheetDrag';

  let {
    open = $bindable(false),
    title,
    description,
    children,
    footer,
    size = 'md',
    onclose,
    status = new DialogStatus(),
    class: klass = '',
  }: {
    open?: boolean;
    title: string;
    description?: string;
    children?: Snippet;
    footer?: Snippet;
    size?: 'sm' | 'md' | 'lg';
    onclose?: () => void;
    /** Where results of actions in the dialog are said; pass one to report from the component that owns the dialog. */
    status?: DialogStatus;
    /** Extra classes on the <dialog>, e.g. to set a width of its own. */
    class?: string;
  } = $props();

  const uid = $props.id();
  let el: HTMLDialogElement;
  const sheet = new MediaQuery(media.phone, false);
  // svelte-ignore state_referenced_locally: one region for the dialog's life; children find it through feedback().
  provideStatus(status);

  // Focus returns to the opener on close (browsers that restore it themselves are left alone).
  let restore: (() => void) | null = null;
  $effect(() => {
    if (open && !el.open) {
      restore = rememberFocus();
      el.showModal();
      // The dialog itself takes focus (DESIGN_DIRECTION 9), not its first control: the close button would wear the
      // ring when opened from the keyboard. The first Tab reaches the controls; a component that wants a control
      // focused (a picker's checked option) focuses it after the dialog has opened.
      el.focus();
    } else if (!open && el.open) el.close();
  });

  function onDialogClose() {
    open = false;
    status.clear();
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
  tabindex="-1"
  class="{size} {klass}"
  aria-labelledby="{uid}-title"
  aria-describedby={description ? `${uid}-desc` : undefined}
  onclose={onDialogClose}
  {onpointerdown}
  {onclick}
>
  <div class="frame">
    <header use:sheetDrag={{ sheet: () => el, close: () => el.close(), enabled: () => sheet.current }}>
      {#if sheet.current}<SheetHandle />{/if}
      <div class="titles">
        <h2 id="{uid}-title">{title}</h2>
        {#if description}<p id="{uid}-desc">{description}</p>{/if}
      </div>
      <Button variant="ghost" size="sm" icon={X} aria-label="Close" onclick={() => el.close()} />
    </header>
    {#if children}<div class="body scroll-shade">{@render children()}</div>{/if}
    <DialogStatusRegion {status} class="status" />
    {#if footer}<footer>{@render footer()}</footer>{/if}
  </div>
</dialog>

<style>
  /* The dialog is its own measure (base.css `--measure-dialog`): helper text, notes and body paragraphs inside it run
     to its padding, never a 40ch column in a 440 px one. */
  dialog {
    --measure: var(--measure-dialog);
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
  /* 480 px is the floor wherever the viewport allows: a 13 px body in a narrower box wraps every other word. */
  dialog.sm {
    width: min(480px, calc(100vw - 32px));
  }
  dialog.lg {
    width: min(720px, calc(100vw - 32px));
  }
  /* Focused on open as a container: no ring of its own (its controls have theirs). */
  dialog:focus,
  dialog:focus-visible {
    outline: none;
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
    position: relative;
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
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-secondary);
  }
  /* A body taller than the dialog scrolls, with a shade at each edge that has more behind it (base.css .scroll-shade,
     shared with the command palette's list). */
  .body {
    padding: var(--space-2) var(--space-5) var(--space-5);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
  }
  /* The status region sits between the body and the footer, so a result is in view however far the body scrolls. */
  .frame > :global(.status.on) {
    padding: 0 var(--space-5) var(--space-4);
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
  @media (max-width: 767px) {
    dialog,
    dialog.sm,
    dialog.lg {
      width: 100%;
      max-width: 100%;
      margin: auto 0 0;
      max-height: calc(100dvh - 24px);
      border-radius: var(--radius-xl) var(--radius-xl) 0 0;
      /* A short pull springs back (ui/sheetDrag clears the transform). */
      transition: transform var(--dur-fast) var(--ease-standard);
    }
    dialog[open] {
      animation: sheet var(--dur-slow) var(--ease-out);
    }
    /* The header is the sheet's grab area: a vertical drag moves the sheet, never the page. */
    header {
      touch-action: none;
    }
    footer {
      flex-wrap: wrap;
      padding-bottom: calc(var(--space-3) + env(safe-area-inset-bottom));
    }
    footer > :global(*),
    footer > :global(.gate) > :global(*) {
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
