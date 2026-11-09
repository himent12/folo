<script lang="ts">
  /**
   * A run banner, one line (web/DESIGN_DIRECTION.md §4.4, the Vercel callout): status icon, title, cause, one action,
   * and a disclosure for the details. Only the icon and the title take the status hue; the cause stays in
   * text.secondary. On phones the cause wraps under the title and the actions go under the text; a `stack` callout
   * gives its first action the full width (the one fix to take), the disclosure under it. An `also` line under the
   * first, with its own mark and hue, joins a second finding about the same run (a completed run's outcome, then the
   * trainer's warning that it was too short to learn), so one run never stacks two banners.
   */
  import type { Snippet } from 'svelte';
  import type { IconType } from '$lib/ui/types';

  let {
    tone,
    icon: Icon,
    title,
    role = 'status',
    label,
    stack = false,
    brief = false,
    runIn = false,
    children,
    actions,
    more,
    moreLabel = 'Details',
    also,
  }: {
    tone: 'success' | 'warning' | 'danger' | 'neutral';
    icon: IconType;
    title: string;
    role?: 'status' | 'alert' | 'note';
    label?: string;
    /** Phones: the first action across the callout. */
    stack?: boolean;
    /** Phones: the cause is a summary the page repeats below; it hides, and the action stays on the title's line. */
    brief?: boolean;
    /** Phones too: the cause runs on after the title ("Too short to learn: 125 optimizer steps") instead of under it. */
    runIn?: boolean;
    /** The cause, after the title. */
    children?: Snippet;
    actions?: Snippet;
    /** What the disclosure opens. */
    more?: Snippet;
    moreLabel?: string;
    /** A second line under the first: its own mark and tone, a title and a short cause. The actions serve both. */
    also?: { icon: IconType; tone: 'success' | 'warning' | 'danger' | 'neutral'; title: string; msg?: string };
  } = $props();

  const uid = $props.id();
  let open = $state(false);
</script>

<div class="callout {tone}" class:stack class:brief class:run-in={runIn} class:two={!!also} {role} aria-label={label}>
  <Icon size={16} aria-hidden="true" />
  <p class="c-text">
    <span class="c-title">{title}</span>{#if children}<span class="c-msg">{@render children()}</span>{/if}
  </p>
  {#if also}
    {@const AlsoIcon = also.icon}
    <span class="c-also-mark {also.tone}" aria-hidden="true"><AlsoIcon size={16} /></span>
    <p class="c-text c-also {also.tone}">
      <span class="c-title">{also.title}</span>{#if also.msg}<span class="c-msg">{also.msg}</span>{/if}
    </p>
  {/if}
  {#if actions || more}
    <div class="c-acts">
      {#if actions}{@render actions()}{/if}
      {#if more}
        <button type="button" class="c-more" aria-expanded={open} aria-controls="{uid}-more" onclick={() => (open = !open)}>{moreLabel}</button>
      {/if}
    </div>
  {/if}
  {#if more}
    <div class="c-detail" id="{uid}-more" hidden={!open}>{@render more()}</div>
  {/if}
</div>

<style>
  .callout {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2) var(--space-3);
    min-height: 44px;
    padding: var(--space-2) var(--space-2) var(--space-2) var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
  }
  .callout > :global(svg) {
    flex-shrink: 0;
  }
  /* The status hue on the mark and the title only; the cause reads in text.secondary. */
  .danger > :global(svg),
  .danger > .c-text:not(.c-also) .c-title {
    color: var(--danger-text);
  }
  .success > :global(svg) {
    color: var(--success-text);
  }
  .neutral > :global(svg) {
    color: var(--text-tertiary);
  }
  .warning > :global(svg),
  .warning > .c-text:not(.c-also) .c-title {
    color: var(--warning-text);
  }
  /* The second line's mark and title take its own hue; its cause stays in text.secondary. */
  .c-also-mark {
    display: inline-flex;
    color: var(--text-tertiary);
  }
  .c-also-mark.warning,
  .c-also.warning .c-title {
    color: var(--warning-text);
  }
  .c-also-mark.danger,
  .c-also.danger .c-title {
    color: var(--danger-text);
  }
  .c-also-mark.success {
    color: var(--success-text);
  }
  .c-also.success .c-title,
  .c-also.neutral .c-title {
    color: var(--text-primary);
  }
  /* Two lines: the actions serve both, on the right across them. */
  .two .c-acts {
    grid-column: 3;
    grid-row: 1 / span 2;
  }
  .two .c-detail {
    grid-row: 3;
  }
  .c-text {
    min-width: 0;
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  .c-title {
    margin-right: var(--space-2);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .c-msg :global(.num) {
    color: var(--text-primary);
  }
  .c-acts {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
  }
  /* The disclosure: the system chevron (as in ErrorState) and a quiet label. */
  .c-more {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    height: var(--control-md);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .c-more::before {
    content: '';
    width: 5px;
    height: 5px;
    margin: 0 4px 2px 2px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .c-more[aria-expanded='true']::before {
    margin-bottom: -2px;
    transform: rotate(-135deg);
  }
  .c-more:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .c-detail {
    grid-column: 2 / -1;
    min-width: 0;
    padding-bottom: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .c-detail[hidden] {
    display: none;
  }
  @media (pointer: coarse) {
    .c-more {
      min-height: 44px;
    }
  }
  @media (max-width: 767px) {
    /* Phones: the mark and the title share the first line, the cause wraps under it, the actions under the text. */
    .callout {
      grid-template-columns: auto minmax(0, 1fr);
      align-items: start;
      padding: var(--space-3);
    }
    .callout > :global(svg) {
      margin-top: 2px;
    }
    .c-msg {
      display: block;
      margin-top: 2px;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
    }
    .run-in .c-msg {
      display: inline;
      font-size: inherit;
      line-height: inherit;
    }
    .c-acts {
      grid-column: 2;
      justify-content: flex-start;
      flex-wrap: wrap;
    }
    .c-detail {
      grid-column: 2;
    }
    .two .c-acts,
    .two .c-detail {
      grid-column: 2;
      grid-row: auto;
    }
    .c-also-mark {
      margin-top: 2px;
    }
    /* The one fix to take, across the callout; the disclosure under it. */
    .stack .c-acts {
      grid-column: 1 / -1;
    }
    .stack .c-acts > :global(.btn:first-child) {
      --h: var(--control-xl);
      flex: 1 0 100%;
      font-size: var(--type-base-size);
    }
    .stack .c-detail {
      grid-column: 1 / -1;
    }
    /* A summary the page repeats below: one row, the action beside the title. */
    .brief {
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: center;
      padding-block: var(--space-2);
    }
    .brief > :global(svg) {
      margin-top: 0;
    }
    .brief .c-msg {
      display: none;
    }
    .brief .c-acts {
      grid-column: 3;
    }
  }
</style>
