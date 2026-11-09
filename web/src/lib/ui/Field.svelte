<script lang="ts">
  /**
   * Label + control + hint/error, laid out consistently. The control is the child snippet; pass `for` = its id, and
   * mark the control invalid (`invalid` on Input / Select / Segmented) while `error` is set. The error carries a danger
   * icon as well as its colour, so it is never red text alone. Where a valid value can be derived from what was typed,
   * `fix` offers it as one click after the message ("Try my-dataset"), an inline action in ink.
   */
  import type { Snippet } from 'svelte';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';

  let {
    label,
    for: htmlFor,
    hint,
    error,
    children,
    aside,
    fix,
  }: {
    label: string;
    for?: string;
    /** Helper text; identifiers in backticks are set as code (`folo serve`). */
    hint?: string;
    error?: string | null;
    children: Snippet;
    aside?: Snippet;
    /** A one-click correction shown after the error: `label` is the whole button text ("Try my-dataset"). */
    fix?: { label: string; apply: () => void } | null;
  } = $props();
</script>

<div class="field" class:invalid={!!error}>
  <div class="top">
    <label for={htmlFor}>{label}</label>
    {#if aside}<span class="aside">{@render aside()}</span>{/if}
  </div>
  {@render children()}
  {#if error}
    <p class="msg error" id={htmlFor ? `${htmlFor}-error` : undefined} role="alert">
      <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" /><span
        >{error}{#if fix}{' '}<button type="button" class="fix" onclick={fix.apply}>{fix.label}</button>{/if}</span
      >
    </p>
  {:else if hint}
    <p class="msg" id={htmlFor ? `${htmlFor}-hint` : undefined}>{#each hint.split('`') as text, i (i)}{#if i % 2}<code>{text}</code>{:else}{text}{/if}{/each}</p>
  {/if}
</div>

<style>
  .field {
    display: grid;
    gap: 6px;
    min-width: 0;
  }
  .top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  label {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .aside {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Helper text keeps a readable measure (DESIGN_DIRECTION §2.3: ≤ 60 characters a line). */
  .msg {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .msg code {
    font-family: var(--font-mono);
  }
  .msg.error {
    display: flex;
    align-items: flex-start;
    gap: var(--space-1);
    color: var(--danger-text);
  }
  /* The offered correction: an inline action in ink (never a status hue), underlined so it reads as one. */
  .fix {
    display: inline;
    padding: 0;
    border: 0;
    background: none;
    font: inherit;
    color: var(--text-primary);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
    cursor: pointer;
    white-space: nowrap;
  }
  /* Touch: a 48 px tall target around the 16 px line (DESIGN_DIRECTION §2.9). */
  @media (pointer: coarse) {
    .fix {
      position: relative;
    }
    .fix::after {
      content: '';
      position: absolute;
      inset: -16px -4px;
    }
  }
  @media (hover: hover) {
    .fix:hover {
      text-decoration-color: currentColor;
    }
  }
  /* The icon sits on the first line's centre (16 px line, 14 px icon). */
  .msg.error :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }
</style>
