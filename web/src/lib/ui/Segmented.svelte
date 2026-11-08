<script lang="ts" generics="T extends string">
  /** A segmented control (radio group). Arrow keys move the selection, as in native radio groups. */
  import type { IconType } from './types';

  interface Option {
    value: T;
    label: string;
    icon?: IconType;
    /** Extra text after the label, e.g. a count. */
    meta?: string;
    /** Shown but not selectable (e.g. "coming soon"). */
    disabled?: boolean;
  }

  let {
    options,
    value = $bindable(),
    label,
    size = 'md',
    iconOnly = false,
    onchange,
  }: { options: Option[]; value: T; label: string; size?: 'sm' | 'md'; iconOnly?: boolean; onchange?: (v: T) => void } =
    $props();

  let root: HTMLDivElement;

  function select(v: T) {
    value = v;
    onchange?.(v);
  }

  function onkeydown(e: KeyboardEvent) {
    const i = options.findIndex((o) => o.value === value);
    const n = options.length;
    // Step over disabled options in the direction of travel.
    const step = (from: number, d: number) => {
      for (let k = 1; k <= n; k++) {
        const j = (((from + d * k) % n) + n) % n;
        if (!options[j]?.disabled) return j;
      }
      return from;
    };
    let j = i;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = step(i, 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = step(i, -1);
    else if (e.key === 'Home') j = step(-1, 1);
    else if (e.key === 'End') j = step(n, -1);
    else return;
    e.preventDefault();
    const next = options[j];
    if (!next || next.disabled) return;
    select(next.value);
    root.querySelectorAll<HTMLButtonElement>('button')[j]?.focus();
  }
</script>

<div class="seg {size}" role="radiogroup" aria-label={label} bind:this={root} {onkeydown} tabindex="-1">
  {#each options as o (o.value)}
    {@const on = o.value === value}
    <button
      type="button"
      role="radio"
      aria-checked={on}
      tabindex={on ? 0 : -1}
      class:on
      class:icon-only={iconOnly && o.icon}
      disabled={o.disabled}
      onclick={() => select(o.value)}
      aria-label={iconOnly ? o.label : undefined}
      title={iconOnly ? o.label : undefined}
    >
      {#if o.icon}<o.icon size={14} strokeWidth={1.75} aria-hidden="true" />{/if}
      {#if !iconOnly || !o.icon}<span>{o.label}</span>{/if}
      {#if o.meta}<span class="meta">{o.meta}</span>{/if}
    </button>
  {/each}
</div>

<style>
  .seg {
    display: inline-flex;
    gap: 2px;
    padding: 2px;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
    max-width: 100%;
    justify-self: start;
    overflow-x: auto;
    scrollbar-width: none;
  }
  :global([data-theme='light']) .seg {
    background: var(--bg-active);
    border-color: transparent;
  }
  .seg:focus {
    outline: none;
  }
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: calc(var(--control-md) - 6px);
    padding: 0 10px;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
  }
  .sm button {
    height: calc(var(--control-sm) - 6px);
    padding: 0 8px;
    font-size: var(--type-sm-size);
  }
  button.icon-only {
    width: calc(var(--control-md) - 6px);
    padding: 0;
  }
  .sm button.icon-only {
    width: calc(var(--control-sm) - 6px);
  }
  button:hover:not(:disabled) {
    color: var(--text-primary);
  }
  button:disabled {
    color: var(--text-disabled);
  }
  button.on {
    background: var(--bg-raised);
    color: var(--text-primary);
    box-shadow:
      0 0 0 1px var(--line-default),
      var(--shadow-sm);
  }
  :global([data-theme='light']) button.on {
    background: var(--bg-panel);
  }
  .meta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  button:focus-visible {
    outline-offset: 0;
  }
</style>
