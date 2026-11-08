<script lang="ts" generics="T extends string">
  /** Underline tabs for switching views inside a page. Arrow keys move between tabs (WAI-ARIA tabs pattern). */
  interface Tab {
    value: T;
    label: string;
    meta?: string;
  }

  let {
    tabs,
    value = $bindable(),
    label,
    onchange,
    idPrefix = 'tab',
  }: { tabs: Tab[]; value: T; label: string; onchange?: (v: T) => void; idPrefix?: string } = $props();

  let root: HTMLDivElement;

  function select(v: T) {
    value = v;
    onchange?.(v);
  }

  function onkeydown(e: KeyboardEvent) {
    const i = tabs.findIndex((t) => t.value === value);
    let j = i;
    if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = tabs.length - 1;
    else return;
    e.preventDefault();
    const next = tabs[j];
    if (!next) return;
    select(next.value);
    root.querySelectorAll<HTMLButtonElement>('[role=tab]')[j]?.focus();
  }
</script>

<div class="tabs" role="tablist" aria-label={label} bind:this={root} {onkeydown} tabindex="-1">
  {#each tabs as t (t.value)}
    {@const on = t.value === value}
    <button
      type="button"
      role="tab"
      id="{idPrefix}-{t.value}"
      aria-selected={on}
      aria-controls="{idPrefix}-panel"
      tabindex={on ? 0 : -1}
      class:on
      onclick={() => select(t.value)}
    >
      {t.label}{#if t.meta}<span class="meta">{t.meta}</span>{/if}
    </button>
  {/each}
</div>

<style>
  .tabs {
    display: flex;
    gap: var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tabs:focus {
    outline: none;
  }
  button {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 36px;
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    transition: color var(--dur-fast) var(--ease-standard);
  }
  button::after {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: -1px;
    height: 2px;
    border-radius: 2px 2px 0 0;
    background: var(--text-primary);
    transform: scaleX(0);
    transition: transform var(--dur-base) var(--ease-out);
  }
  button:hover {
    color: var(--text-primary);
  }
  button.on {
    color: var(--text-primary);
  }
  button.on::after {
    transform: scaleX(1);
  }
  .meta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  button:focus-visible {
    outline-offset: -2px;
  }
</style>
