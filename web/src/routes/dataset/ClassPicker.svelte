<script lang="ts">
  /**
   * The image filter's class picker: a toolbar button naming the active class that opens a list of classes with their
   * swatches and box counts. One class at a time (the images endpoint filters by a single class).
   */
  import Check from '@lucide/svelte/icons/check';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import { tick } from 'svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { compact } from '$lib/format';

  let {
    classes,
    counts,
    value,
    onchange,
  }: {
    classes: readonly string[];
    /** Boxes per class, when the dataset has statistics. */
    counts?: readonly number[];
    value: number | undefined;
    onchange: (cls: number | undefined) => void;
  } = $props();

  const uid = $props.id();
  let open = $state(false);
  let root = $state<HTMLElement>();
  let list = $state<HTMLElement>();

  const options = () => [...(list?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])];

  async function show() {
    open = true;
    await tick();
    (list?.querySelector<HTMLElement>('[aria-selected="true"]') ?? options()[0])?.focus();
  }
  function hide(refocus = true) {
    open = false;
    if (refocus) root?.querySelector<HTMLElement>('.trigger')?.focus();
  }
  function pick(cls: number | undefined) {
    hide();
    if (cls !== value) onchange(cls);
  }

  function onkeydown(e: KeyboardEvent) {
    const all = options();
    const i = all.indexOf(document.activeElement as HTMLElement);
    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      all[Math.max(0, Math.min(all.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))]?.focus();
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      all[e.key === 'Home' ? 0 : all.length - 1]?.focus();
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
    return () => document.removeEventListener('pointerdown', away);
  });
</script>

<div class="picker" bind:this={root}>
  <button
    type="button"
    class="trigger"
    class:active={value !== undefined}
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-controls="{uid}-list"
    aria-label="Filter by class: {value === undefined ? 'all classes' : classes[value]}"
    onclick={() => (open ? hide() : show())}
  >
    {#if value !== undefined}
      <span class="sw" style:background={classInk(value)} aria-hidden="true"></span>
      <span class="truncate">{classes[value]}</span>
    {:else}
      All classes
    {/if}
    <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" />
  </button>
  {#if open}
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div class="list" id="{uid}-list" role="listbox" aria-label="Classes" bind:this={list} {onkeydown}>
      <button type="button" role="option" tabindex="-1" aria-selected={value === undefined} onclick={() => pick(undefined)}>
        <span class="check">{#if value === undefined}<Check size={14} strokeWidth={2} />{/if}</span>
        <span class="name">All classes</span>
      </button>
      {#each classes as c, i (i)}
        <button
          type="button"
          role="option"
          tabindex="-1"
          aria-selected={value === i}
          class:absent={counts?.[i] === 0}
          onclick={() => pick(i)}
        >
          <span class="check">{#if value === i}<Check size={14} strokeWidth={2} />{/if}</span>
          <span class="sw" style:background={classInk(i)} aria-hidden="true"></span>
          <span class="name truncate">{c}</span>
          {#if counts}<span class="n num">{compact(counts[i])}</span>{/if}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .picker {
    position: relative;
    display: inline-flex;
    min-width: 0;
  }
  .trigger {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 220px;
    height: var(--control-sm);
    padding: 0 6px 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .trigger:hover,
  .trigger[aria-expanded='true'] {
    color: var(--text-primary);
    border-color: var(--line-strong);
  }
  .trigger.active {
    color: var(--text-primary);
    border-color: var(--text-tertiary);
  }
  .trigger :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .list {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    z-index: var(--z-overlay);
    min-width: 240px;
    max-height: 320px;
    overflow-y: auto;
    padding: var(--space-1);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    display: grid;
    animation: in var(--dur-fast) var(--ease-out);
  }
  [role='option'] {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    text-align: left;
  }
  [role='option']:hover,
  [role='option']:focus-visible {
    background: var(--bg-hover);
    outline: none;
  }
  .check {
    display: grid;
    place-items: center;
    width: 14px;
    flex-shrink: 0;
    color: var(--text-primary);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .absent {
    color: var(--text-tertiary);
  }
  .absent .sw {
    opacity: 0.4;
  }
  .name {
    flex: 1;
    min-width: 0;
  }
  .n {
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .trigger,
    [role='option'] {
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
