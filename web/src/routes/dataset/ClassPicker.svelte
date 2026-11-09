<script lang="ts">
  /**
   * The image filter's class picker: a toolbar button naming the active class that opens a list of classes with their
   * swatches and box counts. One class at a time (the images endpoint filters by a single class). `block` (phones) makes
   * it the row's select: it takes the width its neighbours leave, with ui/Select's face (the raised field, its edge, the
   * value in ink, the one down chevron at its end, 44 px with 16 px text on touch), so the button after it ends on the
   * content edge whatever the class's name.
   */
  import Check from '@lucide/svelte/icons/check';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import { tick } from 'svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { compact } from '$lib/format';
  import { swatchStyle } from './swatch';

  let {
    classes,
    counts,
    value,
    onchange,
    block = false,
  }: {
    classes: readonly string[];
    /** Boxes per class, when the dataset has statistics. */
    counts?: readonly number[];
    value: number | undefined;
    onchange: (cls: number | undefined) => void;
    /** Fill the row as a select does (phones' one-row toolbar). */
    block?: boolean;
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

<div class="picker" class:block bind:this={root}>
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
      <span class="truncate">All classes</span>
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
          <span class="sw" style={swatchStyle(i, counts?.[i] === 0)} aria-hidden="true"></span>
          <span class="name truncate" title={c}>{c}</span>
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
    min-width: min(240px, calc(100vw - 32px));
    /* Never wider than the screen: long class names truncate inside their row instead. */
    max-width: min(360px, calc(100vw - 32px));
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
    min-width: 0;
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
  [role='option']:hover {
    background: var(--bg-hover);
  }
  /* Keyboard focus is the focus ring, inset like ui/Menu's items, never the hover fill alone (§2.9). */
  [role='option']:focus-visible {
    background: var(--bg-hover);
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: -2px;
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
  .name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .n {
    color: var(--text-tertiary);
  }
  /* The row's select (phones): ui/Select's field, the chevron at its far end. */
  .block {
    display: flex;
  }
  .block .trigger {
    flex: 1;
    max-width: none;
    padding: 0 var(--space-2);
    background: var(--bg-raised);
    border-color: var(--line-control);
    color: var(--text-primary);
    font-weight: var(--weight-regular);
  }
  .block .trigger:hover,
  .block .trigger[aria-expanded='true'] {
    border-color: var(--line-control-hover);
  }
  .block .trigger :global(svg) {
    margin-left: auto;
  }
  @media (pointer: coarse) {
    .trigger,
    [role='option'] {
      height: 44px;
    }
    /* A select's value at 16 px touch text, as ui/Select. */
    .block .trigger {
      font-size: var(--type-lg-size);
    }
  }
  @keyframes in {
    from {
      opacity: 0;
      transform: translateY(-2px);
    }
  }
</style>
